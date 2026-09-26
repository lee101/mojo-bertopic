"""Compiled inner loops for the c-TF-IDF / topic-similarity core of BERTopic.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The CSR layouts are the ones SciPy uses: `indptr` is `Int32` of length
`nrows + 1`, `indices` and `data` are `Int32` / `Float64` of length `nnz`.
Nothing here tokenises text or fits a clustering model; the covered surface is
the arithmetic over a fitted c-TF-IDF matrix.
"""

from std.math import abs, fma, sqrt

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def lp(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def better(av: Float64, ac: Int, bv: Float64, bc: Int) -> Bool:
    """Descending by value, ties broken by the smaller column index."""
    if av > bv:
        return True
    if av < bv:
        return False
    return ac < bc


@export("bt_csr_row_sums")
def bt_csr_row_sums(
    indptr_addr: Int, data_addr: Int, nrows: Int, out_addr: Int
) abi("C"):
    """Row totals: the sum of ``data`` over the nonzeros of each row.

    BERTopic's c-TF-IDF regularises with the mean row total
    (``int(X.sum(axis=1).mean())``), so the row totals are the first thing the
    fitted model needs.
    """
    var indptr = ip(indptr_addr)
    var data = fp(data_addr)
    var out = fp(out_addr)
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        var acc = Float64(0.0)
        for p in range(lo, hi):
            acc += data[unsafe_offset=p]
        out[unsafe_offset=i] = acc


@export("bt_csr_col_sums")
def bt_csr_col_sums(
    indptr_addr: Int, indices_addr: Int, data_addr: Int, nrows: Int, ncols: Int,
    out_addr: Int
) abi("C"):
    """Column totals: the sum of ``data`` over every row touching a column.

    This is the document frequency ``df`` of ``ClassTfidfTransformer.fit``. The
    caller hands in a zeroed buffer of length ``ncols``.
    """
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var out = fp(out_addr)
    var nnz = Int(indptr[unsafe_offset=nrows])
    for p in range(nnz):
        out[unsafe_offset=Int(indices[unsafe_offset=p])] += data[unsafe_offset=p]


@export("bt_csr_rownorm2")
def bt_csr_rownorm2(
    indptr_addr: Int, data_addr: Int, nrows: Int, out_addr: Int
) abi("C"):
    """Squared L2 norm of each row."""
    var indptr = ip(indptr_addr)
    var data = fp(data_addr)
    var out = fp(out_addr)
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        var acc = Float64(0.0)
        for p in range(lo, hi):
            var v = data[unsafe_offset=p]
            acc = fma(v, v, acc)
        out[unsafe_offset=i] = acc


@export("bt_csr_ctfidf")
def bt_csr_ctfidf(
    indptr_addr: Int, indices_addr: Int, data_addr: Int, idf_addr: Int,
    nrows: Int, do_sqrt: Int, out_addr: Int
) abi("C"):
    """One fused pass of ``ClassTfidfTransformer.transform``.

    For every row: L1-normalise (dividing by 1 when the row total is zero, as
    scikit-learn's ``inplace_csr_row_normalize_l1`` does), optionally take the
    square root of the normalised value (``reduce_frequent_words``), then scale
    by the idf of the column the entry lives in.
    """
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var idf = fp(idf_addr)
    var out = fp(out_addr)
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        var total = Float64(0.0)
        for p in range(lo, hi):
            total += abs(data[unsafe_offset=p])
        if total == 0.0:
            total = 1.0
        for p in range(lo, hi):
            var v = data[unsafe_offset=p] / total
            if do_sqrt != 0:
                v = sqrt(v)
            out[unsafe_offset=p] = v * idf[unsafe_offset=Int(indices[unsafe_offset=p])]


@export("bt_csr_top_n")
def bt_csr_top_n(
    indptr_addr: Int, indices_addr: Int, data_addr: Int, nrows: Int, topn: Int,
    idx_out_addr: Int, cnt_out_addr: Int
) abi("C"):
    """Per row, the ``topn`` columns with the largest values, best first.

    BERTopic's ``_top_n_idx_sparse`` uses ``np.argpartition``, whose output
    order is unspecified and which pads short rows with ``None``. This kernel
    returns a deterministic answer instead: columns sorted by descending value,
    ties broken by the smaller column index, short rows padded with ``-1``, and
    the true count written to ``cnt_out``.

    While a row is scanned, ``idx_out`` holds offsets into ``data``; a second
    pass turns them into column indices, which is why ``data`` is still read at
    write-out time.
    """
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var idx_out = lp(idx_out_addr)
    var cnt_out = lp(cnt_out_addr)
    if topn <= 0:
        for i in range(nrows):
            cnt_out[unsafe_offset=i] = 0
        return
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        var base = i * topn
        var filled = 0
        for p in range(lo, hi):
            var value = data[unsafe_offset=p]
            var col = Int(indices[unsafe_offset=p])
            var slot = filled
            if slot > topn - 1:
                slot = topn - 1
            if filled == topn:
                var worst = Int(idx_out[unsafe_offset=base + topn - 1])
                if not better(
                    value,
                    col,
                    data[unsafe_offset=worst],
                    Int(indices[unsafe_offset=worst]),
                ):
                    continue
            var pos = slot
            while pos > 0:
                var prev = Int(idx_out[unsafe_offset=base + pos - 1])
                if better(
                    value,
                    col,
                    data[unsafe_offset=prev],
                    Int(indices[unsafe_offset=prev]),
                ):
                    idx_out[unsafe_offset=base + pos] = idx_out[unsafe_offset=base + pos - 1]
                    pos -= 1
                else:
                    break
            idx_out[unsafe_offset=base + pos] = Int64(p)
            if filled < topn:
                filled += 1
        for j in range(filled):
            var p = Int(idx_out[unsafe_offset=base + j])
            idx_out[unsafe_offset=base + j] = Int64(indices[unsafe_offset=p])
        for j in range(filled, topn):
            idx_out[unsafe_offset=base + j] = -1
        cnt_out[unsafe_offset=i] = Int64(filled)


@export("bt_csr_transpose")
def bt_csr_transpose(
    indptr_addr: Int, indices_addr: Int, data_addr: Int, nrows: Int, ncols: Int,
    t_indptr_addr: Int, t_rows_addr: Int, t_data_addr: Int
) abi("C"):
    """Build the CSC companion of a CSR matrix.

    ``t_indptr`` has ``ncols + 1`` entries; for column ``c`` the slice
    ``t_rows[t_indptr[c]:t_indptr[c+1]]`` lists the rows holding a nonzero in
    that column, ascending, and ``t_data`` holds the matching values. The gram
    kernel needs this because ``A @ A.T`` is only cache friendly when one side
    is column major.
    """
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var t_indptr = ip(t_indptr_addr)
    var t_rows = ip(t_rows_addr)
    var t_data = fp(t_data_addr)
    for c in range(ncols + 1):
        t_indptr[unsafe_offset=c] = 0
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        for p in range(lo, hi):
            t_indptr[unsafe_offset=Int(indices[unsafe_offset=p]) + 1] += 1
    for c in range(ncols):
        t_indptr[unsafe_offset=c + 1] += t_indptr[unsafe_offset=c]
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        for p in range(lo, hi):
            var c = Int(indices[unsafe_offset=p])
            var slot = Int(t_indptr[unsafe_offset=c])
            t_rows[unsafe_offset=slot] = Int32(i)
            t_data[unsafe_offset=slot] = data[unsafe_offset=p]
            t_indptr[unsafe_offset=c] = Int32(slot + 1)
    # The fill loop left t_indptr[c] holding the start of column c+1; shift the
    # cursor back so the array is the usual prefix sum again.
    for c in range(ncols, 0, -1):
        t_indptr[unsafe_offset=c] = t_indptr[unsafe_offset=c - 1]
    t_indptr[unsafe_offset=0] = 0


@export("bt_csr_gram")
def bt_csr_gram(
    indptr_addr: Int, indices_addr: Int, data_addr: Int, t_indptr_addr: Int,
    t_rows_addr: Int, t_data_addr: Int, norm2_addr: Int, nrows: Int, out_addr: Int
) abi("C"):
    """Row-wise cosine similarity of a CSR matrix with itself.

    ``out[i, j] = <a_i, a_j> / (|a_i| |a_j|)``, with scikit-learn's zero-norm
    guard. This is the topic-to-topic similarity matrix BERTopic builds for
    hierarchical topic reduction and for ``similarity_matrix()``.
    """
    var indptr = ip(indptr_addr)
    var indices = ip(indices_addr)
    var data = fp(data_addr)
    var t_indptr = ip(t_indptr_addr)
    var t_rows = ip(t_rows_addr)
    var t_data = fp(t_data_addr)
    var norm2 = fp(norm2_addr)
    var out = fp(out_addr)
    for i in range(nrows):
        for j in range(nrows):
            out[unsafe_offset=i * nrows + j] = 0.0
    for i in range(nrows):
        var lo = Int(indptr[unsafe_offset=i])
        var hi = Int(indptr[unsafe_offset=i + 1])
        for p in range(lo, hi):
            var c = Int(indices[unsafe_offset=p])
            var aic = data[unsafe_offset=p]
            var clo = Int(t_indptr[unsafe_offset=c])
            var chi = Int(t_indptr[unsafe_offset=c + 1])
            for q in range(clo, chi):
                var j = Int(t_rows[unsafe_offset=q])
                var slot = i * nrows + j
                out[unsafe_offset=slot] = fma(
                    aic, t_data[unsafe_offset=q], out[unsafe_offset=slot]
                )
    for i in range(nrows):
        for j in range(nrows):
            var slot = i * nrows + j
            var ni = norm2[unsafe_offset=i]
            var nj = norm2[unsafe_offset=j]
            if ni == 0.0 or nj == 0.0:
                out[unsafe_offset=slot] = 0.0
            else:
                out[unsafe_offset=slot] = out[unsafe_offset=slot] / sqrt(ni * nj)


@export("bt_mmr")
def bt_mmr(
    doc_sim_addr: Int, word_sim_addr: Int, flag_addr: Int, nwords: Int,
    diversity: Float64, top_n: Int, out_addr: Int
) abi("C"):
    """Maximal Marginal Relevance selection over precomputed similarities.

    ``doc_sim`` is the ``nwords x 1`` cosine similarity of each candidate word
    with the document and ``word_sim`` the ``nwords x nwords`` word-to-word
    matrix. ``flag`` is an ``nwords``-entry scratch buffer. The first pick is
    the plain ``argmax`` of ``doc_sim``, as upstream initialises it; every pick
    after that maximises
    ``(1 - diversity) * doc_sim - diversity * max(word_sim[selected])``.

    Ties resolve to the lowest candidate index, matching ``np.argmax``.
    """
    var doc_sim = fp(doc_sim_addr)
    var word_sim = fp(word_sim_addr)
    var flag = ip(flag_addr)
    var out = lp(out_addr)
    if nwords <= 0 or top_n <= 0:
        return
    var keep = min(top_n, nwords)
    for i in range(nwords):
        flag[unsafe_offset=i] = 0
    for step in range(keep):
        var best = -1
        var best_score = Float64(0.0)
        for idx in range(nwords):
            if flag[unsafe_offset=idx] != 0:
                continue
            var score = doc_sim[unsafe_offset=idx]
            if step > 0:
                var target = Float64(0.0)
                for s in range(step):
                    var sim = word_sim[unsafe_offset=idx * nwords + Int(out[unsafe_offset=s])]
                    if s == 0 or sim > target:
                        target = sim
                score = (1.0 - diversity) * score - diversity * target
            if best < 0 or score > best_score:
                best = idx
                best_score = score
        if best < 0:
            return
        out[unsafe_offset=step] = Int64(best)
        flag[unsafe_offset=best] = 1
