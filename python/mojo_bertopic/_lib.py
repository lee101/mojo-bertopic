"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses; ``c_int``
truncates them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-bertopic.so"

_I64 = ctypes.c_int64
_F64 = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.bt_csr_row_sums.restype = None
    lib.bt_csr_row_sums.argtypes = [_I64, _I64, _I64, _I64]
    lib.bt_csr_col_sums.restype = None
    lib.bt_csr_col_sums.argtypes = [_I64, _I64, _I64, _I64, _I64, _I64]
    lib.bt_csr_rownorm2.restype = None
    lib.bt_csr_rownorm2.argtypes = [_I64, _I64, _I64, _I64]
    lib.bt_csr_ctfidf.restype = None
    lib.bt_csr_ctfidf.argtypes = [_I64] * 4 + [_I64, _I64, _I64]
    lib.bt_csr_top_n.restype = None
    lib.bt_csr_top_n.argtypes = [_I64] * 5 + [_I64, _I64]
    lib.bt_csr_transpose.restype = None
    lib.bt_csr_transpose.argtypes = [_I64] * 8
    lib.bt_csr_gram.restype = None
    lib.bt_csr_gram.argtypes = [_I64] * 9
    lib.bt_mmr.restype = None
    lib.bt_mmr.argtypes = [_I64, _I64, _I64, _I64, _F64, _I64, _I64]
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


def _csr_parts(x) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Unpack a SciPy CSR matrix into the exact dtypes the kernel expects.

    The kernel reads `indptr`/`indices` as Int32 and `data` as Float64, which is
    what a CSR matrix of float64 counts already is; anything else is converted,
    because the C ABI has no way to express a different dtype.
    """
    if hasattr(x, "tocsr"):
        x = x.tocsr()
    for field in ("indptr", "indices", "data", "shape"):
        if not hasattr(x, field):
            raise TypeError(
                f"expected a scipy CSR matrix, got {type(x).__name__}"
            )
    indptr = np.ascontiguousarray(x.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(x.indices, dtype=np.int32)
    data = np.ascontiguousarray(x.data, dtype=np.float64)
    nrows, ncols = (int(v) for v in x.shape)
    if indptr.size != nrows + 1:
        raise ValueError("indptr length does not match the row count")
    if indices.size != data.size:
        raise ValueError("indices and data must have the same length")
    return indptr, indices, data, nrows, ncols


def row_sums(x) -> np.ndarray:
    """``x.sum(axis=1)`` for a CSR matrix."""
    indptr, _, data, nrows, _ = _csr_parts(x)
    out = np.empty(nrows, dtype=np.float64)
    lib.bt_csr_row_sums(_addr(indptr), _addr(data), nrows, _addr(out))
    return out


def col_sums(x) -> np.ndarray:
    """``x.sum(axis=0)`` for a CSR matrix, returned dense."""
    indptr, indices, data, nrows, ncols = _csr_parts(x)
    out = np.zeros(ncols, dtype=np.float64)
    if nrows:
        lib.bt_csr_col_sums(
            _addr(indptr), _addr(indices), _addr(data), nrows, ncols, _addr(out)
        )
    return out


def rownorm2(x) -> np.ndarray:
    """Squared L2 norm of every row, without materialising the dense matrix."""
    indptr, _, data, nrows, _ = _csr_parts(x)
    out = np.empty(nrows, dtype=np.float64)
    lib.bt_csr_rownorm2(_addr(indptr), _addr(data), nrows, _addr(out))
    return out


def ctfidf_transform(x, idf, reduce_frequent_words: bool = False) -> np.ndarray:
    """The data array of ``ClassTfidfTransformer.transform`` for a fitted idf.

    Returns a new float64 array of the same shape as ``x.data``; the caller is
    responsible for rebuilding a sparse matrix with the untouched
    ``indptr``/``indices``.
    """
    indptr, indices, data, nrows, _ = _csr_parts(x)
    idf = np.ascontiguousarray(idf, dtype=np.float64)
    out = np.empty_like(data)
    lib.bt_csr_ctfidf(
        _addr(indptr),
        _addr(indices),
        _addr(data),
        _addr(idf),
        nrows,
        1 if reduce_frequent_words else 0,
        _addr(out),
    )
    return out


def top_n(x, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the ``n`` largest entries, best first.

    Returns ``(columns, counts)`` where ``columns`` is ``(nrows, n)`` of column
    indices padded with ``-1`` and ``counts`` is the true number of entries
    kept per row. Values are descending, ties broken by the smaller column.
    """
    indptr, indices, data, nrows, _ = _csr_parts(x)
    if n <= 0:
        return (
            np.zeros((nrows, 0), dtype=np.int64),
            np.zeros(nrows, dtype=np.int64),
        )
    columns = np.empty((nrows, n), dtype=np.int64)
    counts = np.empty(nrows, dtype=np.int64)
    lib.bt_csr_top_n(
        _addr(indptr), _addr(indices), _addr(data), nrows, n, _addr(columns), _addr(counts)
    )
    return columns, counts


def transpose(x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSC row indices and values of a CSR matrix: ``(t_indptr, t_rows, t_data)``."""
    indptr, indices, data, nrows, ncols = _csr_parts(x)
    t_indptr = np.empty(ncols + 1, dtype=np.int32)
    t_rows = np.empty(indices.size, dtype=np.int32)
    t_data = np.empty(indices.size, dtype=np.float64)
    if indices.size:
        lib.bt_csr_transpose(
            _addr(indptr),
            _addr(indices),
            _addr(data),
            nrows,
            ncols,
            _addr(t_indptr),
            _addr(t_rows),
            _addr(t_data),
        )
    else:
        t_indptr[:] = 0
    return t_indptr, t_rows, t_data


def gram(x) -> np.ndarray:
    """Row-wise cosine similarity of a CSR matrix with itself.

    This is the topic-to-topic similarity matrix: element ``[i, j]`` is
    ``cosine_similarity(c_tf_idf_[i], c_tf_idf_[j])``.
    """
    indptr, indices, data, nrows, ncols = _csr_parts(x)
    out = np.zeros((nrows, nrows), dtype=np.float64)
    if nrows == 0:
        return out
    norm2 = rownorm2(x)
    t_indptr, t_rows, t_data = transpose(x)
    if data.size == 0:
        return out
    lib.bt_csr_gram(
        _addr(indptr),
        _addr(indices),
        _addr(data),
        _addr(t_indptr),
        _addr(t_rows),
        _addr(t_data),
        _addr(norm2),
        nrows,
        _addr(out),
    )
    return out


def mmr(doc_sim, word_sim, diversity: float = 0.1, top_n: int = 10) -> np.ndarray:
    """Maximal Marginal Relevance pick, from precomputed similarities.

    ``doc_sim`` is ``(nwords, ndocs)`` and ``word_sim`` is
    ``(nwords, nwords)``, both cosine similarities. Returns the selected
    indices, best first, as far as BERTopic's own loop would take them.
    """
    doc_sim = np.ascontiguousarray(doc_sim, dtype=np.float64)
    word_sim = np.ascontiguousarray(word_sim, dtype=np.float64)
    if doc_sim.ndim != 2 or doc_sim.shape[1] != 1:
        raise ValueError("doc_sim must have shape (nwords, 1)")
    if word_sim.ndim != 2 or word_sim.shape[0] != word_sim.shape[1]:
        raise ValueError("word_sim must be square")
    nwords = doc_sim.shape[0]
    if word_sim.shape[0] != nwords:
        raise ValueError("doc_sim and word_sim disagree on the word count")
    keep = min(max(top_n, 0), nwords)
    out = np.zeros(max(keep, 0), dtype=np.int64)
    if keep == 0:
        return out
    flag = np.zeros(nwords, dtype=np.int32)
    lib.bt_mmr(
        _addr(doc_sim),
        _addr(word_sim),
        _addr(flag),
        nwords,
        ctypes.c_double(float(diversity)),
        keep,
        _addr(out),
    )
    return out
