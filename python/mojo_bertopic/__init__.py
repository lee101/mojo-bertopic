"""mojo-bertopic: the c-TF-IDF / topic-similarity core of BERTopic in Mojo.

The compiled kernels live in ``src/kernels.mojo`` behind a C ABI; this package
owns every array and calls in through ctypes. Nothing here imports ``bertopic``
or ``sklearn`` at import time, so it installs alongside both.

The covered surface is the arithmetic BERTopic performs on a fitted c-TF-IDF
matrix:

* the ``ClassTfidfTransformer.fit`` statistics (document frequency, average
  row length, the idf vector, BM25 variant included);
* the ``ClassTfidfTransformer.transform`` pass (L1 normalise, optional square
  root, idf scaling);
* the per-row top-``n`` term selection behind ``_top_n_idx_sparse``;
* the topic-to-topic cosine similarity matrix, i.e. the "cdist" used by
  ``similarity_matrix`` and hierarchical topic reduction;
* the Maximal Marginal Relevance selection loop of the ``mmr`` representation.

Everything else -- tokenisation, the CountVectorizer, UMAP, HDBSCAN, the
embedding backends and all plotting -- is left to the real ``bertopic``.
"""

from __future__ import annotations

import numpy as np

from . import _lib

__all__ = [
    "col_sums",
    "ctfidf",
    "ctfidf_transform",
    "fit_idf",
    "gram",
    "mmr",
    "row_sums",
    "rownorm2",
    "top_n",
    "top_n_terms",
    "transpose",
]

row_sums = _lib.row_sums
col_sums = _lib.col_sums
rownorm2 = _lib.rownorm2
top_n = _lib.top_n
transpose = _lib.transpose
ctfidf_transform = _lib.ctfidf_transform
gram = _lib.gram


def _as_csr(x):
    if not hasattr(x, "indptr"):
        raise TypeError(f"expected a scipy CSR matrix, got {type(x).__name__}")
    return x if hasattr(x, "tocsr") else x.tocsr()


def fit_idf(X, bm25_weighting: bool = False, multiplier=None) -> np.ndarray:
    """The idf vector of ``bertopic.vectorizers.ClassTfidfTransformer.fit``.

    ``df`` is the column sum of the count matrix and ``avg_nr_samples`` the
    truncated mean row total, exactly as upstream computes them.
    """
    df = col_sums(X)
    avg_nr_samples = int(row_sums(X).mean())
    if bm25_weighting:
        idf = np.log(1 + ((avg_nr_samples - df + 0.5) / (df + 0.5)))
    else:
        idf = np.log((avg_nr_samples / df) + 1)
    if multiplier is not None:
        idf = idf * np.asarray(multiplier, dtype=np.float64)
    return np.ascontiguousarray(idf, dtype=np.float64)


def ctfidf(X, idf=None, reduce_frequent_words: bool = False,
           bm25_weighting: bool = False, multiplier=None):
    """The c-TF-IDF matrix of ``ClassTfidfTransformer``, computed in Mojo.

    Pass a fitted ``idf`` to transform without refitting, exactly as
    ``transform`` does upstream. Returns a ``scipy.sparse.csr_matrix`` with the
    same sparsity pattern and the same values as the real transformer.
    """
    import scipy.sparse as sp

    pattern = _as_csr(X)
    if idf is None:
        idf = fit_idf(pattern, bm25_weighting=bm25_weighting, multiplier=multiplier)
    out_data = ctfidf_transform(pattern, idf, reduce_frequent_words)
    return sp.csr_matrix(
        (out_data, pattern.indices.copy(), pattern.indptr.copy()), shape=pattern.shape
    )


def top_n_terms(X, n: int = 10):
    """Per row, the ``n`` largest terms, best first, as ``(column, score)``.

    This is the arithmetic behind ``BERTopic.get_topics``: upstream picks the
    columns with ``np.argpartition`` and then reads their scores back out of the
    sparse matrix, in an order ``argpartition`` does not define. This returns
    the same terms in a deterministic, score-descending order.
    """
    pattern = _as_csr(X)
    indptr = np.asarray(pattern.indptr)
    indices = np.asarray(pattern.indices)
    data = np.asarray(pattern.data)
    columns, counts = top_n(pattern, n)
    result = []
    for row in range(columns.shape[0]):
        lo = int(indptr[row])
        hi = int(indptr[row + 1])
        picked = []
        for j in range(int(counts[row])):
            col = int(columns[row, j])
            # A canonical CSR stores each row's columns in ascending order.
            pos = lo + int(np.searchsorted(indices[lo:hi], col))
            picked.append((col, float(data[pos])))
        result.append(picked)
    return result


def mmr(doc_sim, word_sim, diversity: float = 0.1, top_n: int = 10) -> np.ndarray:
    """MMR keyword selection over precomputed cosine similarities.

    Mirrors ``bertopic.representation._mmr.mmr``, but operates on the index
    level: ``doc_sim`` is ``(nwords, ndocs)`` and ``word_sim`` is
    ``(nwords, nwords)``.
    """
    return _lib.mmr(doc_sim, word_sim, diversity=diversity, top_n=top_n)
