"""Correctness-gated benchmark for mojo-bertopic.

Every case checks agreement with the real BERTopic (or scikit-learn, which is
what BERTopic calls) before timing, so a regression in the Mojo kernels shows up
as a correctness failure rather than a suspiciously good number.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_bertopic as mbt  # noqa: E402
from bertopic import BERTopic  # noqa: E402
from bertopic.vectorizers import ClassTfidfTransformer  # noqa: E402


def _time(fn, repeats=3):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def counts(n_rows, vocab, per_row, seed=0):
    """A canonical CSR count matrix with roughly `per_row` entries per row."""
    X = sp.random(n_rows, vocab, density=per_row / vocab, format="csr",
                  random_state=seed, dtype=np.float64)
    X.data = np.rint(X.data * 3.0) + 1.0
    X.eliminate_zeros()
    return X


def bench_ctfidf_transform(n_rows=20000, vocab=20000, per_row=50):
    """The c-TF-IDF transform pass against the real ClassTfidfTransformer."""
    X = counts(n_rows, vocab, per_row, seed=1)
    model = ClassTfidfTransformer()
    model.fit(X.copy())
    idf = np.ascontiguousarray(model._idf_diag.diagonal())

    expected = model.transform(X.copy()).tocsr()
    got = mbt.ctfidf(X, idf=idf)
    assert got.nnz == expected.nnz
    assert np.allclose(got.toarray(), expected.toarray(), rtol=1e-11, atol=0.0), \
        "c-TF-IDF mismatch"

    # Re-normalising an l1-normalised row is a no-op, so timing the reference
    # repeatedly does not drift and the comparison stays fair.
    reference = _time(lambda: model.transform(X), 3)
    ours = _time(lambda: mbt.ctfidf_transform(X, idf), 3)
    return f"ctfidf transform nnz={X.nnz}", reference, ours


def bench_gram(n_topics=800, vocab=8000, per_row=120):
    """The topic-to-topic similarity matrix, "cdist" for topic clustering.

    Two references are timed: `sklearn.metrics.pairwise.cosine_similarity`,
    which is what BERTopic calls, and the raw sparse product `X @ X.T`, the
    fastest fair formulation a caller could write by hand.
    """
    X = counts(n_topics, vocab, per_row, seed=2)
    expected = cosine_similarity(X)
    got = mbt.gram(X)
    assert np.allclose(got, expected, rtol=1e-11, atol=1e-13), "gram mismatch"

    def raw_product():
        dense = np.asarray(X.todense())
        norms = np.linalg.norm(dense, axis=1)
        norms[norms == 0.0] = 1.0
        unit = dense / norms[:, None]
        return unit @ unit.T

    reference = _time(lambda: cosine_similarity(X), 3)
    product = _time(raw_product, 3)
    ours = _time(lambda: mbt.gram(X), 3)
    return f"gram {n_topics}x{n_topics}", reference, ours, product


def bench_top_n(n_rows=20000, vocab=20000, per_row=60, n=10):
    """The per-row top-n term selection behind `BERTopic.get_topics`."""
    X = counts(n_rows, vocab, per_row, seed=3)
    dense = np.asarray(X[:64].todense())
    columns, counts_out = mbt.top_n(X, n)
    for row in range(64):
        picked = [int(c) for c in columns[row] if c >= 0]
        values = sorted((dense[row, c] for c in picked), reverse=True)
        assert len(picked) == int(counts_out[row]) == n
        best = np.sort(dense[row])[::-1][:n]
        assert np.allclose(values, best, rtol=0.0, atol=0.0), f"row {row} top-n differs"

    reference = _time(lambda: BERTopic._top_n_idx_sparse(X, n), 3)
    ours_time = _time(lambda: mbt.top_n(X, n), 3)
    return f"top-{n} terms rows={n_rows}", reference, ours_time


def main():
    print(f"{'case':<34}{'reference':>13}{'mojo-bertopic':>17}{'ratio':>9}")
    print("-" * 73)
    for fn in (bench_ctfidf_transform, bench_top_n):
        label, reference, ours = fn()
        ratio = reference / ours if ours else float("nan")
        print(f"{label:<34}{reference * 1e3:>11.2f}ms{ours * 1e3:>15.2f}ms{ratio:>8.2f}x")
    label, reference, ours, product = bench_gram()
    ratio = reference / ours if ours else float("nan")
    print(f"{label:<34}{reference * 1e3:>11.2f}ms{ours * 1e3:>15.2f}ms{ratio:>8.2f}x")
    print(f"{'  (raw X @ X.T reference)':<34}{product * 1e3:>11.2f}ms")


if __name__ == "__main__":
    main()
