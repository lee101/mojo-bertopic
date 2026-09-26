"""Parity tests for the c-TF-IDF kernels against the real BERTopic."""

import numpy as np
import pytest
import scipy.sparse as sp

import mojo_bertopic as mbt
from bertopic.vectorizers import ClassTfidfTransformer


def make_counts(seed=0, n_topics=7, vocab=61, density=0.45):
    """A count matrix that looks like documents-per-topic bags of words.

    Row totals are deliberately varied so the truncated mean row length is far
    from an integer: `fit` uses `int(X.sum(axis=1).mean())` on both sides, and a
    mean sitting on an integer boundary would make the two truncations disagree
    for reasons that have nothing to do with the kernel. Every column is given
    at least one count so `df` never divides by zero.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_topics):
        heavy = rng.integers(0, 3, size=vocab // 3)
        tail = (rng.random(vocab - vocab // 3) < density * 0.2).astype(np.int64)
        rows.append(np.concatenate([heavy, tail]))
    dense = np.asarray(rows, dtype=np.float64)
    for k, col in enumerate(np.flatnonzero(dense.sum(axis=0) == 0.0)):
        dense[k % n_topics, col] = 1.0 + (k % 3)
    return sp.csr_matrix(dense)


def assert_same_matrix(got, expected, rtol=1e-12):
    """Compare two CSR matrices by value, not by storage order.

    Upstream's `X * idf_diag` leaves each row's nonzeros in descending column
    order while the input is canonical ascending, so the `indices` arrays differ
    legitimately even when the matrices are identical.
    """
    assert got.shape == expected.shape
    assert got.nnz == expected.nnz
    np.testing.assert_allclose(got.toarray(), expected.toarray(), rtol=rtol, atol=0.0)


def test_row_and_col_sums_match_scipy():
    X = make_counts()
    np.testing.assert_allclose(mbt.row_sums(X), np.asarray(X.sum(axis=1)).ravel(),
                               rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(mbt.col_sums(X), np.asarray(X.sum(axis=0)).ravel(),
                               rtol=1e-13, atol=0.0)
    # A column present in exactly one row is what a wrong indptr stride or a
    # dropped tail would corrupt.
    Y = sp.csr_matrix(np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(mbt.col_sums(Y), np.array([4.0, 2.0, 3.0]))
    np.testing.assert_array_equal(mbt.row_sums(Y), np.array([2.0, 3.0, 4.0]))


def test_row_sums_of_an_all_zero_matrix():
    Y = sp.csr_matrix((5, 3), dtype=np.float64)
    np.testing.assert_array_equal(mbt.row_sums(Y), np.zeros(5))
    np.testing.assert_array_equal(mbt.col_sums(Y), np.zeros(3))
    np.testing.assert_array_equal(mbt.rownorm2(Y), np.zeros(5))


def test_rownorm2_matches_dense():
    X = make_counts(seed=3)
    dense = np.asarray(X.todense())
    np.testing.assert_allclose(mbt.rownorm2(X), (dense * dense).sum(axis=1),
                               rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("bm25", [False, True])
def test_fit_idf_matches_upstream(bm25):
    X = make_counts(seed=1)
    theirs = ClassTfidfTransformer(bm25_weighting=bm25).fit(X)
    idf = theirs._idf_diag.diagonal()
    np.testing.assert_allclose(mbt.fit_idf(X, bm25_weighting=bm25), idf,
                               rtol=1e-12, atol=0e-12)


def test_fit_idf_multiplier_matches_upstream():
    X = make_counts(seed=2)
    rng = np.random.default_rng(5)
    multiplier = np.where(rng.random(X.shape[1]) < 0.1, 2.0, 1.0)
    theirs = ClassTfidfTransformer().fit(X, multiplier=multiplier)
    idf = theirs._idf_diag.diagonal()
    np.testing.assert_allclose(mbt.fit_idf(X, multiplier=multiplier), idf,
                               rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("reduce_frequent_words", [False, True])
def test_ctfidf_transform_matches_upstream(reduce_frequent_words):
    X = make_counts(seed=4)
    # Upstream's transform normalises the caller's matrix in place, so each side
    # gets its own copy; see test_ctfidf_does_not_mutate_its_input.
    model = ClassTfidfTransformer(reduce_frequent_words=reduce_frequent_words)
    model.fit(X.copy())
    expected = model.transform(X.copy()).tocsr()
    got = mbt.ctfidf(X, idf=model._idf_diag.diagonal(),
                     reduce_frequent_words=reduce_frequent_words)
    # Mojo emits FMA, so the scaled values agree to tolerance, never bit-exactly.
    assert_same_matrix(got, expected)


def test_ctfidf_zero_row_is_left_alone():
    """scikit-learn's l1 normaliser divides a zero row by one, not by zero."""
    X = sp.csr_matrix(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 3.0]]))
    model = ClassTfidfTransformer()
    model.fit(X.copy())
    expected = model.transform(X.copy()).tocsr()
    got = mbt.ctfidf(X, idf=model._idf_diag.diagonal())
    assert got[0].nnz == expected[0].nnz == 0
    assert_same_matrix(got, expected)


def test_ctfidf_end_to_end_without_passing_idf():
    X = make_counts(seed=6)
    expected = ClassTfidfTransformer().fit_transform(X.copy()).tocsr()
    got = mbt.ctfidf(X)
    assert_same_matrix(got, expected)


def test_ctfidf_does_not_mutate_its_input():
    """Upstream normalises the caller's matrix in place; this port does not."""
    before = make_counts(seed=17).data
    upstream_input = make_counts(seed=17)
    ClassTfidfTransformer().fit(upstream_input).transform(upstream_input)
    assert not np.array_equal(before, upstream_input.data)

    mine = make_counts(seed=17)
    untouched = mine.data.copy()
    mbt.ctfidf(mine)
    np.testing.assert_array_equal(mine.data, untouched)


def test_ctfidf_rejects_a_dense_array():
    with pytest.raises(TypeError):
        mbt.ctfidf(np.zeros((3, 4)))
