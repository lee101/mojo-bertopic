"""Parity tests for top-n term selection, the topic similarity matrix and MMR."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

import mojo_bertopic as mbt
from bertopic import BERTopic
from bertopic.representation._mmr import mmr as upstream_mmr


def counts(seed=0, n_topics=9, vocab=47, levels=4):
    """Integer counts, so ties in the top-n boundary are common and realistic."""
    rng = np.random.default_rng(seed)
    dense = (rng.random((n_topics, vocab)) < 0.3) * rng.integers(1, levels, size=(n_topics, vocab))
    return sp.csr_matrix(dense.astype(np.float64))


def test_gram_matches_sklearn_cosine_similarity():
    X = counts(seed=11)
    expected = cosine_similarity(X)
    got = mbt.gram(X)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    # A symmetric matrix with a unit diagonal: scattering into the wrong row, or
    # pairing a row with the wrong column of the CSC side, breaks both.
    np.testing.assert_allclose(got, got.T, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.diag(got), 1.0, rtol=0.0, atol=1e-14)


def test_gram_handles_an_empty_row_like_sklearn():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]]))
    expected = cosine_similarity(X)
    got = mbt.gram(X)
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=1e-15)
    assert got[1].tolist() == [0.0, 0.0, 0.0]


def test_gram_on_a_single_row_and_on_nothing():
    X = sp.csr_matrix(np.array([[1.0, 0.0, 2.0]]))
    np.testing.assert_allclose(mbt.gram(X), np.ones((1, 1)))
    empty = sp.csr_matrix((0, 5), dtype=np.float64)
    assert mbt.gram(empty).shape == (0, 0)


def test_transpose_is_the_csc_companion_of_the_csr():
    """The gram kernel reads a CSC side; a wrong transpose breaks the product."""
    X = counts(seed=12, n_topics=4, vocab=9)
    t_indptr, t_rows, t_data = mbt.transpose(X)
    dense = np.asarray(X.todense())
    assert t_indptr.size == X.shape[1] + 1
    assert int(t_indptr[-1]) == X.nnz
    for c in range(X.shape[1]):
        lo, hi = int(t_indptr[c]), int(t_indptr[c + 1])
        cols = t_rows[lo:hi]
        assert list(cols) == sorted(cols), "CSC row indices must ascend"
        for k, r in enumerate(cols):
            assert t_data[lo + k] == dense[r, c]
    np.testing.assert_allclose(mbt.gram(X), cosine_similarity(X), rtol=1e-12, atol=1e-14)


def test_transpose_of_an_empty_matrix():
    empty = sp.csr_matrix((3, 4), dtype=np.float64)
    t_indptr, t_rows, t_data = mbt.transpose(empty)
    assert t_indptr.tolist() == [0, 0, 0, 0, 0]
    assert t_rows.size == 0 and t_data.size == 0


def test_top_n_picks_the_n_largest_values_of_every_row():
    X = counts(seed=13, n_topics=11, vocab=53)
    n = 5
    columns, counts_out = mbt.top_n(X, n)
    dense = np.asarray(X.todense())
    for row in range(X.shape[0]):
        keep = [int(c) for c in columns[row] if c >= 0]
        assert len(keep) == int(counts_out[row]) == n
        values = [dense[row, c] for c in keep]
        assert values == sorted(values, reverse=True)
        # A valid top-n is exactly the n largest: the smallest kept value is at
        # least as large as the largest unkept one.
        assert min(values) >= max(v for c, v in enumerate(dense[row]) if c not in keep)


def test_top_n_agrees_with_upstream_argpartition_when_the_boundary_is_unambiguous():
    X = counts(seed=15, n_topics=11, vocab=53, levels=997)
    n = 5
    upstream_idx = BERTopic._top_n_idx_sparse(X, n)
    columns, _ = mbt.top_n(X, n)
    dense = np.asarray(X.todense())
    for row in range(X.shape[0]):
        ordered = sorted(range(X.shape[1]), key=lambda c: (-dense[row, c], c))
        # Only compare when no tie straddles the n-th position.
        if dense[row, ordered[n - 1]] == dense[row, ordered[n]]:
            continue
        theirs = {int(i) for i in upstream_idx[row] if i is not None}
        assert {int(c) for c in columns[row]} == theirs


def test_top_n_pads_short_rows_and_reports_counts():
    X = sp.csr_matrix(np.array([[0.0, 0.0, 5.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                               [1.0, 2.0, 0.0, 0.0]]))
    columns, counts_out = mbt.top_n(X, 3)
    assert columns.tolist() == [[2, -1, -1], [-1, -1, -1], [1, 0, -1]]
    assert counts_out.tolist() == [1, 0, 2]


def test_top_n_ties_break_on_the_smaller_column():
    X = sp.csr_matrix(np.array([[0.0, 4.0, 4.0, 4.0, 0.0]]))
    columns, counts_out = mbt.top_n(X, 3)
    assert columns[0].tolist() == [1, 2, 3]
    assert counts_out.tolist() == [3]


def test_top_n_of_zero_is_empty():
    X = counts(seed=16, n_topics=3, vocab=5)
    columns, counts_out = mbt.top_n(X, 0)
    assert columns.shape == (3, 0)
    assert counts_out.tolist() == [0, 0, 0]


def test_top_n_terms_returns_scores_in_descending_order():
    X = counts(seed=14, n_topics=5, vocab=31)
    terms = mbt.top_n_terms(X, 4)
    for row, picked in enumerate(terms):
        assert len(picked) == 4
        scores = [v for _, v in picked]
        assert scores == sorted(scores, reverse=True)
        for col, value in picked:
            assert X[row, col] == value


def test_mmr_matches_upstream_selection():
    rng = np.random.default_rng(21)
    words, dim = 24, 9
    word_embeddings = rng.standard_normal((words, dim))
    doc_embedding = rng.standard_normal((1, dim))
    doc_sim = cosine_similarity(word_embeddings, doc_embedding)
    word_sim = cosine_similarity(word_embeddings)
    for diversity in (0.0, 0.1, 0.5, 0.9, 1.0):
        theirs = upstream_mmr(doc_embedding, word_embeddings,
                              [str(i) for i in range(words)], diversity, 7)
        ours = mbt.mmr(doc_sim, word_sim, diversity=diversity, top_n=7)
        assert [str(int(i)) for i in ours] == theirs


def test_mmr_ties_pick_the_lowest_index_like_argmax():
    doc_sim = np.array([[0.5], [0.5], [0.1]])
    word_sim = np.zeros((3, 3))
    assert mbt.mmr(doc_sim, word_sim, diversity=0.5, top_n=2).tolist() == [0, 1]


def test_mmr_cannot_pick_more_words_than_it_has():
    doc_sim = np.array([[0.9], [0.1]])
    word_sim = np.zeros((2, 2))
    assert mbt.mmr(doc_sim, word_sim, diversity=0.2, top_n=10).tolist() == [0, 1]
    assert mbt.mmr(doc_sim, word_sim, diversity=0.2, top_n=0).size == 0


def test_mmr_rejects_shapes_it_cannot_mean():
    with pytest.raises(ValueError):
        mbt.mmr(np.zeros((4, 2)), np.zeros((4, 4)))
    with pytest.raises(ValueError):
        mbt.mmr(np.zeros((4, 1)), np.zeros((4, 5)))
    with pytest.raises(ValueError):
        mbt.mmr(np.zeros((4, 1)), np.zeros((3, 3)))
