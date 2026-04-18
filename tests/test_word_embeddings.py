"""Tests for analysis/word_embeddings.py — word2vec via gensim."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import gensim  # noqa: F401
    _GENSIM = True
except ImportError:
    _GENSIM = False

skip_no_gensim = pytest.mark.skipif(not _GENSIM, reason="gensim not installed")


# A toy corpus with two clear "topics" so word2vec can learn separable
# vectors even at min_count=1 / vector_size=16.
_CORPUS = [
    "earnings revenue profit margin shareholders dividend payout cash flow",
    "earnings beat revenue forecast guidance shareholders dividend",
    "balance sheet revenue earnings shareholders profit",
    "guidance dividend profit revenue cash flow shareholders",
    "machine learning neural network gpu training inference deployment",
    "deep learning gpu training inference model architecture deployment",
    "transformer attention mechanism training inference deployment model",
    "neural network gpu inference model deployment training pipeline",
] * 4  # repeat to give skip-gram enough context


# ── Gating ────────────────────────────────────────────────────────────────────

def test_train_embeddings_raises_without_gensim():
    from analysis import word_embeddings
    with patch("analysis.word_embeddings._GENSIM_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="gensim"):
            word_embeddings.train_embeddings(["hello world"])


def test_document_embedding_raises_without_gensim():
    from analysis import word_embeddings
    with patch("analysis.word_embeddings._GENSIM_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="gensim"):
            word_embeddings.document_embedding(None, "doc")  # type: ignore[arg-type]


def test_nearest_terms_raises_without_gensim():
    from analysis import word_embeddings
    with patch("analysis.word_embeddings._GENSIM_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="gensim"):
            word_embeddings.nearest_terms(None, "x", k=3)  # type: ignore[arg-type]


def test_train_embeddings_rejects_empty_corpus():
    from analysis.word_embeddings import train_embeddings
    if not _GENSIM:
        pytest.skip("gensim not installed")
    with pytest.raises(ValueError, match="usable"):
        train_embeddings([])
    with pytest.raises(ValueError, match="usable"):
        train_embeddings(["", "  "])


# ── gensim happy paths ────────────────────────────────────────────────────────

@skip_no_gensim
def test_train_embeddings_returns_correct_shape_and_vocab():
    from analysis.word_embeddings import train_embeddings
    result = train_embeddings(_CORPUS, vector_size=16, min_count=1, epochs=5)
    assert result.vector_size == 16
    assert result.vocab_size > 5
    # Model exposes the gensim wv API
    wv = result.model.wv  # type: ignore[attr-defined]
    assert "revenue" in wv
    assert wv["revenue"].shape == (16,)


@skip_no_gensim
def test_document_embedding_shape_and_zero_vector_for_oov():
    from analysis.word_embeddings import document_embedding, train_embeddings
    result = train_embeddings(_CORPUS, vector_size=16, min_count=1, epochs=3)
    vec = document_embedding(result, "earnings revenue dividend")
    assert vec.shape == (16,)
    assert np.linalg.norm(vec) > 0
    zero = document_embedding(result, "qqqqqq zyzzyx")
    assert np.allclose(zero, 0.0)
    empty = document_embedding(result, "")
    assert np.allclose(empty, 0.0)


@skip_no_gensim
def test_nearest_terms_returns_in_vocab_results():
    from analysis.word_embeddings import nearest_terms, train_embeddings
    result = train_embeddings(_CORPUS, vector_size=16, min_count=1, epochs=5)
    out = nearest_terms(result, "revenue", k=3)
    assert len(out) == 3
    for term, score in out:
        assert isinstance(term, str)
        assert -1.0 <= score <= 1.0
        # Returned terms must come from the trained vocabulary
        assert term in result.model.wv  # type: ignore[attr-defined]


@skip_no_gensim
def test_nearest_terms_oov_returns_empty_list():
    from analysis.word_embeddings import nearest_terms, train_embeddings
    result = train_embeddings(_CORPUS, vector_size=16, min_count=1, epochs=3)
    assert nearest_terms(result, "zzzzzzzzzzzz", k=5) == []


@skip_no_gensim
def test_document_embedding_average_invariant_to_token_order():
    """Average-pooling must be order-invariant."""
    from analysis.word_embeddings import document_embedding, train_embeddings
    result = train_embeddings(_CORPUS, vector_size=16, min_count=1, epochs=3)
    a = document_embedding(result, "earnings revenue dividend")
    b = document_embedding(result, "dividend revenue earnings")
    np.testing.assert_allclose(a, b, atol=1e-6)
