"""Tests for analysis/topic_modeling.py — LDA over text corpora."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import sklearn  # noqa: F401
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

skip_no_sklearn = pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")


# Two clearly-separable topic clusters so LDA can recover them.
_FINANCE_DOCS = [
    "earnings revenue profit margin shareholders dividend",
    "balance sheet revenue earnings shareholders",
    "guidance dividend profit revenue cash flow",
    "earnings beat revenue forecast guidance",
    "dividend payout shareholders profit margin",
]
_TECH_DOCS = [
    "machine learning neural network gpu training",
    "deep learning gpu training inference model",
    "transformer architecture attention training inference",
    "neural network gpu inference model deployment",
    "machine learning model training deployment pipeline",
]
_CORPUS = _FINANCE_DOCS + _TECH_DOCS


# ── Guard against missing sklearn ─────────────────────────────────────────────

def test_fit_lda_raises_without_sklearn():
    from analysis import topic_modeling
    with patch("analysis.topic_modeling._SKLEARN_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="scikit-learn"):
            topic_modeling.fit_lda(["hello world"], n_topics=2)


def test_fit_lda_rejects_empty_corpus():
    from analysis.topic_modeling import fit_lda
    if not _SKLEARN:
        pytest.skip("sklearn not installed")
    with pytest.raises(ValueError, match="non-empty"):
        fit_lda([], n_topics=2)
    with pytest.raises(ValueError, match="non-empty"):
        fit_lda(["", "  ", "\n"], n_topics=2)


# ── Happy-path LDA ────────────────────────────────────────────────────────────

@skip_no_sklearn
def test_fit_and_infer_topic_distributions_sum_to_one():
    from analysis.topic_modeling import fit_lda, infer_topics
    result = fit_lda(_CORPUS, n_topics=2, min_df=1)
    dists = infer_topics(result, _CORPUS)
    assert dists.shape == (len(_CORPUS), 2)
    np.testing.assert_allclose(dists.sum(axis=1), np.ones(len(_CORPUS)), atol=1e-6)


@skip_no_sklearn
def test_lda_separates_two_obvious_topic_clusters():
    """With finance vs. tech corpora and n_topics=2, each cluster should
    concentrate mass on a different topic."""
    from analysis.topic_modeling import fit_lda, infer_topics
    result = fit_lda(_CORPUS, n_topics=2, min_df=1)
    fin = infer_topics(result, _FINANCE_DOCS)
    tech = infer_topics(result, _TECH_DOCS)
    # Finance docs should agree on a dominant topic; tech docs should agree on the OTHER topic.
    fin_topic = int(np.argmax(fin.mean(axis=0)))
    tech_topic = int(np.argmax(tech.mean(axis=0)))
    assert fin_topic != tech_topic


@skip_no_sklearn
def test_top_terms_per_topic_returns_known_terms():
    from analysis.topic_modeling import fit_lda, top_terms_per_topic
    result = fit_lda(_CORPUS, n_topics=2, min_df=1)
    top = top_terms_per_topic(result, n=5)
    assert len(top) == 2
    flat = {t for topic in top for t in topic}
    # Some recognisable corpus terms should make it into the top lists.
    assert flat & {"revenue", "earnings", "dividend",
                   "training", "model", "neural", "inference"}


@skip_no_sklearn
def test_infer_topics_empty_input_returns_empty_matrix():
    from analysis.topic_modeling import fit_lda, infer_topics
    result = fit_lda(_CORPUS, n_topics=3, min_df=1)
    out = infer_topics(result, [])
    assert out.shape == (0, 3)


@skip_no_sklearn
def test_infer_topics_blank_documents_get_uniform_row():
    from analysis.topic_modeling import fit_lda, infer_topics
    result = fit_lda(_CORPUS, n_topics=4, min_df=1)
    out = infer_topics(result, ["", "   "])
    np.testing.assert_allclose(out, np.full((2, 4), 0.25))


@skip_no_sklearn
def test_infer_topics_all_oov_returns_uniform():
    """A document whose tokens are entirely out-of-vocabulary should not
    crash; it just yields the uniform fallback."""
    from analysis.topic_modeling import fit_lda, infer_topics
    result = fit_lda(_CORPUS, n_topics=3, min_df=1)
    out = infer_topics(result, ["zyzzyx qqqqqq"])
    # nnz == 0 path → entire row stays at uniform 1/3.
    np.testing.assert_allclose(out, np.full((1, 3), 1.0 / 3.0))


@skip_no_sklearn
def test_ticker_topic_distribution_averages_per_ticker():
    from analysis.topic_modeling import fit_lda, ticker_topic_distribution
    result = fit_lda(_CORPUS, n_topics=2, min_df=1)
    dist = ticker_topic_distribution(result, {
        "FIN": _FINANCE_DOCS,
        "TECH": _TECH_DOCS,
        "EMPTY": [],
    })
    assert set(dist) == {"FIN", "TECH", "EMPTY"}
    for vec in dist.values():
        assert vec.shape == (2,)
        np.testing.assert_allclose(vec.sum(), 1.0, atol=1e-6)
    # Empty bucket falls back to uniform.
    np.testing.assert_allclose(dist["EMPTY"], np.array([0.5, 0.5]))
    # FIN and TECH should differ in their dominant topic.
    assert int(np.argmax(dist["FIN"])) != int(np.argmax(dist["TECH"]))
