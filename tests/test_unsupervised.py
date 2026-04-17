"""Tests for analysis/unsupervised.py — PCA + k-means risk factors.

Uses deterministic synthetic data so the results are reproducible on
every run (matches the style of tests/test_markowitz.py and
tests/test_hrp.py).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.unsupervised import (
    PCAFactors,
    _correlation_distance,
    cluster_assets,
    cluster_members,
    pca_risk_factors,
)


def _two_factor_returns(n: int = 250, seed: int = 7) -> pd.DataFrame:
    """Two latent factors (A=B, C=D correlated; E pure noise)."""
    rng = np.random.default_rng(seed)
    f_ab = rng.normal(0, 0.01, n)
    f_cd = rng.normal(0, 0.01, n)
    cols = {
        "A": f_ab + rng.normal(0, 0.001, n),
        "B": f_ab + rng.normal(0, 0.001, n),
        "C": f_cd + rng.normal(0, 0.001, n),
        "D": f_cd + rng.normal(0, 0.001, n),
        "E": rng.normal(0, 0.01, n),
    }
    return pd.DataFrame(cols)


def _random_returns(n: int = 250, n_assets: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 0.01, (n, n_assets))
    return pd.DataFrame(data, columns=[f"T{i}" for i in range(n_assets)])


# ── pca_risk_factors ──────────────────────────────────────────────────────────

def test_pca_returns_pcafactors_instance():
    returns = _random_returns()
    factors = pca_risk_factors(returns, n_components=3)
    assert isinstance(factors, PCAFactors)


def test_pca_loadings_shape():
    returns = _random_returns(n_assets=6)
    factors = pca_risk_factors(returns, n_components=3)
    assert factors.loadings.shape == (6, 3)
    assert list(factors.loadings.columns) == ["PC1", "PC2", "PC3"]
    assert list(factors.loadings.index) == list(returns.columns)


def test_pca_explained_variance_monotone_decreasing():
    returns = _random_returns(n_assets=8, n=300)
    factors = pca_risk_factors(returns, n_components=5)
    vals = factors.explained_variance.values
    for i in range(1, len(vals)):
        assert vals[i - 1] >= vals[i] - 1e-9


def test_pca_explained_variance_sums_leq_one():
    returns = _random_returns(n_assets=6)
    factors = pca_risk_factors(returns, n_components=3)
    assert factors.explained_variance.sum() <= 1.0 + 1e-9


def test_pca_two_factor_structure_dominates_first_two_components():
    """A/B/C/D driven by two latent factors; PC1+PC2 should capture most var."""
    returns = _two_factor_returns(n=500, seed=11)
    factors = pca_risk_factors(returns, n_components=5)
    top_two = factors.explained_variance.iloc[:2].sum()
    assert top_two > 0.4


def test_pca_components_have_observation_index():
    returns = _random_returns(n_assets=4)
    factors = pca_risk_factors(returns, n_components=2)
    assert list(factors.components.index) == list(returns.index)
    assert factors.components.shape == (len(returns), 2)


def test_pca_n_components_clipped_to_rank():
    """Requesting more components than min(n_rows, n_assets) is capped."""
    returns = _random_returns(n_assets=3, n=100)
    factors = pca_risk_factors(returns, n_components=10)
    assert factors.loadings.shape[1] == 3


def test_pca_empty_returns_empty_factors():
    factors = pca_risk_factors(pd.DataFrame())
    assert factors.loadings.empty
    assert factors.explained_variance.empty
    assert factors.components.empty


def test_pca_none_returns_empty_factors():
    factors = pca_risk_factors(None)  # type: ignore[arg-type]
    assert factors.loadings.empty


def test_pca_single_asset_returns_empty():
    one_col = pd.DataFrame({"AAPL": np.random.randn(100)})
    factors = pca_risk_factors(one_col)
    assert factors.loadings.empty


def test_pca_too_few_observations_returns_empty():
    returns = _random_returns(n=10, n_assets=4)   # below _MIN_OBS=20
    factors = pca_risk_factors(returns)
    assert factors.loadings.empty


# ── cluster_assets ───────────────────────────────────────────────────────────

def test_cluster_assets_returns_series_indexed_by_ticker():
    returns = _random_returns(n_assets=6)
    labels = cluster_assets(returns, n_clusters=3, random_state=0)
    assert isinstance(labels, pd.Series)
    assert list(labels.index) == list(returns.columns)


def test_cluster_assets_label_count_matches_request():
    returns = _random_returns(n_assets=10, n=300)
    labels = cluster_assets(returns, n_clusters=4, random_state=0)
    assert labels.nunique() <= 4


def test_cluster_assets_reproducible_with_seed():
    returns = _random_returns(n_assets=8, n=250)
    a = cluster_assets(returns, n_clusters=3, random_state=7)
    b = cluster_assets(returns, n_clusters=3, random_state=7)
    pd.testing.assert_series_equal(a, b)


def test_cluster_assets_groups_correlated_pairs():
    """A,B correlated; C,D correlated; E independent — A should cluster with
    B and C with D."""
    returns = _two_factor_returns(n=500, seed=2)
    labels = cluster_assets(returns, n_clusters=3, random_state=0)
    assert labels["A"] == labels["B"]
    assert labels["C"] == labels["D"]


def test_cluster_assets_n_clusters_clipped_to_n_assets():
    returns = _random_returns(n_assets=3, n=250)
    labels = cluster_assets(returns, n_clusters=10, random_state=0)
    assert labels.nunique() <= 3


def test_cluster_assets_empty_returns_empty():
    assert cluster_assets(pd.DataFrame()).empty


def test_cluster_assets_single_asset_returns_empty():
    assert cluster_assets(pd.DataFrame({"A": [0.01] * 50})).empty


def test_cluster_assets_too_few_observations_returns_empty():
    returns = _random_returns(n=10, n_assets=4)  # below _MIN_OBS
    assert cluster_assets(returns).empty


# ── cluster_members helper ───────────────────────────────────────────────────

def test_cluster_members_inverts_labels():
    labels = pd.Series(
        {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2}, name="cluster",
    )
    members = cluster_members(labels)
    assert members == {0: ["A", "B"], 1: ["C", "D"], 2: ["E"]}


def test_cluster_members_empty_input():
    assert cluster_members(pd.Series(dtype=int)) == {}
    assert cluster_members(None) == {}  # type: ignore[arg-type]


# ── _correlation_distance ────────────────────────────────────────────────────

def test_correlation_distance_zero_on_diagonal():
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"],
    )
    dist = _correlation_distance(corr)
    assert dist[0, 0] == pytest.approx(0.0)
    assert dist[1, 1] == pytest.approx(0.0)


def test_correlation_distance_symmetric():
    corr = pd.DataFrame(
        [[1.0, 0.3, -0.2],
         [0.3, 1.0, 0.7],
         [-0.2, 0.7, 1.0]],
        index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    dist = _correlation_distance(corr)
    assert np.allclose(dist, dist.T)
