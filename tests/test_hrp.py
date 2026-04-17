"""Unit tests for risk/hrp.py — Hierarchical Risk Parity allocator.

Fixtures mirror tests/test_markowitz.py so that HRP and Markowitz are
covered by the same style of deterministic synthetic data.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.hrp import (
    _correlation_distance,
    _inverse_variance_weights,
    _quasi_diag,
    _recursive_bisection,
    get_hrp_portfolio,
    hrp_weights,
)
from risk.markowitz import OptimalPortfolio


def _make_price_data(n: int = 250, seed: int = 5) -> dict:
    """Three synthetic assets — mirrors tests/test_markowitz.py."""
    np.random.seed(seed)
    tickers = ["AAPL", "MSFT", "GOOG"]
    return {
        t: pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        for t in tickers
    }


def _make_correlated_returns(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Five-asset returns where pairs (0,1) and (2,3) are tightly coupled.

    HRP should group the coupled pairs and push weight onto asset 4 which
    has uncorrelated noise.
    """
    rng = np.random.default_rng(seed)
    base_ab = rng.normal(0, 0.01, n)
    base_cd = rng.normal(0, 0.01, n)
    cols = {
        "A": base_ab + rng.normal(0, 0.001, n),
        "B": base_ab + rng.normal(0, 0.001, n),
        "C": base_cd + rng.normal(0, 0.001, n),
        "D": base_cd + rng.normal(0, 0.001, n),
        "E": rng.normal(0, 0.01, n),
    }
    return pd.DataFrame(cols)


# ── Internal helpers ───────────────────────────────────────────────────────────

def test_correlation_distance_diagonal_is_zero():
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]], index=["A", "B"], columns=["A", "B"],
    )
    dist = _correlation_distance(corr)
    assert dist.loc["A", "A"] == pytest.approx(0.0)
    assert dist.loc["B", "B"] == pytest.approx(0.0)
    assert dist.loc["A", "B"] == pytest.approx(np.sqrt(0.25))


def test_correlation_distance_clips_tiny_negative_noise():
    """Floating-point noise can push the argument slightly below 0 — clip."""
    corr = pd.DataFrame(
        [[1.0 + 1e-12, 1.0], [1.0, 1.0]], index=["A", "B"], columns=["A", "B"],
    )
    dist = _correlation_distance(corr)
    # Must be real (no NaN from sqrt of negative).
    assert not dist.isna().any().any()


def test_inverse_variance_weights_sum_to_one():
    cov = pd.DataFrame(
        np.diag([1.0, 2.0, 4.0]), index=["A", "B", "C"], columns=["A", "B", "C"],
    )
    w = _inverse_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # Lowest variance should get the largest weight.
    assert w["A"] > w["B"] > w["C"]


def test_recursive_bisection_single_cluster_is_one():
    cov = pd.DataFrame([[1.0]], index=["A"], columns=["A"])
    w = _recursive_bisection(cov, ["A"])
    assert w["A"] == pytest.approx(1.0)


# ── hrp_weights ───────────────────────────────────────────────────────────────

def test_hrp_weights_sum_to_one():
    returns = _make_correlated_returns()
    w = hrp_weights(returns)
    assert w.sum() == pytest.approx(1.0)


def test_hrp_weights_all_positive():
    returns = _make_correlated_returns()
    w = hrp_weights(returns)
    assert (w >= 0).all()


def test_hrp_weights_empty_frame_returns_empty():
    w = hrp_weights(pd.DataFrame())
    assert w.empty


def test_hrp_weights_single_asset_returns_one():
    returns = pd.DataFrame({"AAPL": np.random.randn(50) * 0.01})
    w = hrp_weights(returns)
    assert len(w) == 1
    assert w["AAPL"] == pytest.approx(1.0)


def test_hrp_weights_retain_input_order():
    returns = _make_correlated_returns()
    w = hrp_weights(returns)
    # Columns come back in the original order, not the quasi-diag order.
    assert list(w.index) == list(returns.columns)


def test_hrp_weights_penalise_redundant_assets():
    """Two near-identical pairs (A,B) + (C,D) vs an independent E.

    Each correlated pair should share roughly one slice of the bisection,
    so E (alone) should receive more weight than any individual pair
    member."""
    returns = _make_correlated_returns(n=500, seed=11)
    w = hrp_weights(returns)
    assert w["E"] > w["A"]
    assert w["E"] > w["C"]


def test_hrp_weights_reproducible():
    returns = _make_correlated_returns()
    w1 = hrp_weights(returns)
    w2 = hrp_weights(returns)
    pd.testing.assert_series_equal(w1, w2)


# ── get_hrp_portfolio ─────────────────────────────────────────────────────────

def test_get_hrp_portfolio_returns_optimal_portfolio():
    data = _make_price_data()
    p = get_hrp_portfolio(data)
    assert isinstance(p, OptimalPortfolio)
    assert set(p.weights.keys()) == set(data.keys())
    assert sum(p.weights.values()) == pytest.approx(1.0, abs=1e-6)
    assert p.expected_volatility > 0


def test_get_hrp_portfolio_empty_data_returns_none():
    assert get_hrp_portfolio({}) is None


def test_get_hrp_portfolio_none_returns_none():
    assert get_hrp_portfolio(None) is None


def test_get_hrp_portfolio_single_ticker_trivial_weight():
    series = pd.Series(100 + np.cumsum(np.random.randn(60) * 0.5))
    p = get_hrp_portfolio({"AAPL": series})
    assert p is not None
    assert p.weights == {"AAPL": 1.0}


def test_get_hrp_portfolio_single_ticker_too_short_returns_none():
    # Need at least 2 points to have a non-trivial return series.
    p = get_hrp_portfolio({"AAPL": pd.Series([100.0])})
    assert p is None


def test_get_hrp_portfolio_sharpe_matches_formula():
    data = _make_price_data()
    rf = 0.02
    p = get_hrp_portfolio(data, risk_free_rate=rf)
    assert p is not None
    expected_sharpe = (p.expected_return - rf) / p.expected_volatility
    assert p.sharpe_ratio == pytest.approx(expected_sharpe, rel=1e-6)


# ── quasi-diagonalisation (small hand-crafted linkage) ────────────────────────

def test_quasi_diag_simple_linkage():
    """A 3-item single-link tree: items 0+1 merge first, then cluster 3 ∪ 2."""
    # Columns: [left, right, distance, size]
    link = np.array([
        [0, 1, 0.5, 2],   # cluster 3 = {0, 1}
        [3, 2, 0.9, 3],   # cluster 4 = {3, 2}
    ])
    order = _quasi_diag(link)
    assert sorted(order) == [0, 1, 2]
    # The last merge pulled cluster 3 to the left of leaf 2 → 2 should be
    # the last element after quasi-diagonalisation.
    assert order[-1] == 2
