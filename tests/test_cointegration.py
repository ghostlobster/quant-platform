"""Tests for analysis/cointegration.py — Engle-Granger two-step test."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.cointegration import (
    _STATSMODELS_AVAILABLE,
    CointegrationResult,
    _half_life,
    _hedge_ratio_ols,
    engle_granger,
    screen_cointegrated_pairs,
)

# ── Synthetic data helpers ───────────────────────────────────────────────────

def _cointegrated_pair(n: int = 300, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """``y = 2x + stationary ε`` — definitionally cointegrated."""
    rng = np.random.default_rng(seed)
    x = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    eps = rng.normal(scale=0.5, size=n)      # stationary noise
    y = 2.0 * x + eps
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(y, index=idx), pd.Series(x, index=idx)


def _non_cointegrated_pair(n: int = 300, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.default_rng(seed)
    a = 100 + np.cumsum(rng.normal(scale=0.5, size=n))
    b = 100 + np.cumsum(rng.normal(scale=0.5, size=n + 17))[:n]
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(a, index=idx), pd.Series(b, index=idx)


# ── _hedge_ratio_ols ─────────────────────────────────────────────────────────

def test_hedge_ratio_recovers_slope():
    rng = np.random.default_rng(5)
    b = rng.normal(size=200)
    a = 3.0 * b + rng.normal(scale=0.01, size=200)
    assert _hedge_ratio_ols(a, b) == pytest.approx(3.0, rel=0.05)


def test_hedge_ratio_zero_when_denominator_zero():
    zero = np.zeros(10)
    some = np.array([1.0, 2.0, 3.0] * 4)[:10]
    assert _hedge_ratio_ols(some, zero) == 0.0


# ── _half_life ───────────────────────────────────────────────────────────────

def test_half_life_random_walk_longer_than_reverting():
    """A random walk has no mean reversion, so its AR(1) half-life
    should be significantly larger than that of a stationary AR(1)."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=500))
    hl_walk = _half_life(walk)

    reverting = np.zeros(500)
    for i in range(1, 500):
        reverting[i] = 0.9 * reverting[i - 1] + rng.normal(scale=0.1)
    hl_reverting = _half_life(reverting)

    assert hl_walk > hl_reverting * 5


def test_half_life_finite_for_mean_reverting_spread():
    rng = np.random.default_rng(1)
    spread = np.zeros(500)
    for i in range(1, 500):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(scale=0.1)
    hl = _half_life(spread)
    assert 1.0 < hl < 50.0


def test_half_life_short_input_returns_inf():
    assert _half_life(np.zeros(5)) == float("inf")


# ── engle_granger ────────────────────────────────────────────────────────────

def test_engle_granger_returns_dataclass():
    y, x = _cointegrated_pair()
    result = engle_granger(y, x)
    assert isinstance(result, CointegrationResult)


def test_engle_granger_none_inputs_return_empty():
    result = engle_granger(None, pd.Series([1.0, 2.0]))
    assert result.p_value is None
    assert result.converged is False


def test_engle_granger_short_series_returns_empty():
    y = pd.Series(range(5))
    x = pd.Series(range(5))
    result = engle_granger(y, x)
    assert result.converged is False


def test_engle_granger_inner_joins_on_index():
    """Different-length inputs should be inner-joined, not raise."""
    n = 100
    idx_a = pd.date_range("2024-01-01", periods=n, freq="D")
    idx_b = pd.date_range("2024-01-10", periods=n, freq="D")
    y = pd.Series(np.cumsum(np.random.randn(n)), index=idx_a)
    x = pd.Series(np.cumsum(np.random.randn(n)), index=idx_b)
    result = engle_granger(y, x)
    # 91 overlapping days ≥ 30 → test should run.
    assert result.hedge_ratio is not None


@pytest.mark.skipif(not _STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_engle_granger_detects_cointegrated_pair():
    y, x = _cointegrated_pair(n=400, seed=10)
    result = engle_granger(y, x)
    assert result.converged
    assert result.p_value < 0.05
    # Slope recovered with tolerance:
    assert 1.8 < result.hedge_ratio < 2.2


@pytest.mark.skipif(not _STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_engle_granger_rejects_independent_random_walks():
    """Independent RWs should usually fail the test.  Use a few seeds
    to avoid coincidental rejections."""
    p_values = []
    for seed in range(5):
        y, x = _non_cointegrated_pair(n=300, seed=seed)
        r = engle_granger(y, x)
        if r.converged and r.p_value is not None:
            p_values.append(r.p_value)
    # On average, random walks should look non-cointegrated (mean p > 0.1).
    assert np.mean(p_values) > 0.1


def test_engle_granger_degrades_without_statsmodels(monkeypatch):
    monkeypatch.setattr(
        "analysis.cointegration._STATSMODELS_AVAILABLE", False,
    )
    y, x = _cointegrated_pair()
    result = engle_granger(y, x)
    assert result.converged is False
    assert result.p_value is None
    # Hedge ratio + half-life still computed from closed-form OLS / AR(1).
    assert result.hedge_ratio is not None


# ── screen_cointegrated_pairs ───────────────────────────────────────────────

@pytest.mark.skipif(not _STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_screen_identifies_cointegrated_pair_in_noise():
    y, x = _cointegrated_pair(n=400, seed=11)
    # Two extra independent RWs as decoys.
    rng = np.random.default_rng(99)
    noise_a = pd.Series(
        100 + np.cumsum(rng.normal(scale=0.5, size=400)),
        index=y.index, name="NOISE_A",
    )
    noise_b = pd.Series(
        100 + np.cumsum(rng.normal(scale=0.5, size=400)),
        index=y.index, name="NOISE_B",
    )
    data = {"Y": y, "X": x, "NOISE_A": noise_a, "NOISE_B": noise_b}
    hits = screen_cointegrated_pairs(data, significance=0.05)
    # At least the (Y, X) pair should be found.
    assert any(
        {h["a"], h["b"]} == {"Y", "X"} for h in hits
    )


@pytest.mark.skipif(not _STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_screen_sorts_by_p_value_ascending():
    y, x = _cointegrated_pair(n=400, seed=12)
    data = {"Y": y, "X": x}
    hits = screen_cointegrated_pairs(data, significance=0.5)
    # Single pair → trivially sorted, but also verify the return shape.
    assert hits
    assert set(hits[0].keys()) == {
        "a", "b", "p_value", "hedge_ratio", "half_life",
    }


@pytest.mark.skipif(not _STATSMODELS_AVAILABLE, reason="statsmodels not installed")
def test_screen_respects_max_half_life_filter():
    y, x = _cointegrated_pair(n=400, seed=13)
    hits = screen_cointegrated_pairs(
        {"Y": y, "X": x}, significance=0.5, max_half_life=0.001,
    )
    # Impossibly short half-life filter → nothing passes.
    assert hits == []


def test_screen_empty_dict_returns_empty_list():
    assert screen_cointegrated_pairs({}) == []


def test_screen_single_series_returns_empty_list():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    assert screen_cointegrated_pairs({"A": pd.Series(range(50), index=idx)}) == []
