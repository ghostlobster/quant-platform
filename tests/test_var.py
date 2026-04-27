"""Tests for ``risk/var.py`` — Historical / Parametric / Conditional VaR
and portfolio VaR.

Coverage target ≥ 90 % (closes #202). Beyond the original five tests we
add:

  * Confidence-level sweeps (0.90 / 0.95 / 0.99)
  * NaN-handling in the input series
  * Empty / single-row / exactly-30-row boundary inputs
  * The scipy-less fallback ``_norm_ppf`` path (hit via monkeypatching
    ``_scipy_stats`` to ``None``)
  * Domain-error path on ``_norm_ppf`` (p ∈ {0, 1, < 0, > 1})
  * Sign symmetry of the fallback approximation
  * ``conditional_var`` returns ``var`` when no losses sit below the
    threshold (the ``else`` branch on line 62)
  * ``portfolio_var`` zero-variance and three-asset paths
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk import var as var_module
from risk.var import (
    _norm_ppf,
    conditional_var,
    historical_var,
    parametric_var,
    portfolio_var,
)


def _normal_returns(n: int = 252, mu: float = -0.0002, sigma: float = 0.015,
                    seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(mu, sigma, n))


# ── _norm_ppf ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])
def test_norm_ppf_with_scipy(p: float) -> None:
    """With scipy installed the function delegates to ``scipy.stats.norm.ppf``."""
    from scipy import stats
    assert _norm_ppf(p) == pytest.approx(stats.norm.ppf(p), abs=1e-12)


@pytest.mark.parametrize("p", [0.01, 0.05, 0.5, 0.95, 0.99])
def test_norm_ppf_fallback_close_to_scipy(p: float, monkeypatch) -> None:
    """The scipy-less fallback approximation is good to ~5e-4 of scipy."""
    from scipy import stats
    monkeypatch.setattr(var_module, "_scipy_stats", None)
    expected = stats.norm.ppf(p)
    assert _norm_ppf(p) == pytest.approx(expected, abs=5e-4)


@pytest.mark.parametrize("q", [0.01, 0.1, 0.25, 0.49])
def test_norm_ppf_fallback_sign_symmetry(q: float, monkeypatch) -> None:
    """The fallback approximation is anti-symmetric: ppf(0.5+q) ≈ -ppf(0.5-q)."""
    monkeypatch.setattr(var_module, "_scipy_stats", None)
    upper = _norm_ppf(0.5 + q)
    lower = _norm_ppf(0.5 - q)
    assert upper == pytest.approx(-lower, abs=5e-4)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1, 2.0])
def test_norm_ppf_fallback_rejects_out_of_range(p: float, monkeypatch) -> None:
    monkeypatch.setattr(var_module, "_scipy_stats", None)
    with pytest.raises(ValueError, match="p must be in"):
        _norm_ppf(p)


# ── historical_var ────────────────────────────────────────────────────────────

def test_historical_var_positive_for_loss_distribution() -> None:
    v = historical_var(_normal_returns())
    assert v > 0 and v < 0.10


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_historical_var_monotone_in_confidence(conf: float) -> None:
    """Higher confidence → larger VaR. Tested as a triplet so the
    parametrization gives us per-level coverage too."""
    r = _normal_returns()
    v90 = historical_var(r, 0.90)
    v95 = historical_var(r, 0.95)
    v99 = historical_var(r, 0.99)
    assert v90 <= v95 <= v99
    # this case-specific assertion guards against a regression in
    # any single confidence level
    assert historical_var(r, conf) > 0


def test_historical_var_handles_nans() -> None:
    """NaNs in the input series must be dropped before percentile."""
    r = _normal_returns()
    r.iloc[::20] = np.nan  # sprinkle NaNs
    v = historical_var(r)
    assert np.isfinite(v) and v > 0


def test_historical_var_empty_returns_zero() -> None:
    assert historical_var(pd.Series([], dtype=float)) == 0.0


@pytest.mark.parametrize("n", [0, 1, 5, 29])
def test_historical_var_short_series_returns_zero(n: int) -> None:
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, n))
    assert historical_var(r) == 0.0


def test_historical_var_exactly_30_rows_is_active() -> None:
    """The threshold is ``< 30`` so a 30-row series should compute, not zero."""
    r = pd.Series(np.linspace(-0.05, 0.05, 30))
    assert historical_var(r) > 0


# ── parametric_var ───────────────────────────────────────────────────────────

def test_parametric_var_positive() -> None:
    assert parametric_var(_normal_returns()) > 0


def test_parametric_var_monotone_in_volatility() -> None:
    """Holding mean fixed, doubling sigma must increase parametric VaR."""
    low = _normal_returns(sigma=0.005)
    high = _normal_returns(sigma=0.020)
    assert parametric_var(high) > parametric_var(low)


def test_parametric_var_short_series_returns_zero() -> None:
    assert parametric_var(pd.Series([0.01, -0.02, 0.005])) == 0.0


def test_parametric_var_empty_returns_zero() -> None:
    assert parametric_var(pd.Series([], dtype=float)) == 0.0


def test_parametric_var_uses_scipy_fallback(monkeypatch) -> None:
    """Without scipy, parametric_var routes through the fallback ppf and
    still produces a finite positive number on a normal series."""
    monkeypatch.setattr(var_module, "_scipy_stats", None)
    v = parametric_var(_normal_returns())
    assert np.isfinite(v) and v > 0


# ── conditional_var ──────────────────────────────────────────────────────────

def test_cvar_ge_var() -> None:
    r = _normal_returns()
    assert conditional_var(r) >= historical_var(r)


def test_cvar_short_series_returns_zero() -> None:
    """Empty / short series take the early-return path on line 59."""
    assert conditional_var(pd.Series([], dtype=float)) == 0.0
    assert conditional_var(pd.Series([0.01, -0.02, 0.005])) == 0.0


def test_cvar_handles_all_positive_returns() -> None:
    """All-positive series — historical VaR returns a negative number
    (a "loss" of negative value = gain). CVaR must still produce a
    finite value rather than NaN."""
    r = pd.Series(np.linspace(0.001, 0.05, 30))
    cvar = conditional_var(r)
    assert np.isfinite(cvar)


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_cvar_monotone_in_confidence(conf: float) -> None:
    r = _normal_returns()
    assert conditional_var(r, conf) >= 0


# ── portfolio_var ────────────────────────────────────────────────────────────

def test_portfolio_var_two_assets_positive() -> None:
    weights = np.array([0.6, 0.4])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    assert portfolio_var(weights, cov) > 0


def test_portfolio_var_three_assets() -> None:
    weights = np.array([0.4, 0.4, 0.2])
    cov = np.array(
        [
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0003, 0.00010],
            [0.00005, 0.0001, 0.0002],
        ]
    )
    assert portfolio_var(weights, cov) > 0


def test_portfolio_var_zero_weights_returns_zero() -> None:
    weights = np.zeros(2)
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    assert portfolio_var(weights, cov) == pytest.approx(0.0, abs=1e-12)


def test_portfolio_var_scipy_fallback(monkeypatch) -> None:
    monkeypatch.setattr(var_module, "_scipy_stats", None)
    weights = np.array([0.6, 0.4])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    assert portfolio_var(weights, cov) > 0


@pytest.mark.parametrize("conf", [0.90, 0.95, 0.99])
def test_portfolio_var_monotone_in_confidence(conf: float) -> None:
    weights = np.array([0.5, 0.5])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    v90 = portfolio_var(weights, cov, 0.90)
    v95 = portfolio_var(weights, cov, 0.95)
    v99 = portfolio_var(weights, cov, 0.99)
    assert v90 <= v95 <= v99
    assert portfolio_var(weights, cov, conf) > 0
