"""Tests for analysis/risk_metrics.py — VaR, CVaR, and compute_risk_metrics."""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.risk_metrics import (
    historical_var,
    historical_cvar,
    monte_carlo_var,
    compute_risk_metrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _normal_returns(n: int = 100, mu: float = 0.0, sigma: float = 0.01, seed: int = 42) -> list[float]:
    rng = random.Random(seed)
    return [rng.gauss(mu, sigma) for _ in range(n)]


def _nav_from_returns(returns: list[float], start: float = 100_000.0) -> list[float]:
    nav = [start]
    for r in returns:
        nav.append(nav[-1] * (1 + r))
    return nav


# ── historical_var ────────────────────────────────────────────────────────────

def test_historical_var_known_result():
    # Returns: [-0.05, -0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
    # sorted: same. At 95% conf, index = int(0.05 * 10) = 0 → -sorted[0] = 0.05
    returns = [-0.05, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04]
    var = historical_var(returns, confidence=0.95)
    assert var == pytest.approx(0.05)


def test_historical_var_returns_positive():
    returns = _normal_returns(200)
    var = historical_var(returns, 0.95)
    assert var > 0


def test_historical_var_empty_returns_zero():
    assert historical_var([], 0.95) == 0.0


# ── historical_cvar ───────────────────────────────────────────────────────────

def test_historical_cvar_ge_var():
    returns = _normal_returns(200)
    var  = historical_var(returns, 0.95)
    cvar = historical_cvar(returns, 0.95)
    assert cvar >= var


def test_historical_cvar_empty_returns_zero():
    assert historical_cvar([], 0.95) == 0.0


def test_historical_cvar_is_mean_of_tail():
    # With 10 returns sorted ascending, at 95% conf the tail is the worst 1 value
    returns = list(range(-5, 5))  # [-5,-4,-3,-2,-1,0,1,2,3,4]
    cvar = historical_cvar(returns, 0.95)
    # cutoff = int(0.05 * 10) = 0 → max(0,1) = 1 → tail = [-5], cvar = 5.0
    assert cvar == pytest.approx(5.0)


# ── monte_carlo_var ───────────────────────────────────────────────────────────

def test_monte_carlo_var_close_to_historical_for_large_n():
    returns = _normal_returns(500, seed=7)
    hist = historical_var(returns, 0.95)
    mc   = monte_carlo_var(returns, n_sims=50_000, confidence=0.95, seed=7)
    # Should be within 0.5 percentage points for a large sample
    assert abs(mc - hist) < 0.005


def test_monte_carlo_var_empty_returns_zero():
    assert monte_carlo_var([], n_sims=1000) == 0.0


def test_monte_carlo_var_deterministic_with_seed():
    returns = _normal_returns(100)
    v1 = monte_carlo_var(returns, seed=123)
    v2 = monte_carlo_var(returns, seed=123)
    assert v1 == v2


# ── compute_risk_metrics ──────────────────────────────────────────────────────

def test_compute_risk_metrics_insufficient_data_returns_none():
    nav = _nav_from_returns(_normal_returns(8))  # 8 returns → 9 nav values → < 10
    result = compute_risk_metrics(nav)
    assert result is None


def test_compute_risk_metrics_exactly_at_boundary_returns_none():
    # 9 nav values → 8 returns → should still be None
    result = compute_risk_metrics([100_000.0] * 9)
    assert result is None


def test_compute_risk_metrics_all_fields_populated():
    returns = _normal_returns(50, mu=0.0005, sigma=0.012)
    nav = _nav_from_returns(returns)
    m = compute_risk_metrics(nav)
    assert m is not None
    assert m.var_95 > 0
    assert m.var_99 > 0
    assert m.cvar_95 > 0
    assert m.cvar_99 > 0
    assert m.volatility_annual > 0
    assert m.n_observations == 50
    assert m.worst_day_pct < 0          # worst day is a loss (negative %)
    assert m.best_day_pct > 0           # best day is a gain


def test_compute_risk_metrics_var99_ge_var95():
    returns = _normal_returns(100)
    nav = _nav_from_returns(returns)
    m = compute_risk_metrics(nav)
    assert m is not None
    assert m.var_99 >= m.var_95


def test_compute_risk_metrics_cvar_ge_var():
    returns = _normal_returns(100)
    nav = _nav_from_returns(returns)
    m = compute_risk_metrics(nav)
    assert m is not None
    assert m.cvar_95 >= m.var_95
    assert m.cvar_99 >= m.var_99


def test_compute_risk_metrics_volatility_annualised():
    # For a series with known daily vol, annualised vol ≈ daily_vol * sqrt(252)
    returns = _normal_returns(252, mu=0.0, sigma=0.01)
    nav = _nav_from_returns(returns)
    m = compute_risk_metrics(nav)
    assert m is not None
    # annualised vol should be near 0.01 * sqrt(252) * 100 ≈ 15.87%
    expected = 0.01 * math.sqrt(252) * 100
    assert abs(m.volatility_annual - expected) < 3.0  # within 3pp for random sample
