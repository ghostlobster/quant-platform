"""VaR, CVaR (Expected Shortfall) and portfolio risk metrics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class RiskMetrics:
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    worst_day_pct: float
    best_day_pct: float
    volatility_annual: float
    n_observations: int


def _pct_returns(portfolio_values: Sequence[float]) -> list[float]:
    """Convert a series of portfolio values to daily % returns."""
    vals = list(portfolio_values)
    if len(vals) < 2:
        return []
    return [(vals[i] - vals[i - 1]) / vals[i - 1] for i in range(1, len(vals))]


def historical_var(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Historical simulation VaR at given confidence level.
    Returns a positive number representing the loss at the given confidence level.
    e.g. 0.02 means 2% loss at the VaR threshold.
    """
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    idx = int(math.floor((1 - confidence) * len(sorted_r)))
    idx = max(0, min(idx, len(sorted_r) - 1))
    return -sorted_r[idx]


def historical_cvar(returns: Sequence[float], confidence: float = 0.95) -> float:
    """Historical CVaR (Expected Shortfall) — mean of losses beyond VaR threshold."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = int(math.floor((1 - confidence) * len(sorted_r)))
    cutoff = max(1, cutoff)
    tail = sorted_r[:cutoff]
    return -sum(tail) / len(tail)


def monte_carlo_var(
    returns: Sequence[float],
    n_sims: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR via bootstrap resampling of historical returns."""
    if not returns:
        return 0.0
    rng = random.Random(seed)
    returns_list = list(returns)
    simulated = [rng.choice(returns_list) for _ in range(n_sims)]
    simulated.sort()
    idx = int(math.floor((1 - confidence) * n_sims))
    idx = max(0, min(idx, n_sims - 1))
    return -simulated[idx]


def compute_risk_metrics(
    portfolio_values: Sequence[float],
    confidence_levels: tuple[float, float] = (0.95, 0.99),
) -> Optional[RiskMetrics]:
    """Compute a full RiskMetrics snapshot from a portfolio value series."""
    returns = _pct_returns(portfolio_values)
    if len(returns) < 5:
        return None

    c_lo, c_hi = confidence_levels
    ann_vol = float(np.std(returns, ddof=1)) * math.sqrt(252) if len(returns) > 1 else 0.0

    return RiskMetrics(
        var_95=historical_var(returns, c_lo),
        var_99=historical_var(returns, c_hi),
        cvar_95=historical_cvar(returns, c_lo),
        cvar_99=historical_cvar(returns, c_hi),
        worst_day_pct=min(returns) * 100,
        best_day_pct=max(returns) * 100,
        volatility_annual=ann_vol * 100,
        n_observations=len(returns),
    )
