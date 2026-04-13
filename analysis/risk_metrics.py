"""Portfolio risk metrics: VaR, CVaR, and stress statistics."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskMetrics:
    var_95: float          # 1-day VaR at 95% confidence (positive number = loss)
    var_99: float          # 1-day VaR at 99% confidence
    cvar_95: float         # CVaR / Expected Shortfall at 95%
    cvar_99: float         # CVaR at 99%
    worst_day_pct: float   # worst single-day % loss in lookback window
    best_day_pct: float    # best single-day % gain
    volatility_annual: float  # annualised daily volatility
    n_observations: int    # number of return observations used


def historical_var(returns: list[float], confidence: float = 0.95) -> float:
    """Historical simulation VaR. Returns positive loss value."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    index = int((1 - confidence) * len(sorted_r))
    return -sorted_r[max(index, 0)]


def historical_cvar(returns: list[float], confidence: float = 0.95) -> float:
    """CVaR (Expected Shortfall): average of losses beyond VaR."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = int((1 - confidence) * len(sorted_r))
    tail = sorted_r[:max(cutoff, 1)]
    return -sum(tail) / len(tail)


def monte_carlo_var(
    returns: list[float],
    n_sims: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> float:
    """Monte Carlo VaR using bootstrapped returns."""
    if not returns:
        return 0.0
    rng = random.Random(seed)
    simulated = [rng.choice(returns) for _ in range(n_sims)]
    return historical_var(simulated, confidence)


def compute_risk_metrics(
    portfolio_values: list[float],
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> Optional[RiskMetrics]:
    """Compute full risk metrics from a time-series of portfolio NAV values."""
    if len(portfolio_values) < 10:
        logger.warning("risk_metrics.insufficient_data n=%d", len(portfolio_values))
        return None

    returns = [
        (portfolio_values[i] - portfolio_values[i - 1]) / portfolio_values[i - 1]
        for i in range(1, len(portfolio_values))
    ]

    var_95 = historical_var(returns, 0.95)
    var_99 = historical_var(returns, 0.99)
    cvar_95 = historical_cvar(returns, 0.95)
    cvar_99 = historical_cvar(returns, 0.99)

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    vol_daily = math.sqrt(variance)
    vol_annual = vol_daily * math.sqrt(252)

    return RiskMetrics(
        var_95=var_95,
        var_99=var_99,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
        worst_day_pct=min(returns) * 100,
        best_day_pct=max(returns) * 100,
        volatility_annual=vol_annual * 100,
        n_observations=len(returns),
    )
