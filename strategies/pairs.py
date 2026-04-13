"""Pairs trading strategy based on cointegration and z-score mean reversion."""
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PairsResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    is_cointegrated: bool
    p_value: float
    current_zscore: float
    signal: str          # 'buy_spread', 'sell_spread', 'close', 'hold'
    half_life: float     # mean reversion half-life in days


def compute_hedge_ratio(prices_a: pd.Series, prices_b: pd.Series) -> float:
    """OLS regression of price_a on price_b. Returns beta (hedge ratio)."""
    from numpy.linalg import lstsq
    X = np.column_stack([prices_b.values, np.ones(len(prices_b))])
    beta, _, _, _ = lstsq(X, prices_a.values, rcond=None)
    return float(beta[0])


def compute_spread(prices_a: pd.Series, prices_b: pd.Series,
                   hedge_ratio: Optional[float] = None) -> pd.Series:
    """Compute spread = price_a - hedge_ratio * price_b."""
    if hedge_ratio is None:
        hedge_ratio = compute_hedge_ratio(prices_a, prices_b)
    return prices_a - hedge_ratio * prices_b


def compute_zscore(spread: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of the spread."""
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    return ((spread - mean) / std.replace(0, np.nan)).fillna(0)


def compute_half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life: how quickly the spread reverts to mean.
    Uses OLS regression of delta_spread on lagged spread.
    """
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    # Align
    common = lag.index.intersection(delta.index)
    lag, delta = lag[common], delta[common]
    if len(lag) < 10:
        return float('inf')
    X = np.column_stack([lag.values, np.ones(len(lag))])
    from numpy.linalg import lstsq
    beta, _, _, _ = lstsq(X, delta.values, rcond=None)
    lam = beta[0]
    if lam >= 0:
        return float('inf')
    return float(-np.log(2) / lam)


def test_cointegration(prices_a: pd.Series, prices_b: pd.Series,
                       significance: float = 0.05) -> Tuple[bool, float]:
    """
    Engle-Granger cointegration test using ADF on the residuals.
    Returns (is_cointegrated, p_value).
    Falls back to a simple correlation-based check if statsmodels not available.
    """
    try:
        from statsmodels.tsa.stattools import coint
        _, p_value, _ = coint(prices_a, prices_b)
        return p_value < significance, float(p_value)
    except ImportError:
        # Fallback: check if spread is stationary via simple variance ratio
        spread = compute_spread(prices_a, prices_b)
        # If spread has low variance relative to components, treat as cointegrated
        spread_std = spread.std()
        component_std = (prices_a.std() + prices_b.std()) / 2
        ratio = spread_std / component_std if component_std > 0 else 1.0
        p_approx = ratio  # rough proxy — not a real p-value
        return ratio < 0.3, float(p_approx)


def analyse_pair(prices_a: pd.Series, prices_b: pd.Series,
                 ticker_a: str = "A", ticker_b: str = "B",
                 entry_z: float = 2.0, exit_z: float = 0.5,
                 window: int = 20) -> PairsResult:
    """Full pair analysis: cointegration, hedge ratio, z-score, signal."""
    is_coint, p_val = test_cointegration(prices_a, prices_b)
    hedge = compute_hedge_ratio(prices_a, prices_b)
    spread = compute_spread(prices_a, prices_b, hedge)
    zscore = compute_zscore(spread, window)
    half_life = compute_half_life(spread)
    current_z = float(zscore.iloc[-1]) if len(zscore) > 0 else 0.0

    if not is_coint or half_life > 60:
        signal = 'hold'
    elif current_z > entry_z:
        signal = 'sell_spread'   # spread too wide: short A, long B
    elif current_z < -entry_z:
        signal = 'buy_spread'    # spread too narrow: long A, short B
    elif abs(current_z) < exit_z:
        signal = 'close'
    else:
        signal = 'hold'

    return PairsResult(
        ticker_a=ticker_a, ticker_b=ticker_b,
        hedge_ratio=hedge, is_cointegrated=is_coint,
        p_value=p_val, current_zscore=current_z,
        signal=signal, half_life=half_life,
    )


def pairs_backtest(prices_a: pd.Series, prices_b: pd.Series,
                   entry_z: float = 2.0, exit_z: float = 0.5,
                   window: int = 20) -> dict:
    """
    Backtest the pairs strategy on historical data.
    Trades the spread: buy when z < -entry_z, sell when z > entry_z, exit at |z| < exit_z.
    """
    hedge = compute_hedge_ratio(prices_a.iloc[:window*2], prices_b.iloc[:window*2])
    spread = compute_spread(prices_a, prices_b, hedge)
    zscore = compute_zscore(spread, window)

    position = 0   # 1 = long spread, -1 = short spread
    entry_spread = 0.0
    trades = []

    for i in range(window + 1, len(zscore)):
        z = float(zscore.iloc[i])
        s = float(spread.iloc[i])
        if pd.isna(z):
            continue

        if position == 0:
            if z < -entry_z:
                position = 1
                entry_spread = s
            elif z > entry_z:
                position = -1
                entry_spread = s
        else:
            if abs(z) < exit_z:
                pnl = position * (s - entry_spread)
                trades.append(pnl)
                position = 0

    if not trades:
        return {"total_pnl": 0.0, "num_trades": 0, "win_rate": 0.0}

    total_pnl = float(sum(trades))
    win_rate = float(sum(1 for t in trades if t > 0) / len(trades))
    return {
        "total_pnl": total_pnl,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade_pnl": float(np.mean(trades)),
        "hedge_ratio": hedge,
    }
