"""
data/indicators.py — Shared technical indicator helpers.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> Optional[float]:
    """Return the most recent RSI(period) value, or None if insufficient data."""
    if len(series) < period + 1:
        return None
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    rsi      = 100 - (100 / (1 + rs))
    valid    = rsi.dropna()
    return float(valid.iloc[-1]) if not valid.empty else None
