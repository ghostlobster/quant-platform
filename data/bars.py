"""
data/bars.py — AFML Ch 2 information-driven bars.

Time-bar OHLCV (daily, hourly) samples information unevenly: a low-
activity day carries the same weight as a high-activity day even
though it contains far less price discovery.  López de Prado advocates
*information-driven* bars — dollar, volume, and tick bars — that
aggregate tick data until a fixed amount of traded notional, traded
volume, or tick count accumulates.  Resulting bars have more uniform
information content, which usually improves the i.i.d.-ness of features
downstream.

This module ships three aggregators that all accept a raw tick
DataFrame:

  * :func:`dollar_bars` — aggregate until `threshold` dollars trade.
  * :func:`volume_bars` — aggregate until `threshold` shares trade.
  * :func:`tick_bars`   — aggregate every ``n`` ticks.

All three preserve the standard OHLCV schema so they plug directly into
the existing feature pipeline.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 2.3-2.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_PRICE_COL = "price"
_VOLUME_COL = "volume"


def _validate(tick_df: pd.DataFrame) -> pd.DataFrame:
    if tick_df is None or tick_df.empty:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "DollarValue"]
        )
    if _PRICE_COL not in tick_df.columns or _VOLUME_COL not in tick_df.columns:
        raise ValueError(
            f"tick_df must contain '{_PRICE_COL}' and '{_VOLUME_COL}' columns"
        )
    if not tick_df.index.is_monotonic_increasing:
        tick_df = tick_df.sort_index()
    return tick_df


def _aggregate_bar(bar_ticks: pd.DataFrame) -> dict:
    price = bar_ticks[_PRICE_COL].to_numpy(dtype=float)
    volume = bar_ticks[_VOLUME_COL].to_numpy(dtype=float)
    return {
        "Open":        float(price[0]),
        "High":        float(price.max()),
        "Low":         float(price.min()),
        "Close":       float(price[-1]),
        "Volume":      float(volume.sum()),
        "DollarValue": float((price * volume).sum()),
    }


def _bars_from_cuts(tick_df: pd.DataFrame, cuts: list[int]) -> pd.DataFrame:
    """Turn a list of integer cut points into an OHLCV DataFrame.

    ``cuts`` lists the last tick index (inclusive) in each bar.  The
    resulting bar is timestamped with the last tick's index.
    """
    if not cuts:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "DollarValue"]
        )

    rows: list[dict] = []
    timestamps: list = []
    start = 0
    for end in cuts:
        bar_slice = tick_df.iloc[start : end + 1]
        rows.append(_aggregate_bar(bar_slice))
        timestamps.append(bar_slice.index[-1])
        start = end + 1

    out = pd.DataFrame(rows, index=pd.Index(timestamps, name="timestamp"))
    return out[["Open", "High", "Low", "Close", "Volume", "DollarValue"]]


def dollar_bars(tick_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate ticks into bars of (approximately) equal traded dollar value.

    Parameters
    ----------
    tick_df :
        DataFrame indexed by timestamp with ``price`` and ``volume``
        columns.  Rows represent individual ticks (or trade summaries).
    threshold :
        Dollar value at which a new bar is emitted.  Must be positive.

    Returns
    -------
    ``pd.DataFrame`` with columns ``Open, High, Low, Close, Volume,
    DollarValue`` indexed by the timestamp of the last tick in each
    bar.  Any residual ticks whose cumulative dollar value never
    crosses the threshold are silently dropped so the round-trip
    invariant (``sum(dollar) == sum(tick_dollar)`` for complete bars)
    holds.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    tick_df = _validate(tick_df)
    if tick_df.empty:
        return tick_df

    dollar = (
        tick_df[_PRICE_COL].to_numpy(dtype=float)
        * tick_df[_VOLUME_COL].to_numpy(dtype=float)
    )
    cum = np.cumsum(dollar)
    # Find positions where cumulative dollar crosses each multiple of threshold.
    milestones = np.arange(threshold, cum[-1] + 1e-12, threshold)
    if milestones.size == 0:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "DollarValue"]
        )
    cuts = np.searchsorted(cum, milestones, side="left").tolist()
    # Dedup consecutive identical cuts (can happen when a single huge
    # trade blows past multiple thresholds in one tick).
    unique_cuts: list[int] = []
    for c in cuts:
        if not unique_cuts or c != unique_cuts[-1]:
            unique_cuts.append(int(c))
    return _bars_from_cuts(tick_df, unique_cuts)


def volume_bars(tick_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aggregate ticks into bars of (approximately) equal traded volume."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    tick_df = _validate(tick_df)
    if tick_df.empty:
        return tick_df

    volume = tick_df[_VOLUME_COL].to_numpy(dtype=float)
    cum = np.cumsum(volume)
    milestones = np.arange(threshold, cum[-1] + 1e-12, threshold)
    if milestones.size == 0:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "DollarValue"]
        )
    cuts = np.searchsorted(cum, milestones, side="left").tolist()
    unique_cuts: list[int] = []
    for c in cuts:
        if not unique_cuts or c != unique_cuts[-1]:
            unique_cuts.append(int(c))
    return _bars_from_cuts(tick_df, unique_cuts)


def tick_bars(tick_df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Aggregate every ``n`` ticks into a bar.

    Unlike dollar / volume bars, tick bars have deterministic lengths
    (except the final partial bar, which is dropped to match the other
    aggregators' round-trip invariant).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    tick_df = _validate(tick_df)
    if tick_df.empty:
        return tick_df

    total = len(tick_df)
    num_bars = total // n
    if num_bars == 0:
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume", "DollarValue"]
        )
    cuts = [(i + 1) * n - 1 for i in range(num_bars)]
    return _bars_from_cuts(tick_df, cuts)
