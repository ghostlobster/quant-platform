"""
analysis/structural_breaks.py — AFML Ch 17 structural-break detection.

The 4-state regime classifier in :mod:`analysis.regime` captures a small
menu of named market states.  It doesn't localise *when* the state
changed.  AFML Ch 17.4 proposes a symmetric CUSUM filter that fires on
any bar where the cumulative deviation from a reference level exceeds a
threshold — useful for gating triple-barrier labels so we only label
genuinely informative events.

Public surface
--------------
:func:`cusum_events`
    Returns a :class:`pd.DatetimeIndex` of event timestamps.  Drop-in
    replacement for the default ``events`` argument to
    :func:`analysis.triple_barrier.triple_barrier_labels`.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 17.4.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cusum_events(series: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Symmetric CUSUM filter over an ordered price / log-price series.

    The classic filter tracks two cumulative sums — one above the
    previous event and one below — and fires whenever either crosses
    ``threshold``.  After firing, both sums are reset to zero.

    Parameters
    ----------
    series :
        DatetimeIndex-ordered float series.  Most commonly log-prices
        (``np.log(close)``) so that the threshold expresses a constant
        relative move regardless of price level.
    threshold :
        Absolute deviation at which a new event is recorded.  Must be
        positive.

    Returns
    -------
    :class:`pd.DatetimeIndex` of event timestamps (a subset of
    ``series.index``).  May be empty when the series never drifts far
    enough.

    Raises
    ------
    ValueError when ``threshold`` is not positive.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    if series is None or len(series) < 2:
        return pd.DatetimeIndex([])

    values = np.asarray(series.to_numpy(dtype=float))
    idx = series.index

    s_pos = 0.0
    s_neg = 0.0
    events: list = []

    diffs = np.diff(values)
    # AFML iterates over bar-over-bar differences; we skip the first index
    # so the output can never include the very first timestamp (it has no
    # preceding reference).
    for i, d in enumerate(diffs, start=1):
        s_pos = max(0.0, s_pos + d)
        s_neg = min(0.0, s_neg + d)
        if s_neg < -threshold:
            s_neg = 0.0
            events.append(idx[i])
        elif s_pos > threshold:
            s_pos = 0.0
            events.append(idx[i])

    return pd.DatetimeIndex(events)


def cusum_events_from_prices(
    prices: pd.Series,
    threshold: float,
    use_log: bool = True,
) -> pd.DatetimeIndex:
    """Convenience wrapper: transform raw prices before applying CUSUM.

    When ``use_log=True`` (default), works on ``log(prices)`` so the
    threshold expresses a constant percentage-style move.  Otherwise
    uses the raw price series directly.
    """
    if prices is None or len(prices) < 2:
        return pd.DatetimeIndex([])
    series = np.log(prices.astype(float)) if use_log else prices.astype(float)
    return cusum_events(series, threshold=threshold)
