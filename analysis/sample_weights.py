"""
analysis/sample_weights.py — López de Prado Ch 4 sample uniqueness weighting.

Overlapping labels — common with triple-barrier labelling that spans
multiple days — violate the i.i.d. assumption most ML estimators make.
Samples whose label windows overlap with many others are correlated and
should carry less weight; samples with isolated windows are more
informative and should carry more.

This module provides the three primitives from AFML Chapter 4:

  * :func:`num_co_events`  — count how many open labels cover each bar.
  * :func:`sample_uniqueness` — per-event uniqueness in ``[0, 1]``.
  * :func:`sequential_bootstrap` — resampling that prefers unique events.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 4.3-4.5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def num_co_events(
    close_idx: pd.DatetimeIndex,
    events: pd.Series,
) -> pd.Series:
    """Count the number of concurrently-open events covering each bar.

    Parameters
    ----------
    close_idx :
        All bar timestamps for the underlying price series (ascending).
    events :
        ``pd.Series`` indexed by event *start* timestamp whose values are
        the event *end* timestamps (``t1`` from
        :func:`analysis.triple_barrier.triple_barrier_labels`).  Events
        with NaN end are dropped.

    Returns
    -------
    ``pd.Series`` indexed by ``close_idx`` counting how many events are
    open at each bar (``0`` when no events overlap that bar).
    """
    if events is None or len(events) == 0:
        return pd.Series(0, index=close_idx, dtype=int)

    # Clean: drop events with NaN t1, clamp t1 to the last available bar.
    events = events.dropna()
    if events.empty:
        return pd.Series(0, index=close_idx, dtype=int)

    start_series = pd.Series(events.index, index=events.index)
    clamped_end = events.where(events <= close_idx[-1], other=close_idx[-1])

    count = pd.Series(0, index=close_idx, dtype=int)
    for start, end in zip(start_series.values, clamped_end.values):
        if pd.isna(start) or pd.isna(end):
            continue
        left = close_idx.searchsorted(start, side="left")
        right = close_idx.searchsorted(end, side="right")
        if right <= left:
            continue
        count.iloc[left:right] += 1
    return count


def sample_uniqueness(
    events: pd.Series,
    co_events: pd.Series,
) -> pd.Series:
    """Mean per-bar uniqueness over each event's active window.

    For each event ``(t, t1)`` the uniqueness at a bar ``b`` is
    ``1 / num_co_events[b]``.  The event's overall uniqueness is the
    average of that signal across the bars it covers.  Fully isolated
    events get ``1.0``; an event that overlaps ``N`` others uniformly
    gets ``1 / N``.

    Parameters
    ----------
    events :
        ``pd.Series`` indexed by event start timestamp whose values are
        event end timestamps (same shape as :func:`num_co_events`).
    co_events :
        Output of :func:`num_co_events` for the same price index.

    Returns
    -------
    ``pd.Series`` indexed by event start timestamp with uniqueness
    in ``[0, 1]``.
    """
    if events is None or len(events) == 0 or co_events is None or co_events.empty:
        return pd.Series(dtype=float)

    events = events.dropna()
    close_idx = co_events.index

    out = pd.Series(np.nan, index=events.index, dtype=float)
    for start, end in events.items():
        left = close_idx.searchsorted(start, side="left")
        right = close_idx.searchsorted(end, side="right")
        if right <= left:
            continue
        slice_ = co_events.iloc[left:right].astype(float)
        # Where the bar count is 0 — shouldn't happen for an event that
        # just declared itself open — fall back to 1 to avoid div-by-zero.
        slice_ = slice_.replace(0, 1.0)
        out.loc[start] = float((1.0 / slice_).mean())

    return out.dropna()


def sequential_bootstrap(
    events: pd.Series,
    size: int | None = None,
    seed: int = 42,
) -> list:
    """Sequentially bootstrap event indices with inverse-concurrency probs.

    Implements AFML Code Snippet 4.5: at each draw, each still-available
    event is weighted by its *average* uniqueness against the already-
    selected set.  Drawing an event reduces the probability of drawing
    another that overlaps it.

    Parameters
    ----------
    events :
        ``pd.Series`` indexed by event start, values = event end
        timestamps.  NaN ends are dropped.
    size :
        Number of draws.  Defaults to ``len(events)``.
    seed :
        RNG seed for reproducibility.

    Returns
    -------
    ``list`` of length ``size`` of event start timestamps sampled with
    replacement.  Empty list when the input has no usable events.
    """
    if events is None or len(events) == 0:
        return []

    events = events.dropna()
    n = len(events)
    if n == 0:
        return []

    size = int(size) if size is not None else n
    if size <= 0:
        return []

    # Build an indicator matrix: rows = events, cols = bars covered.
    start_values = events.index
    all_bars = pd.Index(sorted(set(start_values).union(set(events.values))))
    ind = pd.DataFrame(0, index=start_values, columns=all_bars, dtype=int)
    for start, end in events.items():
        mask = (all_bars >= start) & (all_bars <= end)
        ind.loc[start, mask] = 1

    rng = np.random.default_rng(seed)
    picks: list = []
    # Concurrency so far — starts at zero across all bars.
    concurrency = pd.Series(0.0, index=all_bars)

    for _ in range(size):
        # For every candidate event, compute its average uniqueness if
        # we were to add it next (AFML Eq 4.3).
        inc = ind.add(concurrency, axis=1)                 # bars × events
        # Uniqueness = 1 / (concurrency + 1) averaged over the event's own bars.
        covered = ind.astype(bool)
        unique_contrib = 1.0 / inc.where(covered)          # NaN outside event
        avg_unique = unique_contrib.mean(axis=1).fillna(0.0)
        total = float(avg_unique.sum())
        if total == 0.0:
            probs = np.full(n, 1.0 / n)
        else:
            probs = (avg_unique / total).values
        choice = rng.choice(n, p=probs)
        picked = start_values[choice]
        picks.append(picked)
        # Update concurrency: add the picked event's coverage mask.
        concurrency = concurrency + ind.iloc[choice]

    return picks


def weights_for_train_index(
    train_index: pd.MultiIndex,
    events: pd.Series,
    close_idx: pd.DatetimeIndex,
) -> np.ndarray:
    """Build a dense ``sample_weight`` array aligned with an ML training frame.

    The ML pipeline trains on a MultiIndex ``(date, ticker)`` matrix;
    sample uniqueness is defined per *event start date*.  This helper
    maps the per-date uniqueness onto each (date, ticker) row.

    Parameters
    ----------
    train_index :
        The ``MultiIndex`` of the training DataFrame
        (``(date, ticker)``).
    events :
        Event series (start → end) over ``close_idx``.
    close_idx :
        Bar index used to compute co-events.

    Returns
    -------
    ``np.ndarray`` of length ``len(train_index)`` with non-negative
    weights.  Rows whose date is missing from ``events`` get weight
    ``1.0`` so they don't silently disappear.
    """
    co = num_co_events(close_idx, events)
    unique = sample_uniqueness(events, co)
    if unique.empty:
        return np.ones(len(train_index), dtype=float)
    # Normalise so the weight vector's mean is 1 → doesn't rescale the
    # effective learning rate relative to the no-weight baseline.
    unique = unique / unique.mean()

    dates = train_index.get_level_values("date")
    mapped = pd.Series(dates).map(unique).to_numpy()
    # Any date outside the event set defaults to 1.0.
    mapped = np.where(np.isnan(mapped), 1.0, mapped)
    return mapped.astype(float)
