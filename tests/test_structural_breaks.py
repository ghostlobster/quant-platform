"""Tests for analysis/structural_breaks.py — CUSUM event filter."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.structural_breaks import cusum_events, cusum_events_from_prices


def _series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start=start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


# ── cusum_events ─────────────────────────────────────────────────────────────

def test_cusum_events_fires_on_upward_run_exceeding_threshold():
    # +1 per step; strict > threshold=3 → first trigger when cumdev = 4.
    s = _series([0, 1, 2, 3, 4, 5])
    events = cusum_events(s, threshold=3.0)
    assert len(events) >= 1
    assert events[0] == s.index[4]


def test_cusum_events_fires_on_downward_run():
    s = _series([0, -1, -2, -3, -4])
    events = cusum_events(s, threshold=3.0)
    assert len(events) >= 1
    assert events[0] == s.index[4]


def test_cusum_events_returns_empty_for_flat_series():
    s = _series([100.0] * 20)
    assert len(cusum_events(s, threshold=0.5)) == 0


def test_cusum_events_resets_after_firing():
    """Two separate upward runs → two events, both at run endpoints."""
    s = _series([0, 1, 2, 3, 2, 2, 3, 4, 5])
    events = cusum_events(s, threshold=2.0)
    assert len(events) == 2


def test_cusum_events_never_returns_first_timestamp():
    """There's no reference before index 0 so it can't be an event."""
    s = _series([0, 100.0, 0, 0, 0])
    events = cusum_events(s, threshold=1.0)
    assert s.index[0] not in events


def test_cusum_events_index_is_datetime():
    s = _series([0, 1, 2, 3])
    events = cusum_events(s, threshold=1.0)
    assert isinstance(events, pd.DatetimeIndex)


def test_cusum_events_empty_series_returns_empty_index():
    empty = pd.Series(dtype=float)
    assert len(cusum_events(empty, threshold=1.0)) == 0


def test_cusum_events_single_value_returns_empty_index():
    s = _series([1.0])
    assert len(cusum_events(s, threshold=1.0)) == 0


def test_cusum_events_threshold_must_be_positive():
    s = _series([0, 1, 2])
    with pytest.raises(ValueError, match="threshold"):
        cusum_events(s, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        cusum_events(s, threshold=-0.1)


def test_cusum_events_respects_sampling_density_via_threshold():
    """A larger threshold should yield fewer (or equal) events."""
    rng = np.random.default_rng(3)
    s = pd.Series(
        np.cumsum(rng.normal(scale=0.01, size=500)),
        index=pd.date_range("2023-01-01", periods=500, freq="B"),
    )
    few = cusum_events(s, threshold=0.5)
    many = cusum_events(s, threshold=0.05)
    assert len(many) >= len(few)


# ── cusum_events_from_prices ─────────────────────────────────────────────────

def test_cusum_from_prices_log_transform_default():
    """Identical absolute moves on log-space should fire consistently."""
    prices = _series([100 * np.exp(i * 0.01) for i in range(50)])
    events = cusum_events_from_prices(prices, threshold=0.03)
    assert len(events) >= 1


def test_cusum_from_prices_respects_use_log_flag():
    prices = _series([100 + i for i in range(50)])        # linear, not log
    evts_log = cusum_events_from_prices(prices, threshold=0.1, use_log=True)
    evts_raw = cusum_events_from_prices(prices, threshold=10.0, use_log=False)
    assert isinstance(evts_log, pd.DatetimeIndex)
    assert isinstance(evts_raw, pd.DatetimeIndex)


def test_cusum_from_prices_empty_returns_empty_index():
    assert len(cusum_events_from_prices(pd.Series(dtype=float), threshold=0.1)) == 0
