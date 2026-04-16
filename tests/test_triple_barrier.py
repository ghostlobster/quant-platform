"""Unit tests for analysis/triple_barrier.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.triple_barrier import (
    daily_volatility,
    triple_barrier_labels,
)


def _series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_daily_volatility_is_positive_for_noisy_returns():
    prices = _series(100 + np.cumsum(np.random.RandomState(0).randn(60)))
    vol = daily_volatility(prices, span=10)
    assert (vol.dropna() >= 0).all()
    assert vol.iloc[-1] > 0


def test_monotone_up_series_hits_profit_take():
    # Steady up-trend of ~1% per day — PT barrier trips first for all but the
    # last event whose forward window is trivially short.
    prices = _series(100 * (1.01 ** np.arange(30)))
    labels = triple_barrier_labels(
        prices, pt_sl=(1.0, 1.0), num_days=10, vol_span=5,
    )
    assert (labels["bin"].iloc[:-1] == 1).all()


def test_monotone_down_series_hits_stop_loss():
    prices = _series(100 * (0.99 ** np.arange(30)))
    labels = triple_barrier_labels(
        prices, pt_sl=(1.0, 1.0), num_days=10, vol_span=5,
    )
    assert (labels["bin"].iloc[:-1] == -1).all()


def test_flat_series_times_out_to_zero():
    prices = _series(np.full(30, 100.0))
    labels = triple_barrier_labels(
        prices, pt_sl=(2.0, 2.0), num_days=3, vol_span=5,
    )
    # Vol is zero → all barriers disabled → every event times out with 0 return.
    assert (labels["bin"] == 0).all()


def test_explicit_events_subset_is_respected():
    prices = _series(100 * (1.01 ** np.arange(30)))
    events = prices.index[[5, 10, 15]]
    labels = triple_barrier_labels(
        prices, events=events, pt_sl=(1.0, 1.0), num_days=5, vol_span=3,
    )
    assert list(labels.index) == list(events)
    assert len(labels) == 3


def test_empty_price_series_returns_empty_frame():
    empty = pd.Series(dtype=float)
    labels = triple_barrier_labels(empty)
    assert labels.empty
    assert list(labels.columns) == ["t1", "ret", "bin", "target"]


def test_columns_and_dtypes():
    prices = _series(100 + np.random.RandomState(1).randn(20).cumsum())
    labels = triple_barrier_labels(prices, pt_sl=(1.5, 1.5), num_days=3)
    assert set(labels.columns) == {"t1", "ret", "bin", "target"}
    assert labels["bin"].isin([-1, 0, 1]).all()


def test_min_ret_filters_low_vol_events():
    prices = _series(np.full(20, 100.0))  # zero volatility
    labels = triple_barrier_labels(
        prices, pt_sl=(1.0, 1.0), num_days=3, vol_span=5, min_ret=0.001,
    )
    assert labels.empty
