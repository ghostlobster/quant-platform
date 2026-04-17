"""Tests for data/bars.py — AFML Ch 2 information-driven bars."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.bars import dollar_bars, tick_bars, volume_bars


def _ticks(
    n: int = 20,
    start: str = "2024-01-01 09:30:00",
    prices: list[float] | None = None,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="1s")
    if prices is None:
        prices = (100 + np.arange(n) * 0.1).tolist()
    if volumes is None:
        volumes = [100.0] * n
    return pd.DataFrame({"price": prices, "volume": volumes}, index=idx)


# ── dollar_bars ──────────────────────────────────────────────────────────────

def test_dollar_bars_output_has_ohlcv_schema():
    df = dollar_bars(_ticks(), threshold=10_000.0)
    assert list(df.columns) == [
        "Open", "High", "Low", "Close", "Volume", "DollarValue",
    ]


def test_dollar_bars_round_trip_preserves_total_notional():
    """Sum of bar DollarValue equals sum of tick (price × volume) for
    all bars *completed* before the final residual."""
    ticks = _ticks(n=50, prices=[100.0] * 50, volumes=[10.0] * 50)
    threshold = 2_500.0                     # exactly 2.5 ticks per bar
    bars = dollar_bars(ticks, threshold)
    assert not bars.empty
    # Every complete bar sums to a multiple of ~threshold (give or take
    # the per-tick granularity of 100 * 10 = 1000).
    assert bars["DollarValue"].sum() <= (ticks["price"] * ticks["volume"]).sum()


def test_dollar_bars_emits_single_bar_when_threshold_just_fits():
    ticks = _ticks(n=10, prices=[100.0] * 10, volumes=[1.0] * 10)
    # Total traded notional = 1000
    bars = dollar_bars(ticks, threshold=1_000.0)
    assert len(bars) == 1
    assert bars["Volume"].iloc[0] == pytest.approx(10.0)
    assert bars["DollarValue"].iloc[0] == pytest.approx(1_000.0)


def test_dollar_bars_empty_input_returns_empty_frame():
    bars = dollar_bars(pd.DataFrame(), threshold=1_000.0)
    assert bars.empty


def test_dollar_bars_below_threshold_returns_empty_frame():
    ticks = _ticks(n=3, prices=[100.0] * 3, volumes=[1.0] * 3)   # total $300
    assert dollar_bars(ticks, threshold=10_000.0).empty


def test_dollar_bars_threshold_must_be_positive():
    with pytest.raises(ValueError, match="threshold"):
        dollar_bars(_ticks(), threshold=0.0)


def test_dollar_bars_timestamp_is_last_tick_in_bar():
    ticks = _ticks(n=10, prices=[100.0] * 10, volumes=[1.0] * 10)
    bars = dollar_bars(ticks, threshold=500.0)   # every 5 ticks → 2 bars
    assert len(bars) == 2
    assert bars.index[0] == ticks.index[4]
    assert bars.index[1] == ticks.index[9]


def test_dollar_bars_ohlc_matches_first_last_min_max():
    prices = [100.0, 105.0, 98.0, 103.0, 101.0]
    ticks = _ticks(n=5, prices=prices, volumes=[1.0] * 5)
    bars = dollar_bars(ticks, threshold=505.0)   # total = 100+105+98+103+101
    assert len(bars) == 1
    row = bars.iloc[0]
    assert row["Open"] == pytest.approx(100.0)
    assert row["High"] == pytest.approx(105.0)
    assert row["Low"] == pytest.approx(98.0)
    assert row["Close"] == pytest.approx(101.0)


def test_dollar_bars_sorts_unsorted_index():
    ticks = _ticks(n=5, prices=[100.0] * 5, volumes=[1.0] * 5)
    shuffled = ticks.iloc[[2, 0, 4, 1, 3]]
    bars = dollar_bars(shuffled, threshold=500.0)
    assert len(bars) == 1
    assert bars.index[0] == ticks.index[4]


def test_dollar_bars_raises_on_missing_columns():
    bad = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-01-01")])
    with pytest.raises(ValueError, match="price"):
        dollar_bars(bad, threshold=100.0)


# ── volume_bars ──────────────────────────────────────────────────────────────

def test_volume_bars_splits_on_cumulative_volume():
    ticks = _ticks(n=6, prices=[100.0] * 6, volumes=[1.0] * 6)
    bars = volume_bars(ticks, threshold=2.0)       # every 2 ticks
    assert len(bars) == 3
    assert (bars["Volume"] == 2.0).all()


def test_volume_bars_threshold_must_be_positive():
    with pytest.raises(ValueError):
        volume_bars(_ticks(), threshold=-1.0)


def test_volume_bars_empty_returns_empty_frame():
    assert volume_bars(pd.DataFrame(), threshold=100.0).empty


def test_volume_bars_round_trip_volume_sums():
    ticks = _ticks(n=20, prices=[100.0] * 20, volumes=[0.5] * 20)
    threshold = 2.5
    bars = volume_bars(ticks, threshold=threshold)
    # Residual 4 ticks (total 10.0, 4 bars at vol 2.5 each = 10.0) — clean.
    assert bars["Volume"].sum() == pytest.approx(ticks["volume"].sum())


# ── tick_bars ────────────────────────────────────────────────────────────────

def test_tick_bars_uniform_chunks():
    ticks = _ticks(n=10, prices=list(range(10)), volumes=[1.0] * 10)
    bars = tick_bars(ticks, n=5)
    assert len(bars) == 2
    assert bars.iloc[0]["Open"] == pytest.approx(0)
    assert bars.iloc[0]["Close"] == pytest.approx(4)
    assert bars.iloc[1]["Open"] == pytest.approx(5)
    assert bars.iloc[1]["Close"] == pytest.approx(9)


def test_tick_bars_drops_partial_final_bar():
    ticks = _ticks(n=11)                 # 11 ticks, n=5 → 2 bars, 1 leftover
    bars = tick_bars(ticks, n=5)
    assert len(bars) == 2


def test_tick_bars_n_larger_than_data_returns_empty():
    ticks = _ticks(n=3)
    assert tick_bars(ticks, n=10).empty


def test_tick_bars_n_must_be_positive():
    with pytest.raises(ValueError):
        tick_bars(_ticks(), n=0)


def test_tick_bars_empty_input_returns_empty_frame():
    assert tick_bars(pd.DataFrame(), n=5).empty


def test_tick_bars_dollarvalue_column_populated():
    ticks = _ticks(n=4, prices=[100.0, 200.0, 150.0, 50.0], volumes=[1.0] * 4)
    bars = tick_bars(ticks, n=2)
    assert bars.iloc[0]["DollarValue"] == pytest.approx(300.0)
    assert bars.iloc[1]["DollarValue"] == pytest.approx(200.0)
