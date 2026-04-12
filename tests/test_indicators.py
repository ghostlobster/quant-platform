import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.indicators import compute_rsi


def make_price_series(n=60):
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.Series(prices)


def test_rsi_returns_float_for_long_series():
    series = make_price_series(60)
    result = compute_rsi(series, period=14)
    assert result is not None
    assert isinstance(result, float)
    assert 0.0 <= result <= 100.0


def test_rsi_returns_none_for_short_series():
    series = make_price_series(5)
    result = compute_rsi(series, period=14)
    assert result is None


def test_rsi_returns_none_for_exactly_period_length():
    # period+1 rows required; exactly period rows → None
    series = pd.Series([float(i) for i in range(14)])
    assert compute_rsi(series, period=14) is None


def test_rsi_range():
    # RSI must always be in [0, 100]
    for seed in range(5):
        np.random.seed(seed)
        prices = 100 + np.cumsum(np.random.randn(50) * 1.0)
        result = compute_rsi(pd.Series(prices), period=14)
        if result is not None:
            assert 0.0 <= result <= 100.0


def test_rsi_high_for_rising_prices():
    # Upward-biased random walk with enough noise to produce some down-days
    np.random.seed(7)
    steps = np.random.randn(30) * 1.0 + 0.8   # mean step +0.8, std 1.0 → some losses
    prices = pd.Series(100.0 + np.cumsum(steps))
    result = compute_rsi(prices)
    assert result is not None
    assert result > 50.0


def test_rsi_low_for_falling_prices():
    prices = pd.Series([200.0 - i * 1.5 for i in range(30)])
    result = compute_rsi(prices)
    assert result is not None
    assert result < 50.0
