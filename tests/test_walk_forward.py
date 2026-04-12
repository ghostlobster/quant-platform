import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.walk_forward import walk_forward, WalkForwardResult


def make_ohlcv(n=300, seed=7):
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)


def test_walk_forward_returns_result():
    df = make_ohlcv(300)
    wf = walk_forward(df, strategy="sma_crossover", train_periods=60, test_periods=60, step=30)
    assert isinstance(wf, WalkForwardResult)
    assert len(wf.windows) > 0

def test_consistency_score_range():
    df = make_ohlcv(300)
    wf = walk_forward(df, strategy="sma_crossover", train_periods=60, test_periods=60, step=30)
    assert 0.0 <= wf.consistency_score <= 1.0

def test_avg_sharpe_is_float():
    df = make_ohlcv(300)
    wf = walk_forward(df, strategy="rsi_mean_revert", train_periods=60, test_periods=60, step=30)
    assert isinstance(wf.avg_sharpe, float)

def test_short_df_returns_empty():
    df = make_ohlcv(20)
    wf = walk_forward(df, train_periods=60, test_periods=60, step=30)
    assert wf.total_trades == 0
