import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch

import plotly.graph_objects as go

from backtester.walk_forward import WalkForwardResult, build_walk_forward_chart, walk_forward


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


def test_walk_forward_segment_exception_skipped():
    """run_backtest raising in one segment must not abort the whole walk."""
    df = make_ohlcv(300)
    call_count = {"n": 0}

    def occasionally_fail(segment, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise ValueError("injected failure")
        from backtester.engine import run_backtest as _real
        return _real(segment, **kwargs)

    with patch("backtester.walk_forward.run_backtest", side_effect=occasionally_fail):
        wf = walk_forward(df, strategy="sma_crossover", train_periods=60, test_periods=60, step=30)

    # At least one successful window despite the first failure
    assert isinstance(wf, WalkForwardResult)
    assert len(wf.windows) >= 1


# --- build_walk_forward_chart ---

def test_build_chart_empty_returns_empty_figure():
    fig = build_walk_forward_chart(WalkForwardResult())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_build_chart_with_results():
    df = make_ohlcv(300)
    wf = walk_forward(df, strategy="sma_crossover", train_periods=60, test_periods=60, step=30)
    fig = build_walk_forward_chart(wf)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
