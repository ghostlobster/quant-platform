import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.engine import BacktestResult, run_backtest, run_signal_backtest


def make_ohlcv(n=120):
    np.random.seed(99)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open":   close * 0.99,
        "High":   close * 1.01,
        "Low":    close * 0.98,
        "Close":  close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)


def test_sma_crossover_returns_result():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="TEST",
                          start_date=None, end_date=None)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.num_trades, int)
    assert result.num_trades >= 0
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown_pct, float)


def test_rsi_mean_revert_returns_result():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="rsi_mean_revert", ticker="TEST",
                          start_date=None, end_date=None)
    assert isinstance(result, BacktestResult)
    assert result.num_trades >= 0


def test_max_drawdown_non_positive():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="TEST",
                          start_date=None, end_date=None)
    assert result.max_drawdown_pct <= 0.0


def test_ticker_stored_uppercase():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="aapl")
    assert result.ticker == "AAPL"


def test_equity_curve_is_dataframe():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="TEST")
    assert isinstance(result.equity_curve, pd.DataFrame)


def test_raises_on_insufficient_data():
    small = make_ohlcv(30)
    with pytest.raises(ValueError, match="Not enough data"):
        run_backtest(small, strategy="sma_crossover")


def test_raises_on_unknown_strategy():
    df = make_ohlcv(120)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_backtest(df, strategy="unknown_strategy")


def test_sortino_calmar_are_floats():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="TEST")
    assert isinstance(result.sortino_ratio, float)
    assert isinstance(result.calmar_ratio, float)


def test_stop_losses_triggered_is_non_negative_int():
    df = make_ohlcv(120)
    result = run_backtest(df, strategy="sma_crossover", ticker="TEST")
    assert isinstance(result.stop_losses_triggered, int)
    assert result.stop_losses_triggered >= 0


# ── run_signal_backtest tests ─────────────────────────────────────────────────

def test_signal_backtest_all_long_returns_result():
    """Continuous +1 signal should produce a result with num_trades >= 1."""
    df = make_ohlcv(120)
    signals = pd.Series(1.0, index=df.index)
    result = run_signal_backtest(df, signals, strategy_name="ml_signal", ticker="TEST")
    assert isinstance(result, BacktestResult)
    assert result.strategy == "ml_signal"
    assert result.ticker == "TEST"
    assert result.num_trades >= 1
    assert isinstance(result.total_return_pct, float)
    assert isinstance(result.sharpe_ratio, float)
    assert isinstance(result.max_drawdown_pct, float)
    assert result.max_drawdown_pct <= 0.0


def test_signal_backtest_all_flat_no_trades():
    """All-zero (flat) signals should produce zero trades."""
    df = make_ohlcv(120)
    signals = pd.Series(0.0, index=df.index)
    result = run_signal_backtest(df, signals, ticker="FLAT")
    assert result.num_trades == 0
    assert result.total_return_pct == 0.0


def test_signal_backtest_positive_scores_treated_as_long():
    """Any positive score (including fractional) drives a long entry."""
    df = make_ohlcv(120)
    # Alternating small positive/negative scores
    scores = np.where(np.arange(120) % 20 < 10, 0.1, -0.1)
    signals = pd.Series(scores, index=df.index)
    result = run_signal_backtest(df, signals, ticker="SCORE")
    assert result.num_trades >= 1


def test_signal_backtest_returns_backtest_result_type():
    df = make_ohlcv(120)
    signals = pd.Series(np.random.randn(120), index=df.index)
    result = run_signal_backtest(df, signals)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.DataFrame)
    assert "Equity" in result.equity_curve.columns
    assert "BuyHold" in result.equity_curve.columns


def test_signal_backtest_raises_on_insufficient_data():
    small = make_ohlcv(1)
    signals = pd.Series([1.0], index=small.index)
    with pytest.raises(ValueError, match="Not enough data"):
        run_signal_backtest(small, signals)
