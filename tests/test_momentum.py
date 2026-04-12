import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.momentum import compute_momentum_score, momentum_signals, momentum_backtest, MomentumSignal


def make_ohlcv(n=200, seed=42, trend=0.001):
    np.random.seed(seed)
    close = 100 * np.cumprod(1 + np.random.normal(trend, 0.015, n))
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.02,
        "Low": close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=idx)


def test_momentum_score_range():
    df = make_ohlcv()
    score = compute_momentum_score(df)
    valid = score.dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all()


def test_momentum_signals_returns_list():
    df = make_ohlcv(200)
    sigs = momentum_signals(df, ticker="TEST")
    assert isinstance(sigs, list)
    for s in sigs:
        assert isinstance(s, MomentumSignal)
        assert s.signal in ('buy', 'sell')


def test_signals_alternate_buy_sell():
    df = make_ohlcv(200)
    sigs = momentum_signals(df)
    if len(sigs) >= 2:
        for i in range(1, len(sigs)):
            assert sigs[i].signal != sigs[i-1].signal


def test_momentum_backtest_returns_dict():
    df = make_ohlcv(200)
    result = momentum_backtest(df)
    assert "total_return" in result
    assert "num_trades" in result
    assert "win_rate" in result
    assert 0.0 <= result["win_rate"] <= 1.0


def test_short_df_returns_zero():
    df = make_ohlcv(10)
    result = momentum_backtest(df)
    assert result["num_trades"] == 0


def make_reversing_ohlcv(n=200):
    """Data with a strong uptrend then downtrend to guarantee buy+sell signals."""
    np.random.seed(0)
    half = n // 2
    up   = 100 * np.cumprod(1 + np.random.normal(0.01, 0.005, half))
    down = up[-1] * np.cumprod(1 + np.random.normal(-0.012, 0.005, n - half))
    close = np.concatenate([up, down])
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": np.ones(n) * 1_000_000},
        index=idx,
    )


def test_sell_signal_generated():
    """Reversing data should produce at least one sell signal (covers lines 76-81)."""
    df = make_reversing_ohlcv()
    sigs = momentum_signals(df, ticker="REV",
                            buy_threshold=0.1, sell_threshold=-0.1)
    sell_sigs = [s for s in sigs if s.signal == "sell"]
    assert len(sell_sigs) >= 1, "Expected at least one sell signal with reversing data"
    assert all(s.strength >= 0 for s in sell_sigs)


def test_backtest_records_trades():
    """Reversing data should yield at least one completed trade (covers lines 98-116)."""
    df = make_reversing_ohlcv()
    result = momentum_backtest(df,
                               buy_threshold=0.1, sell_threshold=-0.1)
    assert result["num_trades"] >= 1
    assert 0.0 <= result["win_rate"] <= 1.0
    assert isinstance(result["total_return"], float)


def test_nan_score_rows_skipped():
    """Checks that NaN momentum scores are skipped without error (covers line 65)."""
    df = make_ohlcv(60)
    # Use a very large lookback so most rows have NaN score
    sigs = momentum_signals(df, ticker="SKIP", lookback=50,
                            buy_threshold=0.3, sell_threshold=-0.3)
    assert isinstance(sigs, list)  # should not raise even if empty
