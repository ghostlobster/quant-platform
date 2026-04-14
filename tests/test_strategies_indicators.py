"""
Tests for strategies/indicators.py

Covers: add_rsi, add_macd, add_bollinger_bands, add_ema, add_volume_sma,
        add_all, _latest, generate_signals.
"""
import numpy as np
import pandas as pd
import pytest

from strategies.indicators import (
    _latest,
    add_all,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_rsi,
    add_volume_sma,
    generate_signals,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 100, seed: int = 42, start: float = 100.0) -> pd.DataFrame:
    np.random.seed(seed)
    close = start + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )


# ── add_rsi ───────────────────────────────────────────────────────────────────

class TestAddRsi:
    def test_adds_rsi_column(self):
        df = make_ohlcv()
        result = add_rsi(df)
        assert "RSI_14" in result.columns

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        cols = list(df.columns)
        add_rsi(df)
        assert list(df.columns) == cols

    def test_custom_window(self):
        df = make_ohlcv()
        result = add_rsi(df, window=10)
        assert "RSI_10" in result.columns
        assert "RSI_14" not in result.columns

    def test_rsi_range_valid(self):
        df = make_ohlcv(100)
        result = add_rsi(df)
        valid = result["RSI_14"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all() and (valid <= 100).all()


# ── add_macd ──────────────────────────────────────────────────────────────────

class TestAddMacd:
    def test_adds_macd_columns(self):
        df = make_ohlcv()
        result = add_macd(df)
        for col in ("MACD_line", "MACD_signal", "MACD_hist"):
            assert col in result.columns

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        cols = list(df.columns)
        add_macd(df)
        assert list(df.columns) == cols

    def test_hist_equals_line_minus_signal(self):
        df = make_ohlcv(100)
        result = add_macd(df)
        valid = result.dropna(subset=["MACD_line", "MACD_signal", "MACD_hist"])
        diff = (valid["MACD_line"] - valid["MACD_signal"] - valid["MACD_hist"]).abs()
        assert (diff < 1e-8).all()


# ── add_bollinger_bands ───────────────────────────────────────────────────────

class TestAddBollingerBands:
    def test_adds_bb_columns(self):
        df = make_ohlcv()
        result = add_bollinger_bands(df)
        for col in ("BB_upper", "BB_mid", "BB_lower", "BB_pct_b"):
            assert col in result.columns

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        cols = list(df.columns)
        add_bollinger_bands(df)
        assert list(df.columns) == cols

    def test_upper_above_lower(self):
        df = make_ohlcv(100)
        result = add_bollinger_bands(df)
        valid = result.dropna(subset=["BB_upper", "BB_lower"])
        assert (valid["BB_upper"] >= valid["BB_lower"]).all()


# ── add_ema ───────────────────────────────────────────────────────────────────

class TestAddEma:
    def test_adds_default_ema_columns(self):
        df = make_ohlcv()
        result = add_ema(df)
        assert "EMA_20" in result.columns
        assert "EMA_50" in result.columns

    def test_custom_windows(self):
        df = make_ohlcv()
        result = add_ema(df, windows=[10, 30])
        assert "EMA_10" in result.columns
        assert "EMA_30" in result.columns
        assert "EMA_20" not in result.columns

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        cols = list(df.columns)
        add_ema(df)
        assert list(df.columns) == cols


# ── add_volume_sma ────────────────────────────────────────────────────────────

class TestAddVolumeSma:
    def test_adds_vol_sma_column(self):
        df = make_ohlcv()
        result = add_volume_sma(df)
        assert "Vol_SMA_20" in result.columns

    def test_does_not_mutate_input(self):
        df = make_ohlcv()
        cols = list(df.columns)
        add_volume_sma(df)
        assert list(df.columns) == cols


# ── add_all ───────────────────────────────────────────────────────────────────

class TestAddAll:
    def test_adds_all_indicator_columns(self):
        df = make_ohlcv(100)
        result = add_all(df)
        expected = [
            "RSI_14", "MACD_line", "MACD_signal", "MACD_hist",
            "BB_upper", "BB_mid", "BB_lower", "BB_pct_b",
            "EMA_20", "EMA_50", "Vol_SMA_20",
        ]
        for col in expected:
            assert col in result.columns

    def test_preserves_original_columns(self):
        df = make_ohlcv(100)
        result = add_all(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in result.columns


# ── _latest ───────────────────────────────────────────────────────────────────

class TestLatest:
    def test_returns_last_non_nan(self):
        df = pd.DataFrame({"A": [1.0, 2.0, float("nan"), 3.0]})
        assert _latest(df, "A") == pytest.approx(3.0)

    def test_returns_none_for_all_nan(self):
        df = pd.DataFrame({"A": [float("nan"), float("nan")]})
        assert _latest(df, "A") is None

    def test_returns_float(self):
        df = pd.DataFrame({"A": [42.0]})
        val = _latest(df, "A")
        assert isinstance(val, float)


# ── generate_signals ──────────────────────────────────────────────────────────

def _make_signal_df(
    rsi=50.0,
    macd_prev_hist=-0.05,
    macd_curr_hist=0.05,
    macd_line=0.1,
    macd_signal=0.05,
    bb_pct_b=0.5,
    bb_upper=110.0,
    bb_lower=90.0,
    close=100.0,
    ema20=102.0,
    ema50=100.0,
) -> pd.DataFrame:
    """Build a minimal 2-row indicator DataFrame."""
    return pd.DataFrame(
        {
            "Close":       [close, close],
            "RSI_14":      [float("nan"), rsi],
            "MACD_line":   [float("nan"), macd_line],
            "MACD_signal": [float("nan"), macd_signal],
            "MACD_hist":   [macd_prev_hist, macd_curr_hist],
            "BB_pct_b":    [float("nan"), bb_pct_b],
            "BB_upper":    [float("nan"), bb_upper],
            "BB_lower":    [float("nan"), bb_lower],
            "EMA_20":      [float("nan"), ema20],
            "EMA_50":      [float("nan"), ema50],
        }
    )


def _sig(signals, indicator_substr: str) -> dict:
    return next(s for s in signals if indicator_substr in s["indicator"])


class TestGenerateSignals:
    # ── RSI ──────────────────────────────────────────────────────────────────
    def test_rsi_oversold(self):
        df = _make_signal_df(rsi=25.0)
        s = _sig(generate_signals(df), "RSI")
        assert s["signal"] == "Oversold"
        assert s["bullish"] is True

    def test_rsi_overbought(self):
        df = _make_signal_df(rsi=75.0)
        s = _sig(generate_signals(df), "RSI")
        assert s["signal"] == "Overbought"
        assert s["bullish"] is False

    def test_rsi_neutral(self):
        df = _make_signal_df(rsi=50.0)
        s = _sig(generate_signals(df), "RSI")
        assert s["signal"] == "Neutral"
        assert s["bullish"] is None

    # ── MACD ─────────────────────────────────────────────────────────────────
    def test_macd_bullish_crossover(self):
        # prev_hist < 0, curr_hist >= 0
        df = _make_signal_df(macd_prev_hist=-0.1, macd_curr_hist=0.1)
        s = _sig(generate_signals(df), "MACD")
        assert s["signal"] == "Bullish crossover"
        assert s["bullish"] is True

    def test_macd_bearish_crossover(self):
        # prev_hist > 0, curr_hist <= 0
        df = _make_signal_df(macd_prev_hist=0.1, macd_curr_hist=-0.1)
        s = _sig(generate_signals(df), "MACD")
        assert s["signal"] == "Bearish crossover"
        assert s["bullish"] is False

    def test_macd_bullish_momentum(self):
        df = _make_signal_df(macd_prev_hist=0.05, macd_curr_hist=0.1)
        s = _sig(generate_signals(df), "MACD")
        assert s["signal"] == "Bullish momentum"

    def test_macd_bearish_momentum(self):
        df = _make_signal_df(macd_prev_hist=-0.05, macd_curr_hist=-0.1)
        s = _sig(generate_signals(df), "MACD")
        assert s["signal"] == "Bearish momentum"

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    def test_bb_above_upper(self):
        df = _make_signal_df(bb_pct_b=1.1)
        s = _sig(generate_signals(df), "Bollinger")
        assert s["signal"] == "Above upper band"
        assert s["bullish"] is False

    def test_bb_below_lower(self):
        df = _make_signal_df(bb_pct_b=-0.1)
        s = _sig(generate_signals(df), "Bollinger")
        assert s["signal"] == "Below lower band"
        assert s["bullish"] is True

    def test_bb_near_upper(self):
        df = _make_signal_df(bb_pct_b=0.85)
        s = _sig(generate_signals(df), "Bollinger")
        assert s["signal"] == "Near upper band"

    def test_bb_near_lower(self):
        df = _make_signal_df(bb_pct_b=0.15)
        s = _sig(generate_signals(df), "Bollinger")
        assert s["signal"] == "Near lower band"

    def test_bb_mid_band(self):
        df = _make_signal_df(bb_pct_b=0.5)
        s = _sig(generate_signals(df), "Bollinger")
        assert s["signal"] == "Mid-band"
        assert s["bullish"] is None

    # ── EMA trend ─────────────────────────────────────────────────────────────
    def test_ema_uptrend(self):
        df = _make_signal_df(ema20=105.0, ema50=100.0)
        s = _sig(generate_signals(df), "EMA")
        assert s["signal"] == "Uptrend"
        assert s["bullish"] is True

    def test_ema_downtrend(self):
        df = _make_signal_df(ema20=95.0, ema50=100.0)
        s = _sig(generate_signals(df), "EMA")
        assert s["signal"] == "Downtrend"
        assert s["bullish"] is False

    def test_no_indicator_columns_returns_empty(self):
        df = pd.DataFrame({"Close": [100.0, 100.0]})
        assert generate_signals(df) == []

    def test_all_four_signal_types_present(self):
        df = _make_signal_df(rsi=50.0, ema20=102.0, ema50=100.0)
        signals = generate_signals(df)
        indicators = [s["indicator"] for s in signals]
        assert any("RSI" in i for i in indicators)
        assert any("MACD" in i for i in indicators)
        assert any("Bollinger" in i for i in indicators)
        assert any("EMA" in i for i in indicators)
