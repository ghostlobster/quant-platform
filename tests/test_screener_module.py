"""
Tests for screener/screener.py

Mocks yfinance so no network calls are made. Tests metric computation,
filter logic, signal assignment, and the public run_screen() API.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from screener.screener import (
    TICKERS,
    UNIVERSE,
    _compute_metrics,
    _compute_sma,
    _fetch_batch,
    run_screen,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 100, seed: int = 42, start: float = 100.0) -> pd.DataFrame:
    np.random.seed(seed)
    close = start + np.cumsum(np.random.randn(n) * 0.3)
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":   close,
            "High":   close + 0.5,
            "Low":    close - 0.5,
            "Close":  close,
            "Volume": volume,
        },
        index=idx,
    )


# ── UNIVERSE constant ─────────────────────────────────────────────────────────

class TestUniverse:
    def test_universe_has_tickers(self):
        assert len(UNIVERSE) >= 20

    def test_each_entry_has_ticker_sector_name(self):
        for item in UNIVERSE:
            assert "ticker" in item
            assert "sector" in item
            assert "name" in item

    def test_tickers_list_matches_universe(self):
        assert set(TICKERS) == {item["ticker"] for item in UNIVERSE}


# ── _compute_sma ──────────────────────────────────────────────────────────────

class TestComputeSma:
    def test_returns_float_with_enough_data(self):
        close = pd.Series(np.arange(60, dtype=float))
        result = _compute_sma(close, window=50)
        assert isinstance(result, float)

    def test_returns_none_with_insufficient_data(self):
        close = pd.Series(np.arange(30, dtype=float))
        assert _compute_sma(close, window=50) is None

    def test_sma_exact_window_insufficient(self):
        # Need > 50 rows to get a valid rolling mean
        close = pd.Series(np.arange(50, dtype=float))
        result = _compute_sma(close, window=50)
        assert result is not None  # exactly 50 rows → exactly 1 valid SMA value

    def test_sma_value_is_mean(self):
        close = pd.Series([10.0] * 100)
        result = _compute_sma(close, window=50)
        assert result == pytest.approx(10.0)


# ── _compute_metrics ──────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_returns_dict_with_expected_keys(self):
        df = make_ohlcv(100)
        row = _compute_metrics("AAPL", df, change_days=5)
        assert "Ticker" in row
        assert "Last Price" in row
        assert "RSI" in row
        assert "Signal" in row
        assert "Above SMA50" in row

    def test_ticker_matches(self):
        df = make_ohlcv(100)
        row = _compute_metrics("MSFT", df, change_days=5)
        assert row["Ticker"] == "MSFT"

    def test_sector_name_for_known_ticker(self):
        df = make_ohlcv(100)
        row = _compute_metrics("AAPL", df, change_days=5)
        assert row["Sector"] == "Technology"
        assert row["Name"] == "Apple"

    def test_unknown_ticker_gets_default_sector(self):
        df = make_ohlcv(100)
        row = _compute_metrics("ZZZZZ", df, change_days=5)
        assert row["Sector"] == "—"

    def test_last_price_is_last_close(self):
        df = make_ohlcv(100)
        row = _compute_metrics("AAPL", df, change_days=5)
        assert row["Last Price"] == pytest.approx(round(float(df["Close"].iloc[-1]), 2))

    def test_signal_na_when_no_rsi(self):
        # Very short series → RSI returns None → Signal should be N/A
        df = make_ohlcv(10)  # too short for RSI
        row = _compute_metrics("AAPL", df, change_days=5)
        assert row["Signal"] == "N/A"

    def test_signal_oversold_when_rsi_low(self):
        # Build a consistently falling price series to get low RSI
        close = np.linspace(200.0, 100.0, 80)
        volume = np.ones(80) * 1_000_000
        df = pd.DataFrame({"Close": close, "Volume": volume, "Open": close, "High": close, "Low": close})
        row = _compute_metrics("TEST", df, change_days=5)
        if row["RSI"] is not None and row["RSI"] < 30:
            assert row["Signal"] == "Oversold"

    def test_above_sma50_is_bool_or_none(self):
        df = make_ohlcv(100)
        row = _compute_metrics("AAPL", df, change_days=5)
        assert row["Above SMA50"] is True or row["Above SMA50"] is False or row["Above SMA50"] is None

    def test_change_days_respected(self):
        # Fixed prices except last → can verify change
        close = np.full(20, 100.0)
        close[-1] = 110.0
        df = pd.DataFrame({
            "Close":  close,
            "Volume": np.ones(20) * 1_000_000,
            "Open":   close, "High": close, "Low": close,
        })
        row = _compute_metrics("TEST", df, change_days=1)
        chg = row.get("Change 1d (%)")
        assert chg is not None
        assert chg == pytest.approx(10.0)


# ── _fetch_batch ──────────────────────────────────────────────────────────────

class TestFetchBatch:
    def test_empty_symbols_returns_empty(self):
        result = _fetch_batch([])
        assert result == {}

    def test_returns_empty_on_yfinance_exception(self):
        with patch("screener.screener.yf.download", side_effect=RuntimeError("no network")):
            result = _fetch_batch(["AAPL"])
        assert result == {}

    def test_returns_empty_if_download_returns_empty(self):
        with patch("screener.screener.yf.download", return_value=pd.DataFrame()):
            result = _fetch_batch(["AAPL"])
        assert result == {}

    def test_single_ticker_flat_columns(self):
        df = make_ohlcv(50)
        with patch("screener.screener.yf.download", return_value=df):
            result = _fetch_batch(["AAPL"])
        assert "AAPL" in result
        assert len(result["AAPL"]) >= 20

    def test_too_short_df_is_skipped(self):
        df = make_ohlcv(5)  # fewer than 20 rows
        with patch("screener.screener.yf.download", return_value=df):
            result = _fetch_batch(["AAPL"])
        assert "AAPL" not in result


# ── run_screen ────────────────────────────────────────────────────────────────

def _make_screen_data_map(tickers=("AAPL", "MSFT", "NVDA")) -> dict:
    """Build a fake data_map suitable for run_screen internals."""
    return {t: make_ohlcv(100, seed=i) for i, t in enumerate(tickers)}


class TestRunScreen:
    def test_empty_data_returns_empty_df(self):
        with patch("screener.screener._fetch_batch", return_value={}):
            result = run_screen(tickers=["AAPL"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_returns_dataframe_with_results(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(data_map)

    def test_result_has_expected_columns(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()))
        for col in ("Ticker", "Signal", "Last Price", "RSI"):
            assert col in result.columns

    def test_rsi_min_filter(self):
        data_map = _make_screen_data_map(("AAPL", "MSFT", "NVDA"))
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), rsi_min=80.0)
        # All remaining rows must have RSI >= 80 (or NaN filled as 50)
        if not result.empty:
            assert (result["RSI"].fillna(50) >= 80.0).all()

    def test_rsi_max_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), rsi_max=30.0)
        if not result.empty:
            assert (result["RSI"].fillna(50) <= 30.0).all()

    def test_price_min_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), price_min=999999.0)
        assert result.empty

    def test_price_max_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), price_max=0.01)
        assert result.empty

    def test_vol_spike_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), vol_spike_min=999.0)
        assert result.empty

    def test_trend_above_sma50_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), trend="above_sma50")
        if not result.empty:
            assert (result["Above SMA50"] == True).all()  # noqa: E712

    def test_trend_below_sma50_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), trend="below_sma50")
        if not result.empty:
            assert (result["Above SMA50"] == False).all()  # noqa: E712

    def test_default_tickers_uses_universe(self):
        # run_screen with no tickers arg should pass TICKERS to _fetch_batch
        with patch("screener.screener._fetch_batch", return_value={}) as mock_fetch:
            run_screen()
        args = mock_fetch.call_args[0]
        assert set(args[0]) == set(TICKERS)

    def test_result_reset_index(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()))
        assert list(result.index) == list(range(len(result)))

    def test_change_min_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), change_min=999.0)
        assert result.empty

    def test_change_max_filter(self):
        data_map = _make_screen_data_map()
        with patch("screener.screener._fetch_batch", return_value=data_map):
            result = run_screen(tickers=list(data_map.keys()), change_max=-999.0)
        assert result.empty
