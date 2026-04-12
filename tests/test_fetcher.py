"""
Tests for data/fetcher.py

Mocks yfinance and the SQLite DB so no I/O occurs.
"""
from __future__ import annotations

import io
import sqlite3
import time
from unittest.mock import MagicMock, patch, call
import numpy as np
import pandas as pd
import pytest

# ── In-memory DB fixture ──────────────────────────────────────────────────────

class _NoClose:
    """Wraps sqlite3.Connection making close() a no-op so in-memory data persists."""
    def __init__(self, conn: sqlite3.Connection):
        object.__setattr__(self, '_c', conn)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, '_c'), name)

    def __enter__(self):
        return object.__getattribute__(self, '_c').__enter__()

    def __exit__(self, *args):
        return object.__getattribute__(self, '_c').__exit__(*args)

    def close(self):
        pass  # no-op

    def execute(self, *a, **kw):
        return object.__getattribute__(self, '_c').execute(*a, **kw)

    def executemany(self, *a, **kw):
        return object.__getattribute__(self, '_c').executemany(*a, **kw)


def _make_memory_conn() -> _NoClose:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_cache (
            ticker      TEXT    NOT NULL,
            period      TEXT    NOT NULL,
            fetched_at  REAL    NOT NULL,
            data_json   TEXT    NOT NULL,
            PRIMARY KEY (ticker, period)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            ticker   TEXT PRIMARY KEY,
            added_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS paper_account (
            id INTEGER PRIMARY KEY CHECK (id=1),
            cash_balance REAL NOT NULL,
            realised_pnl REAL NOT NULL DEFAULT 0
        )
    """)
    conn.execute("INSERT OR IGNORE INTO paper_account VALUES (1, 100000, 0)")
    conn.commit()
    return _NoClose(conn)


def _ohlcv_df(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": np.ones(n) * 1e6},
        index=idx,
    )


# Import the module under test after setting up helpers
from data.fetcher import _ttl_for, _flatten_columns, _cache_read, _cache_write, fetch_ohlcv, fetch_latest_price


# ── _ttl_for ──────────────────────────────────────────────────────────────────

class TestTtlFor:
    def test_intraday_periods(self):
        for p in ("1d", "2d", "5d"):
            assert _ttl_for(p) == 3_600

    def test_short_periods(self):
        for p in ("1mo", "3mo"):
            assert _ttl_for(p) == 14_400

    def test_historical_periods(self):
        for p in ("6mo", "1y", "2y", "5y"):
            assert _ttl_for(p) == 86_400


# ── _flatten_columns ──────────────────────────────────────────────────────────

class TestFlattenColumns:
    def test_drops_nan_close(self):
        df = _ohlcv_df(5)
        # introduce a NaN close row
        df.loc[df.index[2], "Close"] = float("nan")
        from data.fetcher import _flatten_columns
        result = _flatten_columns(df)
        assert result["Close"].isna().sum() == 0

    def test_handles_multiindex_columns(self):
        df = _ohlcv_df(5)
        # Simulate yfinance multi-index: (Price, Ticker)
        df.columns = pd.MultiIndex.from_tuples(
            [(col, "AAPL") for col in df.columns]
        )
        result = _flatten_columns(df)
        assert isinstance(result.columns, pd.Index)
        assert "Close" in result.columns

    def test_passthrough_for_normal_columns(self):
        df = _ohlcv_df(5)
        result = _flatten_columns(df)
        assert "Close" in result.columns


# ── _cache_read / _cache_write ────────────────────────────────────────────────

class TestCache:
    def test_cache_miss_returns_none(self):
        conn = _make_memory_conn()
        with patch("data.fetcher.get_connection", return_value=conn):
            result = _cache_read("AAPL", "6mo")
        assert result is None

    def test_write_then_read_returns_df(self):
        conn = _make_memory_conn()
        df = _ohlcv_df(20)
        with patch("data.fetcher.get_connection", return_value=conn):
            _cache_write("AAPL", "6mo", df)
        with patch("data.fetcher.get_connection", return_value=conn):
            result = _cache_read("AAPL", "6mo")
        assert result is not None
        assert len(result) == len(df)

    def test_expired_cache_returns_none(self):
        conn = _make_memory_conn()
        df = _ohlcv_df(10)
        old_time = time.time() - 200_000  # way past any TTL
        data_json = df.to_json(orient="split", date_format="iso")
        conn.execute(
            "INSERT INTO price_cache (ticker, period, fetched_at, data_json) VALUES (?,?,?,?)",
            ("AAPL", "6mo", old_time, data_json),
        )
        conn.commit()
        with patch("data.fetcher.get_connection", return_value=conn):
            result = _cache_read("AAPL", "6mo")
        assert result is None

    def test_cache_write_upserts(self):
        conn = _make_memory_conn()
        df1 = _ohlcv_df(5)
        df2 = _ohlcv_df(8)
        with patch("data.fetcher.get_connection", return_value=conn):
            _cache_write("AAPL", "6mo", df1)
            _cache_write("AAPL", "6mo", df2)  # upsert
        with patch("data.fetcher.get_connection", return_value=conn):
            result = _cache_read("AAPL", "6mo")
        assert len(result) == 8


# ── fetch_ohlcv ───────────────────────────────────────────────────────────────

class TestFetchOhlcv:
    def test_returns_cached_df_on_hit(self):
        conn = _make_memory_conn()
        df = _ohlcv_df(10)
        data_json = df.to_json(orient="split", date_format="iso")
        conn.execute(
            "INSERT INTO price_cache (ticker, period, fetched_at, data_json) VALUES (?,?,?,?)",
            ("AAPL", "6mo", time.time(), data_json),
        )
        conn.commit()
        with patch("data.fetcher.get_connection", return_value=conn), \
             patch("data.fetcher.init_db"), \
             patch("data.fetcher.yf.download") as mock_dl:
            result = fetch_ohlcv("AAPL", "6mo")
        mock_dl.assert_not_called()
        assert len(result) == 10

    def test_downloads_when_cache_miss(self):
        conn = _make_memory_conn()
        raw = _ohlcv_df(20)
        with patch("data.fetcher.get_connection", return_value=conn), \
             patch("data.fetcher.init_db"), \
             patch("data.fetcher.yf.download", return_value=raw) as mock_dl:
            result = fetch_ohlcv("TSLA", "6mo")
        mock_dl.assert_called_once()
        assert len(result) == 20

    def test_raises_on_empty_yfinance_response(self):
        conn = _make_memory_conn()
        with patch("data.fetcher.get_connection", return_value=conn), \
             patch("data.fetcher.init_db"), \
             patch("data.fetcher.yf.download", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No data returned"):
                fetch_ohlcv("INVALID", "6mo")

    def test_ticker_uppercased(self):
        conn = _make_memory_conn()
        raw = _ohlcv_df(10)
        with patch("data.fetcher.get_connection", return_value=conn), \
             patch("data.fetcher.init_db"), \
             patch("data.fetcher.yf.download", return_value=raw):
            result = fetch_ohlcv("aapl", "6mo")
        assert result is not None


# ── fetch_latest_price ────────────────────────────────────────────────────────

class TestFetchLatestPrice:
    def test_returns_dict_with_price_keys(self):
        df = _ohlcv_df(5)
        with patch("data.fetcher.fetch_ohlcv", return_value=df):
            result = fetch_latest_price("AAPL")
        assert "price" in result
        assert "prev_close" in result
        assert "change" in result
        assert "pct_change" in result
        assert "error" in result
        assert result["error"] is None

    def test_returns_error_on_insufficient_data(self):
        df = _ohlcv_df(1)
        with patch("data.fetcher.fetch_ohlcv", return_value=df):
            result = fetch_latest_price("AAPL")
        assert result["error"] == "insufficient data"
        assert result["price"] is None

    def test_returns_error_on_exception(self):
        with patch("data.fetcher.fetch_ohlcv", side_effect=ValueError("no data")):
            result = fetch_latest_price("AAPL")
        assert result["error"] is not None
        assert result["price"] is None

    def test_price_computation(self):
        df = _ohlcv_df(5)
        with patch("data.fetcher.fetch_ohlcv", return_value=df):
            result = fetch_latest_price("AAPL")
        expected_price = round(float(df["Close"].iloc[-1]), 2)
        expected_prev  = round(float(df["Close"].iloc[-2]), 2)
        assert result["price"] == pytest.approx(expected_price)
        assert result["prev_close"] == pytest.approx(expected_prev)
        assert result["change"] == pytest.approx(round(expected_price - expected_prev, 2))

    def test_ticker_key_uppercased(self):
        df = _ohlcv_df(5)
        with patch("data.fetcher.fetch_ohlcv", return_value=df):
            result = fetch_latest_price("aapl")
        assert result["ticker"] == "AAPL"
