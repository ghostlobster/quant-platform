"""
tests/test_e2e_polygon_backfill_to_fetch_ohlcv.py — backfill → cache → read.

End-to-end check that ``cron.polygon_backfill`` populates both the SQLite
JSON cache and the DuckDB columnar cache, and that a subsequent
``data.fetcher.fetch_ohlcv`` for the same ``(ticker, period)`` hits the
cache without ever calling yfinance.

Network is mocked at the boundary: we patch ``adapters.market_data.
polygon_adapter._requests`` to return a deterministic Polygon payload
and assert ``yfinance.download`` is never invoked.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import adapters.market_data.polygon_adapter as polymod
import cron.polygon_backfill as backfill
import data.db as db_module


def _polygon_payload(rows: int) -> dict:
    base_t = 1_700_000_000_000  # ms
    one_day_ms = 86_400_000
    return {
        "results": [
            {
                "t": base_t + i * one_day_ms,
                "o": 100.0 + i,
                "h": 101.0 + i,
                "l":  99.0 + i,
                "c": 100.5 + i,
                "v": 1_000 * (i + 1),
            }
            for i in range(rows)
        ],
    }


@pytest.fixture
def fake_polygon(monkeypatch):
    """Stub Polygon HTTP — the adapter receives a pre-baked payload."""
    payload = _polygon_payload(30)

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(
        polymod,
        "_requests",
        SimpleNamespace(get=lambda *args, **kwargs: _Resp()),
    )
    return payload


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "quant_tsdb.duckdb"))
    # Reset the DuckDB connection singleton between tests.
    import adapters.tsdb.duckdb_adapter as dad

    dad._connection = None
    yield tmp_path
    dad._connection = None


def test_backfill_populates_sqlite_then_fetch_ohlcv_hits_cache(
    fake_polygon, isolated_caches, monkeypatch,
):
    """SQLite path: backfill writes price_cache; fetch_ohlcv reads it
    without touching yfinance."""
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    monkeypatch.setenv("TSDB_PROVIDER", "sqlite")  # default — only SQLite path

    rc = backfill.main([
        "--tickers", "AAPL", "--days", "30",
        "--timeframe", "1Day", "--period", "1mo",
    ])
    assert rc == 0

    # If fetch_ohlcv falls through to yfinance, the test fails loudly.
    import yfinance

    monkeypatch.setattr(
        yfinance,
        "download",
        lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("yfinance.download must not be called when cache is warm"),
        ),
    )
    from data.fetcher import fetch_ohlcv

    df = fetch_ohlcv("AAPL", "1mo")
    assert len(df) == 30
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    # Polygon stub returns close = 100.5 + i.
    assert df["Close"].iloc[0] == pytest.approx(100.5)
    assert df["Close"].iloc[-1] == pytest.approx(100.5 + 29)


def test_backfill_populates_duckdb_when_active(
    fake_polygon, isolated_caches, monkeypatch,
):
    """DuckDB path: backfill writes to BOTH caches when TSDB_PROVIDER=duckdb."""
    duckdb = pytest.importorskip("duckdb")  # noqa: F841

    monkeypatch.setenv("POLYGON_API_KEY", "key")
    monkeypatch.setenv("TSDB_PROVIDER", "duckdb")

    rc = backfill.main([
        "--tickers", "AAPL", "--days", "30",
        "--timeframe", "1Day", "--period", "1mo",
    ])
    assert rc == 0

    from data import duckdb_cache as dc

    cached = dc.read("AAPL", "1mo", ttl_seconds=3600)
    assert cached is not None
    assert len(cached) == 30
    assert list(cached.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_backfill_intraday_does_not_warm_sqlite_cache(
    fake_polygon, isolated_caches, monkeypatch,
):
    """Intraday timeframes are fetched but skip the daily-only cache schema."""
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    monkeypatch.setenv("TSDB_PROVIDER", "sqlite")

    rc = backfill.main([
        "--tickers", "AAPL", "--days", "5",
        "--timeframe", "5Min", "--period", "5d",
    ])
    assert rc == 0

    from data.fetcher import _cache_read

    # Intraday → no SQLite cache row written.
    assert _cache_read("AAPL", "5d") is None
