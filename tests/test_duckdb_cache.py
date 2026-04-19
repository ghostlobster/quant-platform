"""
tests/test_duckdb_cache.py — DuckDB OHLCV cache round-trip + fetcher wiring.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.db as db_module

try:
    import duckdb  # noqa: F401
    _DUCKDB = True
except ImportError:
    _DUCKDB = False


pytestmark = pytest.mark.skipif(
    not _DUCKDB, reason="duckdb package not installed in this environment",
)


@pytest.fixture(autouse=True)
def _duckdb_env(tmp_path, monkeypatch):
    """Point DuckDB at a per-test file and reset its singleton connection."""
    monkeypatch.setenv("TSDB_PROVIDER", "duckdb")
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "test.duckdb"))
    import adapters.tsdb.duckdb_adapter as dad

    # Reset the module-level connection singleton so each test uses its own file.
    dad._connection = None
    yield
    dad._connection = None


@pytest.fixture
def temp_sqlite(tmp_path, monkeypatch):
    """Isolate the SQLite JSON cache to a per-test file."""
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    return tmp_path


def _make_df(rows: int = 5) -> pd.DataFrame:
    start = datetime(2025, 1, 1)
    idx = pd.DatetimeIndex([start + timedelta(days=i) for i in range(rows)])
    return pd.DataFrame(
        {
            "Open":   [100.0 + i for i in range(rows)],
            "High":   [101.0 + i for i in range(rows)],
            "Low":    [ 99.0 + i for i in range(rows)],
            "Close":  [100.5 + i for i in range(rows)],
            "Volume": [1_000 * (i + 1) for i in range(rows)],
        },
        index=idx,
    )


# ── Basic round-trip ─────────────────────────────────────────────────────────

def test_is_active_reflects_env(monkeypatch):
    from data import duckdb_cache as dc

    monkeypatch.setenv("TSDB_PROVIDER", "duckdb")
    assert dc.is_active() is True
    monkeypatch.setenv("TSDB_PROVIDER", "sqlite")
    assert dc.is_active() is False


def test_write_then_read_roundtrip():
    from data import duckdb_cache as dc

    df = _make_df(5)
    dc.write("AAPL", "1mo", df)
    got = dc.read("AAPL", "1mo", ttl_seconds=3600)
    assert got is not None
    assert len(got) == 5
    assert list(got.columns) == ["Open", "High", "Low", "Close", "Volume"]
    pd.testing.assert_series_equal(
        got["Close"].reset_index(drop=True),
        df["Close"].reset_index(drop=True),
        check_names=False,
    )


def test_read_returns_none_on_empty_table():
    from data import duckdb_cache as dc

    assert dc.read("AAPL", "1mo", ttl_seconds=3600) is None


def test_write_is_noop_when_inactive(monkeypatch):
    from data import duckdb_cache as dc

    monkeypatch.setenv("TSDB_PROVIDER", "sqlite")
    # Should not raise even though we never init the adapter.
    dc.write("AAPL", "1mo", _make_df(3))
    assert dc.read("AAPL", "1mo", ttl_seconds=3600) is None


def test_read_expires_after_ttl(monkeypatch):
    from data import duckdb_cache as dc

    dc.write("AAPL", "1mo", _make_df(3))
    # Force negative TTL so any fetched_at is "too old".
    got = dc.read("AAPL", "1mo", ttl_seconds=-1)
    assert got is None


def test_upsert_replaces_rows():
    from data import duckdb_cache as dc

    dc.write("AAPL", "1mo", _make_df(3))
    dc.write("AAPL", "1mo", _make_df(5))
    got = dc.read("AAPL", "1mo", ttl_seconds=3600)
    assert got is not None
    assert len(got) == 5


def test_keys_isolated_by_ticker_and_period():
    from data import duckdb_cache as dc

    dc.write("AAPL", "1mo", _make_df(3))
    dc.write("SPY", "1mo", _make_df(4))
    dc.write("AAPL", "1y", _make_df(2))
    assert len(dc.read("AAPL", "1mo", ttl_seconds=3600)) == 3
    assert len(dc.read("SPY",  "1mo", ttl_seconds=3600)) == 4
    assert len(dc.read("AAPL", "1y",  ttl_seconds=3600)) == 2


# ── Fetcher integration ──────────────────────────────────────────────────────

def test_fetcher_warms_duckdb_from_sqlite(temp_sqlite, monkeypatch):
    """When SQLite has a hit and DuckDB is empty, fetcher warms DuckDB."""
    from data import duckdb_cache as dc
    from data.db import init_db
    from data.fetcher import _cache_write, fetch_ohlcv

    init_db()
    df = _make_df(4)
    _cache_write("AAPL", "1mo", df)

    # Guard: no yfinance call expected on this path.
    monkeypatch.setattr(
        "yfinance.download",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not hit network")),
    )
    returned = fetch_ohlcv("AAPL", "1mo")
    assert len(returned) == 4

    # DuckDB must now have the same 4 rows cached.
    cached = dc.read("AAPL", "1mo", ttl_seconds=3600)
    assert cached is not None
    assert len(cached) == 4


def test_fetcher_prefers_duckdb_when_active(temp_sqlite):
    """A pre-populated DuckDB cache short-circuits the SQLite path."""
    from data import duckdb_cache as dc
    from data.fetcher import fetch_ohlcv

    dc.write("AAPL", "1mo", _make_df(6))
    got = fetch_ohlcv("AAPL", "1mo")
    assert len(got) == 6
