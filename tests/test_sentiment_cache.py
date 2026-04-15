"""Tests for the 30-min sentiment SQLite TTL cache (Issue #23)."""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import patch

import pytest


class _NoCloseConn:
    """Wraps a sqlite3 connection so close() is a no-op (keeps in-memory DB alive)."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *args):
        return self._conn.__exit__(*args)

    def close(self):
        pass  # no-op — don't close the in-memory DB

    def execute(self, *a, **kw):
        return self._conn.execute(*a, **kw)

    def executemany(self, *a, **kw):
        return self._conn.executemany(*a, **kw)

    def commit(self):
        return self._conn.commit()

    @property
    def row_factory(self):
        return self._conn.row_factory

    @row_factory.setter
    def row_factory(self, v):
        self._conn.row_factory = v


@pytest.fixture
def mock_conn(tmp_path):
    """Return a fresh in-memory SQLite connection with the cache table."""
    raw = sqlite3.connect(":memory:")
    raw.row_factory = sqlite3.Row
    raw.execute("PRAGMA journal_mode=WAL")
    raw.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            symbol      TEXT NOT NULL,
            provider    TEXT NOT NULL,
            score       REAL NOT NULL,
            fetched_at  REAL NOT NULL,
            PRIMARY KEY (symbol, provider)
        )
    """)
    raw.commit()
    return _NoCloseConn(raw)


def _patch_conn(mock_conn):
    """Context manager: patch get_connection() to return mock_conn."""
    return patch("adapters.sentiment.cache.get_connection", return_value=mock_conn)


def test_cache_miss_returns_none(mock_conn):
    with _patch_conn(mock_conn):
        from adapters.sentiment.cache import cache_read
        result = cache_read("AAPL", "vader")
    assert result is None


def test_cache_write_and_hit(mock_conn):
    with _patch_conn(mock_conn):
        from adapters.sentiment.cache import cache_read, cache_write
        cache_write("AAPL", "vader", 0.35)
        result = cache_read("AAPL", "vader")
    assert result == pytest.approx(0.35)


def test_cache_expired_returns_none(mock_conn):
    # Write a record with a very old timestamp
    mock_conn.execute(
        "INSERT INTO sentiment_cache (symbol, provider, score, fetched_at) VALUES (?,?,?,?)",
        ("MSFT", "stocktwits", -0.1, time.time() - 3600),
    )
    mock_conn.commit()
    with _patch_conn(mock_conn):
        import adapters.sentiment.cache as cache_mod
        result = cache_mod.cache_read("MSFT", "stocktwits", ttl=1800)
    assert result is None


def test_cache_upsert_updates_score(mock_conn):
    with _patch_conn(mock_conn):
        from adapters.sentiment.cache import cache_read, cache_write
        cache_write("TSLA", "vader", 0.1)
        cache_write("TSLA", "vader", 0.9)
        result = cache_read("TSLA", "vader")
    assert result == pytest.approx(0.9)


def test_symbol_normalised_to_uppercase(mock_conn):
    with _patch_conn(mock_conn):
        from adapters.sentiment.cache import cache_read, cache_write
        cache_write("aapl", "vader", 0.5)
        result = cache_read("AAPL", "vader")
    assert result == pytest.approx(0.5)
