"""
Tests for data/watchlist.py

Uses an in-memory SQLite database so no file I/O occurs.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import patch

from data.watchlist import (
    _DEFAULT_TICKERS,
    _ensure_defaults,
    add_ticker,
    get_watchlist,
    is_in_watchlist,
    remove_ticker,
)

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


def _raw_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE watchlist (
            ticker   TEXT PRIMARY KEY,
            added_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _mem_conn() -> _NoClose:
    return _NoClose(_raw_conn())


# ── _ensure_defaults ──────────────────────────────────────────────────────────

class TestEnsureDefaults:
    def test_seeds_empty_watchlist(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn):
            _ensure_defaults()
        tickers = [r[0] for r in conn.execute("SELECT ticker FROM watchlist").fetchall()]
        assert set(_DEFAULT_TICKERS).issubset(set(tickers))

    def test_does_not_duplicate_if_already_seeded(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn):
            _ensure_defaults()
            _ensure_defaults()  # second call should be a no-op
        count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        assert count == len(_DEFAULT_TICKERS)


# ── get_watchlist ─────────────────────────────────────────────────────────────

class TestGetWatchlist:
    def test_returns_list(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = get_watchlist()
        assert isinstance(result, list)

    def test_returns_default_tickers_on_empty_db(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = get_watchlist()
        assert set(_DEFAULT_TICKERS).issubset(set(result))

    def test_returns_added_ticker(self):
        conn = _mem_conn()
        import time
        conn.execute("INSERT INTO watchlist VALUES (?, ?)", ("NVDA", time.time()))
        conn.commit()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"), \
             patch("data.watchlist._ensure_defaults"):  # skip seeding
            result = get_watchlist()
        assert "NVDA" in result


# ── add_ticker ────────────────────────────────────────────────────────────────

class TestAddTicker:
    def test_add_new_ticker_returns_true(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = add_ticker("NVDA")
        assert result is True
        tickers = [r[0] for r in conn.execute("SELECT ticker FROM watchlist").fetchall()]
        assert "NVDA" in tickers

    def test_add_duplicate_returns_false(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            add_ticker("NVDA")
            result = add_ticker("NVDA")
        assert result is False

    def test_ticker_uppercased(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            add_ticker("nvda")
        tickers = [r[0] for r in conn.execute("SELECT ticker FROM watchlist").fetchall()]
        assert "NVDA" in tickers


# ── remove_ticker ─────────────────────────────────────────────────────────────

class TestRemoveTicker:
    def test_remove_existing_returns_true(self):
        conn = _mem_conn()
        import time
        conn.execute("INSERT INTO watchlist VALUES (?, ?)", ("NVDA", time.time()))
        conn.commit()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = remove_ticker("NVDA")
        assert result is True
        count = conn.execute("SELECT COUNT(*) FROM watchlist WHERE ticker='NVDA'").fetchone()[0]
        assert count == 0

    def test_remove_nonexistent_returns_false(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = remove_ticker("ZZZZZ")
        assert result is False

    def test_remove_uppercases_ticker(self):
        conn = _mem_conn()
        import time
        conn.execute("INSERT INTO watchlist VALUES (?, ?)", ("NVDA", time.time()))
        conn.commit()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            result = remove_ticker("nvda")
        assert result is True


# ── is_in_watchlist ───────────────────────────────────────────────────────────

class TestIsInWatchlist:
    def test_returns_true_for_present_ticker(self):
        conn = _mem_conn()
        import time
        conn.execute("INSERT INTO watchlist VALUES (?, ?)", ("AAPL", time.time()))
        conn.commit()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            assert is_in_watchlist("AAPL") is True

    def test_returns_false_for_absent_ticker(self):
        conn = _mem_conn()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            assert is_in_watchlist("ZZZZ") is False

    def test_case_insensitive_lookup(self):
        conn = _mem_conn()
        import time
        conn.execute("INSERT INTO watchlist VALUES (?, ?)", ("AAPL", time.time()))
        conn.commit()
        with patch("data.watchlist.get_connection", return_value=conn), \
             patch("data.watchlist.init_db"):
            assert is_in_watchlist("aapl") is True
