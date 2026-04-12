"""Tests for data/db.py — SQLite schema helpers."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import pytest
from unittest.mock import patch
from pathlib import Path


def test_get_connection_returns_connection(tmp_path):
    db_file = tmp_path / "test.db"
    with patch("data.db._DB_PATH", db_file):
        from data.db import get_connection
        conn = get_connection()
        assert isinstance(conn, sqlite3.Connection)
        conn.close()


def test_get_connection_wal_mode(tmp_path):
    db_file = tmp_path / "test.db"
    with patch("data.db._DB_PATH", db_file):
        from data.db import get_connection
        conn = get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()


def test_init_db_creates_tables(tmp_path):
    db_file = tmp_path / "test.db"
    with patch("data.db._DB_PATH", db_file):
        from data.db import init_db, get_connection
        init_db()
        conn = get_connection()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
    assert "price_cache" in tables
    assert "watchlist" in tables
    assert "paper_account" in tables
    assert "paper_positions" in tables
    assert "paper_trades" in tables
    assert "portfolio_history" in tables


def test_init_db_seeds_paper_account(tmp_path):
    db_file = tmp_path / "test.db"
    with patch("data.db._DB_PATH", db_file), \
         patch.dict(os.environ, {"PAPER_STARTING_CASH": "50000"}):
        from data.db import init_db, get_connection
        init_db()
        conn = get_connection()
        row = conn.execute("SELECT cash_balance FROM paper_account WHERE id=1").fetchone()
        conn.close()
    assert row is not None
    assert row[0] == 50000.0


def test_init_db_idempotent(tmp_path):
    """Calling init_db twice must not raise or reset the account balance."""
    db_file = tmp_path / "test.db"
    with patch("data.db._DB_PATH", db_file), \
         patch.dict(os.environ, {"PAPER_STARTING_CASH": "75000"}):
        from data.db import init_db, get_connection
        init_db()
        init_db()  # second call must be a no-op
        conn = get_connection()
        row = conn.execute("SELECT cash_balance FROM paper_account WHERE id=1").fetchone()
        conn.close()
    assert row[0] == 75000.0
