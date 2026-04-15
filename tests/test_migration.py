"""Tests for scripts/migrate_to_tsdb.py (Issue #25)."""
from __future__ import annotations

import json
import sqlite3
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def src_conn():
    """In-memory SQLite source DB with the tables that migrate_to_tsdb reads."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE price_cache (
            ticker TEXT NOT NULL,
            data_json TEXT NOT NULL,
            PRIMARY KEY (ticker)
        )
    """)
    conn.execute("""
        CREATE TABLE paper_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            executed_at REAL,
            ticker TEXT,
            action TEXT,
            shares REAL,
            price REAL,
            cost_basis REAL,
            realised_pnl REAL
        )
    """)
    conn.execute("""
        CREATE TABLE portfolio_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_date TEXT,
            total_value REAL
        )
    """)
    conn.commit()
    return conn


@pytest.fixture
def mock_tsdb():
    """Mock TSDB that records writes."""
    tsdb = MagicMock()
    tsdb.written = {}

    def _write(table, records):
        tsdb.written.setdefault(table, []).extend(records)

    tsdb.write.side_effect = _write
    return tsdb


def test_migrate_price_cache_empty(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_price_cache
    written = migrate_price_cache(src_conn, mock_tsdb)
    assert written == 0


def test_migrate_price_cache_with_data(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_price_cache

    # Insert a price_cache row with valid OHLCV data
    data = {
        "Open": {"2024-01-02": 180.0},
        "High": {"2024-01-02": 185.0},
        "Low":  {"2024-01-02": 179.0},
        "Close": {"2024-01-02": 183.0},
        "Volume": {"2024-01-02": 55_000_000.0},
    }
    src_conn.execute(
        "INSERT INTO price_cache (ticker, data_json) VALUES (?, ?)",
        ("AAPL", json.dumps(data)),
    )
    src_conn.commit()

    written = migrate_price_cache(src_conn, mock_tsdb)
    assert written == 1
    mock_tsdb.create_table.assert_called_once()
    rows = mock_tsdb.written.get("ohlcv_prices", [])
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["close"] == 183.0


def test_migrate_price_cache_skips_malformed(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_price_cache

    src_conn.execute(
        "INSERT INTO price_cache (ticker, data_json) VALUES (?, ?)",
        ("BAD", "not-json{{{"),
    )
    src_conn.commit()

    written = migrate_price_cache(src_conn, mock_tsdb)
    assert written == 0


def test_migrate_paper_trades_empty(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_paper_trades
    written = migrate_paper_trades(src_conn, mock_tsdb)
    assert written == 0


def test_migrate_paper_trades_with_data(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_paper_trades

    src_conn.execute(
        "INSERT INTO paper_trades (executed_at, ticker, action, shares, price, cost_basis, realised_pnl) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), "MSFT", "buy", 10.0, 300.0, 3000.0, 0.0),
    )
    src_conn.commit()

    written = migrate_paper_trades(src_conn, mock_tsdb)
    assert written == 1
    rows = mock_tsdb.written.get("execution_history", [])
    assert len(rows) == 1
    assert rows[0]["symbol"] == "MSFT"


def test_migrate_portfolio_history_with_data(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_portfolio_history

    src_conn.execute(
        "INSERT INTO portfolio_history (record_date, total_value) VALUES (?, ?)",
        ("2024-06-01", 105_000.0),
    )
    src_conn.commit()

    written = migrate_portfolio_history(src_conn, mock_tsdb)
    assert written == 1
    rows = mock_tsdb.written.get("portfolio_snapshots", [])
    assert len(rows) == 1
    assert rows[0]["total_value"] == pytest.approx(105_000.0)


def test_migrate_portfolio_history_empty(src_conn, mock_tsdb):
    from scripts.migrate_to_tsdb import migrate_portfolio_history
    written = migrate_portfolio_history(src_conn, mock_tsdb)
    assert written == 0
