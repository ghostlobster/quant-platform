"""
Tests for scheduler/alerts.py

Uses an in-memory SQLite database and mocks plyer / channels so no
side-effects occur.
"""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scheduler.alerts import (
    ALERT_TYPES,
    init_alerts_table,
    add_alert,
    get_alerts,
    delete_alert,
    toggle_alert,
    check_alerts,
    _notify,
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


def _mem_conn() -> _NoClose:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker       TEXT    NOT NULL,
            alert_type   TEXT    NOT NULL,
            threshold    REAL    NOT NULL,
            enabled      INTEGER NOT NULL DEFAULT 1,
            created_at   REAL    NOT NULL,
            triggered_at REAL
        )
    """)
    conn.commit()
    return _NoClose(conn)


# ── ALERT_TYPES ───────────────────────────────────────────────────────────────

def test_alert_types_has_four_entries():
    assert set(ALERT_TYPES.keys()) == {"price_above", "price_below", "rsi_above", "rsi_below"}


# ── init_alerts_table ─────────────────────────────────────────────────────────

def test_init_alerts_table_creates_table():
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.row_factory = sqlite3.Row
    conn = _NoClose(raw)
    with patch("scheduler.alerts.get_connection", return_value=conn):
        init_alerts_table()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert any(t["name"] == "alerts" for t in tables)


# ── add_alert ─────────────────────────────────────────────────────────────────

class TestAddAlert:
    def test_returns_integer_id(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            alert_id = add_alert("AAPL", "price_above", 200.0)
        assert isinstance(alert_id, int)
        assert alert_id > 0

    def test_alert_stored_in_db(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            add_alert("MSFT", "price_below", 300.0)
        row = conn.execute("SELECT * FROM alerts WHERE ticker='MSFT'").fetchone()
        assert row is not None
        assert row["alert_type"] == "price_below"
        assert row["threshold"] == pytest.approx(300.0)
        assert row["enabled"] == 1

    def test_ticker_uppercased(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            add_alert("aapl", "rsi_above", 70.0)
        row = conn.execute("SELECT ticker FROM alerts").fetchone()
        assert row["ticker"] == "AAPL"

    def test_invalid_alert_type_raises(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             pytest.raises(ValueError, match="Unknown alert_type"):
            add_alert("AAPL", "invalid_type", 100.0)

    def test_all_valid_alert_types_accepted(self):
        conn = _mem_conn()
        for alert_type in ALERT_TYPES:
            with patch("scheduler.alerts.get_connection", return_value=conn):
                add_alert("AAPL", alert_type, 50.0)
        count = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        assert count == len(ALERT_TYPES)


# ── get_alerts ────────────────────────────────────────────────────────────────

class TestGetAlerts:
    def test_returns_empty_df_when_no_alerts(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            df = get_alerts()
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "ID" in df.columns

    def test_returns_df_with_alerts(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            add_alert("AAPL", "price_above", 200.0)
        with patch("scheduler.alerts.get_connection", return_value=conn):
            df = get_alerts()
        assert len(df) == 1
        assert df.iloc[0]["Ticker"] == "AAPL"
        assert df.iloc[0]["Threshold"] == pytest.approx(200.0)

    def test_df_columns_correct(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            df = get_alerts()
        expected = {"ID", "Ticker", "Type", "Threshold", "Enabled", "Created", "Last Triggered"}
        assert expected.issubset(set(df.columns))

    def test_enabled_field_is_bool(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            add_alert("AAPL", "rsi_below", 30.0)
        with patch("scheduler.alerts.get_connection", return_value=conn):
            df = get_alerts()
        assert isinstance(df.iloc[0]["Enabled"], bool)

    def test_last_triggered_is_dash_when_never_fired(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            add_alert("AAPL", "price_above", 200.0)
        with patch("scheduler.alerts.get_connection", return_value=conn):
            df = get_alerts()
        assert df.iloc[0]["Last Triggered"] == "—"


# ── delete_alert ──────────────────────────────────────────────────────────────

class TestDeleteAlert:
    def test_removes_alert_from_db(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            alert_id = add_alert("AAPL", "price_above", 200.0)
        with patch("scheduler.alerts.get_connection", return_value=conn):
            delete_alert(alert_id)
        count = conn.execute("SELECT COUNT(*) FROM alerts WHERE id=?", (alert_id,)).fetchone()[0]
        assert count == 0

    def test_delete_nonexistent_is_noop(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            delete_alert(9999)  # Should not raise


# ── toggle_alert ──────────────────────────────────────────────────────────────

class TestToggleAlert:
    def test_disable_alert(self):
        conn = _mem_conn()
        with patch("scheduler.alerts.get_connection", return_value=conn):
            alert_id = add_alert("AAPL", "price_above", 200.0)
        with patch("scheduler.alerts.get_connection", return_value=conn):
            toggle_alert(alert_id, enabled=False)
        row = conn.execute("SELECT enabled FROM alerts WHERE id=?", (alert_id,)).fetchone()
        assert row["enabled"] == 0

    def test_enable_alert(self):
        conn = _mem_conn()
        conn.execute(
            "INSERT INTO alerts (ticker, alert_type, threshold, enabled, created_at) VALUES (?,?,?,?,?)",
            ("AAPL", "price_above", 200.0, 0, time.time()),
        )
        conn.commit()
        alert_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        with patch("scheduler.alerts.get_connection", return_value=conn):
            toggle_alert(alert_id, enabled=True)
        row = conn.execute("SELECT enabled FROM alerts WHERE id=?", (alert_id,)).fetchone()
        assert row["enabled"] == 1


# ── check_alerts ──────────────────────────────────────────────────────────────

class TestCheckAlerts:
    def _insert_alert(self, conn, ticker, alert_type, threshold, enabled=1):
        conn.execute(
            "INSERT INTO alerts (ticker, alert_type, threshold, enabled, created_at) VALUES (?,?,?,?,?)",
            (ticker, alert_type, threshold, enabled, time.time()),
        )
        conn.commit()

    def test_price_above_triggers(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 150.0)
        current_data = {"AAPL": {"price": 160.0, "rsi": 55.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert len(triggered) == 1
        assert triggered[0]["ticker"] == "AAPL"
        assert triggered[0]["alert_type"] == "price_above"

    def test_price_above_does_not_trigger_below(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 200.0)
        current_data = {"AAPL": {"price": 150.0, "rsi": 55.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert triggered == []

    def test_price_below_triggers(self):
        conn = _mem_conn()
        self._insert_alert(conn, "MSFT", "price_below", 300.0)
        current_data = {"MSFT": {"price": 280.0, "rsi": 40.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert len(triggered) == 1
        assert triggered[0]["alert_type"] == "price_below"

    def test_rsi_above_triggers(self):
        conn = _mem_conn()
        self._insert_alert(conn, "NVDA", "rsi_above", 70.0)
        current_data = {"NVDA": {"price": 500.0, "rsi": 75.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert len(triggered) == 1
        assert triggered[0]["alert_type"] == "rsi_above"

    def test_rsi_below_triggers(self):
        conn = _mem_conn()
        self._insert_alert(conn, "NVDA", "rsi_below", 30.0)
        current_data = {"NVDA": {"price": 500.0, "rsi": 25.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert len(triggered) == 1

    def test_disabled_alert_not_triggered(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0, enabled=0)
        current_data = {"AAPL": {"price": 200.0, "rsi": 55.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert triggered == []

    def test_missing_ticker_in_data_skipped(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0)
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts({})  # AAPL not in data
        assert triggered == []

    def test_none_price_for_price_alert_skipped(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0)
        current_data = {"AAPL": {"price": None, "rsi": None}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert triggered == []

    def test_triggered_at_stamped(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0)
        current_data = {"AAPL": {"price": 200.0, "rsi": 55.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            check_alerts(current_data)
        row = conn.execute("SELECT triggered_at FROM alerts").fetchone()
        assert row["triggered_at"] is not None

    def test_result_has_message_key(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0)
        current_data = {"AAPL": {"price": 200.0, "rsi": 55.0}}
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert "message" in triggered[0]
        assert "fired_at" in triggered[0]

    def test_multiple_alerts_multiple_tickers(self):
        conn = _mem_conn()
        self._insert_alert(conn, "AAPL", "price_above", 100.0)
        self._insert_alert(conn, "MSFT", "price_below", 400.0)
        current_data = {
            "AAPL": {"price": 200.0, "rsi": 50.0},
            "MSFT": {"price": 350.0, "rsi": 45.0},
        }
        with patch("scheduler.alerts.get_connection", return_value=conn), \
             patch("scheduler.alerts.broadcast"), \
             patch("scheduler.alerts._notify"):
            triggered = check_alerts(current_data)
        assert len(triggered) == 2


# ── _notify ───────────────────────────────────────────────────────────────────

def test_notify_silently_handles_plyer_failure():
    """_notify should not raise even if plyer is unavailable."""
    with patch("scheduler.alerts.logger"):
        _notify("Test Title", "Test message")  # should not raise
