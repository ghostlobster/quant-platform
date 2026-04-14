"""
adapters/tsdb/sqlite_adapter.py — TSDBProvider backed by SQLite (dev default).

Thread-safe singleton connection.  Database file defaults to quant_tsdb.db in
the project root (separate from quant.db used by the core app).

ENV vars
--------
    SQLITE_TSDB_PATH   path to .db file (default: quant_tsdb.db)
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_connection: Optional[sqlite3.Connection] = None


def _get_conn(path: str) -> sqlite3.Connection:
    global _connection
    if _connection is not None:
        return _connection
    with _lock:
        if _connection is None:
            conn = sqlite3.connect(path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            _connection = conn
    return _connection


class SQLiteTSDBAdapter:
    """TSDBProvider using SQLite — zero-dependency, great for local dev."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or os.environ.get("SQLITE_TSDB_PATH", "quant_tsdb.db")
        self._conn = _get_conn(self._path)

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        placeholders = ", ".join("?" * len(cols))
        col_str = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})"
        with _lock:
            with self._conn:
                self._conn.executemany(sql, [tuple(r[c] for c in cols) for r in records])
        logger.debug("TSDB write: %d rows → %s", len(records), table)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with _lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(r) for r in rows]

    def create_table(self, table: str, schema: str) -> None:
        ddl = f"CREATE TABLE IF NOT EXISTS {table} ({schema})"
        with _lock:
            with self._conn:
                self._conn.execute(ddl)
        logger.debug("TSDB create_table: %s", table)

    def close(self) -> None:
        global _connection
        with _lock:
            if _connection is not None:
                _connection.close()
                _connection = None
