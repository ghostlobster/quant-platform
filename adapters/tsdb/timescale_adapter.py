"""
adapters/tsdb/timescale_adapter.py — TSDBProvider backed by TimescaleDB.

Requires:  pip install psycopg2-binary
ENV vars:  TIMESCALE_DSN  e.g. postgresql://user:pass@localhost:5432/quant
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2 as _psycopg2
    import psycopg2.extras as _extras
except ImportError:
    _psycopg2 = None  # type: ignore[assignment]
    _extras = None  # type: ignore[assignment]

_lock = threading.Lock()
_connection: Optional[object] = None


def _get_conn(dsn: str) -> object:
    global _connection
    if _connection is not None:
        return _connection
    with _lock:
        if _connection is None:
            if _psycopg2 is None:
                raise ImportError(
                    "psycopg2-binary is required for TimescaleAdapter. "
                    "Install it with: pip install psycopg2-binary"
                )
            conn = _psycopg2.connect(dsn)
            conn.autocommit = False
            _connection = conn
    return _connection


class TimescaleAdapter:
    """TSDBProvider using TimescaleDB via psycopg2."""

    def __init__(self, dsn: str | None = None) -> None:
        if _psycopg2 is None:
            raise ImportError(
                "psycopg2-binary is required for TimescaleAdapter. "
                "Install it with: pip install psycopg2-binary"
            )
        self._dsn = dsn or os.environ.get("TIMESCALE_DSN", "")
        if not self._dsn:
            raise ValueError("TIMESCALE_DSN must be set to use TimescaleAdapter.")
        self._conn = _get_conn(self._dsn)

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        col_str = ", ".join(cols)
        placeholders = ", ".join(f"%({c})s" for c in cols)
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
        with _lock:
            with self._conn.cursor() as cur:
                _extras.execute_batch(cur, sql, records)
            self._conn.commit()
        logger.debug("Timescale write: %d rows → %s", len(records), table)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with _lock:
            with self._conn.cursor(_extras.RealDictCursor) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [dict(r) for r in rows]

    def create_table(self, table: str, schema: str) -> None:
        ddl = f"CREATE TABLE IF NOT EXISTS {table} ({schema})"
        with _lock:
            with self._conn.cursor() as cur:
                cur.execute(ddl)
            self._conn.commit()
        logger.debug("Timescale create_table: %s", table)

    def close(self) -> None:
        global _connection
        with _lock:
            if _connection is not None:
                _connection.close()
                _connection = None
