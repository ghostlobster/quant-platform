"""
adapters/tsdb/duckdb_adapter.py — TSDBProvider backed by DuckDB.

Requires:  pip install duckdb
ENV vars:  DUCKDB_PATH  (default: quant_tsdb.duckdb)
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import duckdb as _duckdb
except ImportError:
    _duckdb = None  # type: ignore[assignment]

_lock = threading.Lock()
_connection: Optional[object] = None


def _get_conn(path: str) -> object:
    global _connection
    if _connection is not None:
        return _connection
    with _lock:
        if _connection is None:
            if _duckdb is None:
                raise ImportError(
                    "duckdb package is required for DuckDBAdapter. "
                    "Install it with: pip install duckdb"
                )
            _connection = _duckdb.connect(path)
    return _connection


class DuckDBAdapter:
    """TSDBProvider using DuckDB — columnar, fast for analytic queries."""

    def __init__(self, path: str | None = None) -> None:
        if _duckdb is None:
            raise ImportError(
                "duckdb package is required for DuckDBAdapter. "
                "Install it with: pip install duckdb"
            )
        self._path = path or os.environ.get("DUCKDB_PATH", "quant_tsdb.duckdb")
        self._conn = _get_conn(self._path)

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        import pandas as pd
        df = pd.DataFrame(records)  # noqa: F841 — referenced by DuckDB's SELECT * FROM df
        with _lock:
            self._conn.execute(f"INSERT INTO {table} SELECT * FROM df")
        logger.debug("DuckDB write: %d rows → %s", len(records), table)

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with _lock:
            result = self._conn.execute(sql, list(params))
            cols = [d[0] for d in result.description]
            rows = result.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def create_table(self, table: str, schema: str) -> None:
        ddl = f"CREATE TABLE IF NOT EXISTS {table} ({schema})"
        with _lock:
            self._conn.execute(ddl)
        logger.debug("DuckDB create_table: %s", table)

    def close(self) -> None:
        global _connection
        with _lock:
            if _connection is not None:
                _connection.close()
                _connection = None
