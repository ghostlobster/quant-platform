"""
providers/tsdb.py — TSDBProvider protocol and factory.

ENV vars
--------
    TSDB_PROVIDER   sqlite | duckdb | timescale  (default: sqlite)
    TIMESCALE_DSN   postgresql://user:pass@host:5432/db
    DUCKDB_PATH     path to .duckdb file (default: quant_tsdb.duckdb)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TSDBProvider(Protocol):
    """Duck-typed interface for time-series database operations."""

    def write(self, table: str, records: list[dict]) -> None:
        """Insert *records* into *table*."""
        ...

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        """Execute *sql* with *params* and return rows as dicts."""
        ...

    def create_table(self, table: str, schema: str) -> None:
        """
        Create *table* if it does not exist.

        Parameters
        ----------
        table  : table name
        schema : SQL column definitions, e.g.
                 ``"ts TIMESTAMP, symbol TEXT, close REAL"``
        """
        ...

    def close(self) -> None:
        """Release underlying connection/resources."""
        ...


def get_tsdb(provider: Optional[str] = None) -> TSDBProvider:
    """
    Return a configured TSDBProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the TSDB_PROVIDER env var.  One of:
        ``sqlite``, ``duckdb``, ``timescale``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (provider or os.environ.get("TSDB_PROVIDER", "sqlite")).lower().strip()
    if name == "sqlite":
        from adapters.tsdb.sqlite_adapter import SQLiteTSDBAdapter
        return SQLiteTSDBAdapter()
    if name == "duckdb":
        from adapters.tsdb.duckdb_adapter import DuckDBAdapter
        return DuckDBAdapter()
    if name == "timescale":
        from adapters.tsdb.timescale_adapter import TimescaleAdapter
        return TimescaleAdapter()
    raise ValueError(
        f"Unknown TSDB provider: {name!r}. "
        "Valid options: sqlite, duckdb, timescale"
    )
