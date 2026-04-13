from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class TSDBProvider(Protocol):
    def write(self, table: str, records: list[dict]) -> None: ...
    def query(self, sql: str, params: tuple = ()) -> list[dict]: ...
    def create_table(self, table: str, schema: str) -> None: ...
    def close(self) -> None: ...


def get_tsdb(provider: Optional[str] = None) -> TSDBProvider:
    name = (provider or os.environ.get("TSDB_PROVIDER", "sqlite")).lower()
    if name == "sqlite":
        from adapters.tsdb.sqlite_adapter import SQLiteAdapter
        return SQLiteAdapter()
    elif name == "duckdb":
        from adapters.tsdb.duckdb_adapter import DuckDBAdapter
        return DuckDBAdapter()
    elif name == "timescale":
        from adapters.tsdb.timescale_adapter import TimescaleAdapter
        return TimescaleAdapter()
    raise ValueError(f"Unknown TSDB provider: {name!r}. Valid: sqlite, duckdb, timescale")
