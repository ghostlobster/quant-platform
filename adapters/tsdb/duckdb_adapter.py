from __future__ import annotations

import os


class DuckDBAdapter:
    def __init__(self) -> None:
        try:
            import duckdb

            db_path = os.environ.get("DUCKDB_PATH", "quant_platform.duckdb")
            self._conn = duckdb.connect(db_path)
        except ImportError as e:
            raise ImportError("duckdb not installed. Run: pip install duckdb") from e

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        rows = [tuple(r[c] for c in cols) for r in records]
        self._conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})", rows
        )

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        result = self._conn.execute(sql, list(params))
        cols = [d[0] for d in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def create_table(self, table: str, schema: str) -> None:
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")

    def close(self) -> None:
        self._conn.close()
