from __future__ import annotations

import os
import sqlite3


class SQLiteAdapter:
    def __init__(self) -> None:
        db_path = os.environ.get("SQLITE_DB_PATH", "quant_platform.db")
        self._conn = sqlite3.connect(db_path, check_same_thread=False)

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        placeholders = ",".join("?" * len(cols))
        col_str = ",".join(cols)
        with self._conn:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
                [tuple(r[c] for c in cols) for r in records],
            )

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def create_table(self, table: str, schema: str) -> None:
        with self._conn:
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")

    def close(self) -> None:
        self._conn.close()
