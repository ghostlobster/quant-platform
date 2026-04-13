"""TimescaleDB adapter via psycopg2."""
from __future__ import annotations

import os


class TimescaleAdapter:
    def __init__(self) -> None:
        try:
            import psycopg2
            import psycopg2.extras

            dsn = os.environ.get("TIMESCALE_DSN", "")
            if not dsn:
                raise ValueError("TIMESCALE_DSN environment variable is required")
            self._conn = psycopg2.connect(dsn)
            self._extras = psycopg2.extras
        except ImportError as e:
            raise ImportError("psycopg2 not installed. Run: pip install psycopg2-binary") from e

    def write(self, table: str, records: list[dict]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        col_str = ",".join(cols)
        placeholders = ",".join(f"%({c})s" for c in cols)
        with self._conn.cursor() as cur:
            self._extras.execute_batch(
                cur,
                f"INSERT INTO {table} ({col_str}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                records,
            )
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def create_table(self, table: str, schema: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
