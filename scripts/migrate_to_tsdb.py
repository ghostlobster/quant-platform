"""
scripts/migrate_to_tsdb.py — Migrate existing SQLite data from quant.db into the
configured TSDB backend (DuckDB or TimescaleDB).

Usage
-----
    python scripts/migrate_to_tsdb.py [--provider duckdb|timescale|sqlite]

ENV vars honoured
-----------------
    TSDB_PROVIDER       sqlite | duckdb | timescale  (default: sqlite)
    SQLITE_TSDB_PATH    destination path for SQLite TSDB (default: quant_tsdb.db)
    DUCKDB_PATH         destination path for DuckDB (default: quant_tsdb.duckdb)
    TIMESCALE_DSN       postgresql:// connection string

The migration is IDEMPOTENT — running it twice does not duplicate data.
Each table uses INSERT OR REPLACE / ON CONFLICT DO NOTHING semantics so
existing rows are skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import structlog

# Ensure the project root is on sys.path when run directly.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

log = structlog.get_logger(__name__)


# ── Schema definitions for TSDB destination tables ────────────────────────────

_OHLCV_SCHEMA = (
    "symbol TEXT NOT NULL, "
    "ts TEXT NOT NULL, "
    "open REAL, high REAL, low REAL, close REAL, volume REAL, "
    "PRIMARY KEY (symbol, ts)"
)

_TRADES_SCHEMA = (
    "id INTEGER NOT NULL, "
    "executed_at REAL NOT NULL, "
    "symbol TEXT NOT NULL, "
    "action TEXT NOT NULL, "
    "shares REAL NOT NULL, "
    "price REAL NOT NULL, "
    "cost_basis REAL, "
    "realised_pnl REAL, "
    "PRIMARY KEY (id)"
)

_PORTFOLIO_HISTORY_SCHEMA = (
    "id INTEGER NOT NULL, "
    "record_date TEXT NOT NULL UNIQUE, "
    "total_value REAL NOT NULL, "
    "PRIMARY KEY (id)"
)


# ── Migration helpers ──────────────────────────────────────────────────────────

def migrate_price_cache(src_conn, tsdb) -> int:
    """
    Read price_cache JSON blobs from quant.db and write OHLCV rows to TSDB.

    Returns the number of rows written.
    """
    tsdb.create_table("ohlcv_prices", _OHLCV_SCHEMA)

    rows = src_conn.execute(
        "SELECT ticker, data_json FROM price_cache"
    ).fetchall()

    written = 0
    for row in rows:
        ticker = row[0] if isinstance(row, (list, tuple)) else row["ticker"]
        data_json = row[1] if isinstance(row, (list, tuple)) else row["data_json"]
        try:
            data = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            log.warning("Skipping malformed price_cache row", ticker=ticker)
            continue

        records: list[dict] = []
        # data is a dict-of-dicts: {column -> {ts -> value}}
        if isinstance(data, dict):
            columns = list(data.keys())
            # Gather all timestamps
            timestamps: set[str] = set()
            for col_data in data.values():
                if isinstance(col_data, dict):
                    timestamps.update(col_data.keys())
            for ts in sorted(timestamps):
                record: dict = {"symbol": ticker, "ts": ts}
                for col in columns:
                    col_lower = col.lower()
                    if col_lower in ("open", "high", "low", "close", "volume"):
                        col_data = data.get(col, {})
                        record[col_lower] = col_data.get(ts) if isinstance(col_data, dict) else None
                if "close" in record:
                    records.append(record)

        if records:
            tsdb.write("ohlcv_prices", records)
            written += len(records)
            log.info("Migrated price cache", ticker=ticker, rows=len(records))

    return written


def migrate_paper_trades(src_conn, tsdb) -> int:
    """
    Read paper_trades from quant.db and write to TSDB execution_history table.

    Returns the number of rows written.
    """
    tsdb.create_table("execution_history", _TRADES_SCHEMA)

    rows = src_conn.execute(
        "SELECT id, executed_at, ticker, action, shares, price, cost_basis, realised_pnl "
        "FROM paper_trades ORDER BY id"
    ).fetchall()

    if not rows:
        return 0

    records = []
    for row in rows:
        if hasattr(row, "keys"):
            r = dict(row)
        else:
            r = {
                "id": row[0], "executed_at": row[1], "symbol": row[2],
                "action": row[3], "shares": row[4], "price": row[5],
                "cost_basis": row[6], "realised_pnl": row[7],
            }
        # Normalise ticker → symbol
        r.setdefault("symbol", r.pop("ticker", ""))
        records.append(r)

    tsdb.write("execution_history", records)
    log.info("Migrated paper trades", count=len(records))
    return len(records)


def migrate_portfolio_history(src_conn, tsdb) -> int:
    """
    Read portfolio_history from quant.db and write to TSDB.

    Returns the number of rows written.
    """
    tsdb.create_table("portfolio_snapshots", _PORTFOLIO_HISTORY_SCHEMA)

    rows = src_conn.execute(
        "SELECT id, record_date, total_value FROM portfolio_history ORDER BY id"
    ).fetchall()

    if not rows:
        return 0

    records = []
    for row in rows:
        if hasattr(row, "keys"):
            records.append(dict(row))
        else:
            records.append({"id": row[0], "record_date": row[1], "total_value": row[2]})

    tsdb.write("portfolio_snapshots", records)
    log.info("Migrated portfolio history", count=len(records))
    return len(records)


# ── Entry point ────────────────────────────────────────────────────────────────

def main(provider: str | None = None) -> None:
    """Run all migrations from quant.db → configured TSDB."""
    from data.db import get_connection, _DB_PATH
    from providers.tsdb import get_tsdb

    log.info("Starting TSDB migration", source=str(_DB_PATH), provider=provider or "env")

    tsdb = get_tsdb(provider)
    src_conn = get_connection()

    try:
        start = time.time()
        ohlcv_rows = migrate_price_cache(src_conn, tsdb)
        trade_rows = migrate_paper_trades(src_conn, tsdb)
        portfolio_rows = migrate_portfolio_history(src_conn, tsdb)
        elapsed = round(time.time() - start, 2)

        log.info(
            "Migration complete",
            ohlcv_rows=ohlcv_rows,
            trade_rows=trade_rows,
            portfolio_rows=portfolio_rows,
            elapsed_seconds=elapsed,
        )
    finally:
        src_conn.close()
        tsdb.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Migrate quant.db data to the configured TSDB backend."
    )
    parser.add_argument(
        "--provider",
        choices=["sqlite", "duckdb", "timescale"],
        default=None,
        help="TSDB provider to migrate to (overrides TSDB_PROVIDER env var)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(provider=args.provider)
