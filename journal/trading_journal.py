"""
journal/trading_journal.py — SQLite-backed trading journal.

Records trade entries and exits with signal/regime metadata for
performance analysis.

DB path is controlled by the JOURNAL_DB_PATH env var
(default: journal_trades.db in the working directory).

Public API
----------
  init_journal_table()        → None   (idempotent bootstrap)
  log_entry(...)              → int    (new trade_id)
  log_exit(trade_id, ...)     → None
  get_journal(...)            → pd.DataFrame
  win_rate_by_signal_source() → pd.DataFrame
  avg_pnl_by_regime()         → pd.DataFrame
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker        TEXT    NOT NULL,
    side          TEXT    NOT NULL,
    qty           INTEGER NOT NULL,
    entry_price   REAL    NOT NULL,
    entry_time    TEXT    NOT NULL,
    signal_source TEXT,
    regime        TEXT,
    entry_notes   TEXT,
    exit_price    REAL,
    exit_time     TEXT,
    pnl           REAL,
    exit_reason   TEXT,
    exit_notes    TEXT
)
"""


# ── Connection helpers ────────────────────────────────────────────────────────

def _get_db_path() -> str:
    return os.getenv("JOURNAL_DB_PATH", "journal_trades.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_journal_table() -> None:
    """Create the trades table if it doesn't exist. Safe to call repeatedly."""
    conn = _get_connection()
    with conn:
        conn.execute(_CREATE_SQL)
    conn.close()


def _ensure_table() -> None:
    init_journal_table()


# ── Public API ────────────────────────────────────────────────────────────────

def log_entry(
    ticker: str,
    side: str,
    qty: int,
    price: float,
    signal_source: str,
    regime: str = "",
    notes: str = "",
) -> int:
    """Insert a trade entry record; returns the trade_id."""
    _ensure_table()
    now = datetime.now(timezone.utc).isoformat()
    ticker = ticker.upper().strip()
    side   = side.upper().strip()

    conn = _get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO trades
                    (ticker, side, qty, entry_price, entry_time,
                     signal_source, regime, entry_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker, side, int(qty), float(price), now,
                    signal_source or None,
                    regime or None,
                    notes or None,
                ),
            )
            trade_id = cur.lastrowid
        logger.info(
            "Journal entry: %s %s x%d @ $%.4f (id=%d)",
            side, ticker, int(qty), float(price), trade_id,
        )
        return int(trade_id)
    finally:
        conn.close()


def log_exit(
    trade_id: int,
    price: float,
    pnl: float,
    exit_reason: str,
    notes: str = "",
) -> None:
    """Update the entry record with exit details."""
    _ensure_table()
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    try:
        with conn:
            conn.execute(
                """
                UPDATE trades
                SET exit_price=?, exit_time=?, pnl=?, exit_reason=?, exit_notes=?
                WHERE id=?
                """,
                (
                    float(price), now, float(pnl),
                    exit_reason or None,
                    notes or None,
                    int(trade_id),
                ),
            )
        logger.info(
            "Journal exit: trade_id=%d exit_price=%.4f pnl=%.4f",
            trade_id, price, pnl,
        )
    finally:
        conn.close()


def get_journal(
    start_date: str = None,
    end_date: str = None,
    ticker: str = None,
) -> pd.DataFrame:
    """
    Return journal rows as a DataFrame; all filters optional.

    Parameters
    ----------
    start_date : ISO date string 'YYYY-MM-DD' (inclusive lower bound on entry_time)
    end_date   : ISO date string 'YYYY-MM-DD' (inclusive upper bound on entry_time)
    ticker     : Filter to a single ticker symbol (case-insensitive)
    """
    _ensure_table()

    conditions: list[str] = []
    params: list = []

    if start_date:
        conditions.append("entry_time >= ?")
        params.append(start_date)
    if end_date:
        # Extend end_date to include the full day
        end_val = end_date if len(end_date) > 10 else end_date + "T23:59:59"
        conditions.append("entry_time <= ?")
        params.append(end_val)
    if ticker:
        conditions.append("ticker = ?")
        params.append(ticker.upper().strip())

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    sql = f"SELECT * FROM trades {where} ORDER BY entry_time DESC"

    conn = _get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    _COLS = [
        "id", "ticker", "side", "qty", "entry_price", "entry_time",
        "signal_source", "regime", "entry_notes",
        "exit_price", "exit_time", "pnl", "exit_reason", "exit_notes",
    ]
    if not rows:
        return pd.DataFrame(columns=_COLS)

    return pd.DataFrame([dict(row) for row in rows])


def win_rate_by_signal_source() -> pd.DataFrame:
    """
    Return aggregated win-rate stats grouped by signal_source.

    Columns: signal_source | total_trades | wins | win_rate | avg_pnl

    Only closed trades (pnl IS NOT NULL) are counted.
    """
    _ensure_table()
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(signal_source, '') AS signal_source,
                COUNT(*)                    AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                AVG(pnl)                    AS avg_pnl
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY signal_source
            ORDER BY total_trades DESC
            """
        ).fetchall()
    finally:
        conn.close()

    _COLS = ["signal_source", "total_trades", "wins", "win_rate", "avg_pnl"]
    if not rows:
        return pd.DataFrame(columns=_COLS)

    records = []
    for row in rows:
        total = int(row["total_trades"])
        wins  = int(row["wins"] or 0)
        records.append({
            "signal_source": row["signal_source"],
            "total_trades":  total,
            "wins":          wins,
            "win_rate":      round(wins / total, 4) if total > 0 else 0.0,
            "avg_pnl":       round(float(row["avg_pnl"]), 4) if row["avg_pnl"] is not None else 0.0,
        })
    return pd.DataFrame(records)


def avg_pnl_by_regime() -> pd.DataFrame:
    """
    Return aggregated PnL stats grouped by regime.

    Columns: regime | total_trades | avg_pnl | win_rate

    Only closed trades (pnl IS NOT NULL) are counted.
    """
    _ensure_table()
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(regime, '') AS regime,
                COUNT(*)             AS total_trades,
                AVG(pnl)             AS avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM trades
            WHERE pnl IS NOT NULL
            GROUP BY regime
            ORDER BY avg_pnl DESC
            """
        ).fetchall()
    finally:
        conn.close()

    _COLS = ["regime", "total_trades", "avg_pnl", "win_rate"]
    if not rows:
        return pd.DataFrame(columns=_COLS)

    records = []
    for row in rows:
        total = int(row["total_trades"])
        wins  = int(row["wins"] or 0)
        records.append({
            "regime":        row["regime"],
            "total_trades":  total,
            "avg_pnl":       round(float(row["avg_pnl"]), 4) if row["avg_pnl"] is not None else 0.0,
            "win_rate":      round(wins / total, 4) if total > 0 else 0.0,
        })
    return pd.DataFrame(records)
