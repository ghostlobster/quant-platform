"""
data/watchlist.py — SQLite-backed watchlist for tracking tickers.

Provides simple CRUD helpers. The watchlist table is shared with the rest of
the app via data/db.py — no direct SQL outside this module.
"""
import time
from typing import List

import structlog

from data.db import get_connection, init_db

logger = structlog.get_logger(__name__)

# Default tickers pre-loaded on first run so the watchlist isn't empty
_DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY", "QQQ"]


def _ensure_defaults() -> None:
    """
    If the watchlist is empty, seed it with a handful of common tickers.
    Only runs once (on first startup).
    """
    conn = get_connection()
    try:
        count = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        if count == 0:
            with conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO watchlist (ticker, added_at) VALUES (?, ?)",
                    [(t, time.time()) for t in _DEFAULT_TICKERS],
                )
            logger.info("Watchlist seeded with default tickers: %s", _DEFAULT_TICKERS)
    finally:
        conn.close()


def get_watchlist() -> List[str]:
    """Return all tickers in the watchlist, ordered by the time they were added."""
    init_db()
    _ensure_defaults()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ticker FROM watchlist ORDER BY added_at ASC"
        ).fetchall()
    finally:
        conn.close()
    return [row["ticker"] for row in rows]


def add_ticker(ticker: str) -> bool:
    """
    Add a ticker to the watchlist.

    Returns True if added, False if it was already present.
    """
    init_db()
    ticker = ticker.upper().strip()
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker=?", (ticker,)
        ).fetchone()
        if existing:
            return False
        with conn:
            conn.execute(
                "INSERT INTO watchlist (ticker, added_at) VALUES (?, ?)",
                (ticker, time.time()),
            )
    finally:
        conn.close()
    logger.info("Added %s to watchlist", ticker)
    return True


def remove_ticker(ticker: str) -> bool:
    """
    Remove a ticker from the watchlist.

    Returns True if removed, False if it wasn't in the list.
    """
    init_db()
    ticker = ticker.upper().strip()
    conn = get_connection()
    try:
        existing = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker=?", (ticker,)
        ).fetchone()
        if not existing:
            return False
        with conn:
            conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker,))
    finally:
        conn.close()
    logger.info("Removed %s from watchlist", ticker)
    return True


def is_in_watchlist(ticker: str) -> bool:
    """Return True if the ticker is currently in the watchlist."""
    init_db()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM watchlist WHERE ticker=?", (ticker.upper().strip(),)
        ).fetchone()
    finally:
        conn.close()
    return row is not None
