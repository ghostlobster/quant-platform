"""
adapters/sentiment/cache.py — SQLite-backed TTL cache for sentiment scores.

Uses the shared quant.db via data/db.py:get_connection().
Cache table: sentiment_cache (symbol, provider, score, fetched_at)
Default TTL: 1800 seconds (30 minutes).
"""
from __future__ import annotations

import time

from data.db import get_connection

_DEFAULT_TTL = 1800  # 30 minutes


def cache_read(symbol: str, provider: str, ttl: int = _DEFAULT_TTL) -> float | None:
    """
    Return cached sentiment score for (symbol, provider) if fresh, else None.

    Parameters
    ----------
    symbol   : ticker symbol, e.g. 'AAPL'
    provider : adapter name, e.g. 'vader' or 'stocktwits'
    ttl      : cache lifetime in seconds (default 1800 = 30 min)

    Returns
    -------
    float score in [-1.0, 1.0] if cache hit and not stale, else None.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT score, fetched_at FROM sentiment_cache WHERE symbol=? AND provider=?",
            (symbol.upper(), provider),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    age = time.time() - float(row["fetched_at"])
    if age > ttl:
        return None
    return float(row["score"])


def cache_write(symbol: str, provider: str, score: float) -> None:
    """
    Upsert a sentiment score into the cache with the current timestamp.

    Parameters
    ----------
    symbol   : ticker symbol, e.g. 'AAPL'
    provider : adapter name, e.g. 'vader' or 'stocktwits'
    score    : sentiment score in [-1.0, 1.0]
    """
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO sentiment_cache (symbol, provider, score, fetched_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (symbol, provider) DO UPDATE
                    SET score=excluded.score, fetched_at=excluded.fetched_at
                """,
                (symbol.upper(), provider, float(score), time.time()),
            )
    finally:
        conn.close()
