"""
data/duckdb_cache.py — DuckDB-backed OHLCV cache for hot-path backtests.

Wraps :class:`adapters.tsdb.duckdb_adapter.DuckDBAdapter` with a
DataFrame-friendly read/write surface keyed by ``(ticker, period)``, the
same key schema :mod:`data.fetcher` uses for its SQLite cache. Bulk
walk-forward reads can run an order of magnitude faster against DuckDB's
columnar storage than the JSON-encoded SQLite table.

Activation — set ``TSDB_PROVIDER=duckdb`` in the environment. When unset
or set to any other value the helpers are no-ops and :func:`is_active`
returns ``False`` so :mod:`data.fetcher` keeps the SQLite path.

Schema
------
::

    CREATE TABLE price_cache_duckdb (
        ticker     TEXT,
        period     TEXT,
        ts         TIMESTAMP,
        open       DOUBLE,
        high       DOUBLE,
        low        DOUBLE,
        close      DOUBLE,
        volume     DOUBLE,
        fetched_at TIMESTAMP,
        PRIMARY KEY (ticker, period, ts)
    )
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_TABLE = "price_cache_duckdb"
_SCHEMA = (
    "ticker TEXT, period TEXT, ts TIMESTAMP, "
    "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, "
    "fetched_at TIMESTAMP, PRIMARY KEY (ticker, period, ts)"
)


def is_active() -> bool:
    """Return ``True`` when ``TSDB_PROVIDER=duckdb``."""
    return os.environ.get("TSDB_PROVIDER", "sqlite").lower().strip() == "duckdb"


def _get_adapter():
    """Lazy import so the sqlite default never pulls in duckdb."""
    from adapters.tsdb.duckdb_adapter import DuckDBAdapter

    return DuckDBAdapter()


def _ensure_table(adapter) -> None:
    adapter.create_table(_TABLE, _SCHEMA)


def read(ticker: str, period: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
    """Return the cached DataFrame if fresh, else ``None``.

    Returns ``None`` silently when DuckDB is not installed, the table is
    empty for this key, or the rows have aged past ``ttl_seconds``.
    """
    if not is_active():
        return None
    try:
        adapter = _get_adapter()
        _ensure_table(adapter)
        rows = adapter.query(
            f"SELECT ts, open, high, low, close, volume, fetched_at "
            f"FROM {_TABLE} WHERE ticker = ? AND period = ? ORDER BY ts ASC",
            (ticker.upper(), period),
        )
    except Exception as exc:
        logger.debug("duckdb_cache read failed", error=str(exc))
        return None

    if not rows:
        return None

    newest_fetch = max(r["fetched_at"] for r in rows if r.get("fetched_at") is not None)
    if newest_fetch is not None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        if isinstance(newest_fetch, datetime):
            age = (now - newest_fetch).total_seconds()
            if age > ttl_seconds:
                logger.debug(
                    "duckdb_cache expired",
                    ticker=ticker, period=period,
                    age=age, ttl=ttl_seconds,
                )
                return None

    df = pd.DataFrame(rows)[["ts", "open", "high", "low", "close", "volume"]]
    df.columns = ["ts", "Open", "High", "Low", "Close", "Volume"]
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    df.index.name = None
    logger.debug("duckdb_cache hit", ticker=ticker, period=period, rows=len(df))
    return df


def write(ticker: str, period: str, df: pd.DataFrame) -> None:
    """Upsert ``df`` into the DuckDB cache for ``(ticker, period)``.

    No-ops when ``TSDB_PROVIDER != duckdb`` or when ``df`` is empty.
    """
    if not is_active() or df is None or df.empty:
        return
    try:
        adapter = _get_adapter()
        _ensure_table(adapter)
    except Exception as exc:
        logger.debug("duckdb_cache init failed", error=str(exc))
        return

    fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)
    records: list[dict] = []
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        records.append(
            {
                "ticker": ticker.upper(),
                "period": period,
                "ts": ts.to_pydatetime(),
                "open": _as_float(row.get("Open")),
                "high": _as_float(row.get("High")),
                "low": _as_float(row.get("Low")),
                "close": _as_float(row.get("Close")),
                "volume": _as_float(row.get("Volume")),
                "fetched_at": fetched_at,
            }
        )
    if not records:
        return
    try:
        # Upsert: replace any existing rows for this (ticker, period).
        adapter.query(
            f"DELETE FROM {_TABLE} WHERE ticker = ? AND period = ?",
            (ticker.upper(), period),
        )
        adapter.write(_TABLE, records)
        logger.debug(
            "duckdb_cache write", ticker=ticker, period=period, rows=len(records),
        )
    except Exception as exc:
        logger.warning("duckdb_cache write failed", error=str(exc))


def _as_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
