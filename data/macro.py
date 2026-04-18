"""
data/macro.py — Cached macro / alternative-data series.

Mirror of :mod:`data.fetcher` but for macro indicators (FRED, etc.).
Reads through :func:`providers.macro.get_macro` and stores observations
in the ``macro_cache`` table of ``quant.db`` so subsequent reads avoid
the upstream HTTP round-trip.

Public API
----------
``fetch_macro_series(series_id, start, end)`` — single series, cached.
``macro_context_features(dates, series_ids)`` — date-indexed DataFrame
of multiple series, ready to merge into ``data/features.py`` outputs.

Usage
-----
    from data.macro import fetch_macro_series, macro_context_features

    vix = fetch_macro_series("VIXCLS", start="2023-01-01")
    ctx = macro_context_features(my_dates, ["VIXCLS", "T10Y2Y"])
"""
from __future__ import annotations

import time
from typing import Iterable, Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_CONTEXT_SERIES = ("VIXCLS", "T10Y2Y", "DGS10")


def _read_cache(series_id: str) -> pd.Series:
    try:
        from data.db import get_connection
    except Exception:
        return pd.Series(dtype=float, name=series_id)
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT obs_date, value FROM macro_cache WHERE series_id = ? ORDER BY obs_date",
            (series_id,),
        ).fetchall()
    except Exception as exc:
        log.warning("macro: cache read failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float, name=series_id)
    if not rows:
        return pd.Series(dtype=float, name=series_id)
    dates = pd.DatetimeIndex([r[0] for r in rows])
    values = [r[1] for r in rows]
    return pd.Series(values, index=dates, name=series_id, dtype=float)


def _write_cache(series_id: str, series: pd.Series) -> None:
    if series.empty:
        return
    try:
        from data.db import get_connection
        conn = get_connection()
        now = time.time()
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO macro_cache (series_id, obs_date, value, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (series_id, ts.date().isoformat(), float(v), now)
                    for ts, v in series.items()
                    if pd.notna(v)
                ],
            )
    except Exception as exc:
        log.warning("macro: cache write failed", series_id=series_id, error=str(exc))


def fetch_macro_series(
    series_id: str,
    start: str | None = None,
    end: str | None = None,
    provider_name: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """Return a date-indexed observation series for ``series_id``.

    On a cache hit (and no explicit ``start`` / ``end`` window), the
    cached series is returned directly. Otherwise the configured
    :class:`providers.macro.MacroDataProvider` is queried and the
    result upserted into ``macro_cache``.
    """
    if use_cache and start is None and end is None:
        cached = _read_cache(series_id)
        if not cached.empty:
            return cached

    try:
        from providers.macro import get_macro
        provider = get_macro(provider_name)
    except Exception as exc:
        log.warning("macro: provider unavailable", error=str(exc))
        return pd.Series(dtype=float, name=series_id)

    try:
        series = provider.get_series(series_id, start=start, end=end)
    except Exception as exc:
        log.warning("macro: provider call failed", series_id=series_id, error=str(exc))
        return pd.Series(dtype=float, name=series_id)

    if use_cache:
        _write_cache(series_id, series)
    return series


def macro_context_features(
    dates: Iterable[pd.Timestamp],
    series_ids: Iterable[str] = _DEFAULT_CONTEXT_SERIES,
    provider_name: Optional[str] = None,
) -> pd.DataFrame:
    """Build a date-indexed DataFrame with one column per macro series.

    Each series is forward-filled onto the requested ``dates`` so daily
    panels (which include weekends / holidays missing from FRED) get a
    valid value. Missing series collapse to NaN columns rather than
    raising.
    """
    idx = pd.DatetimeIndex(sorted(set(pd.Timestamp(d) for d in dates)))
    if len(idx) == 0:
        return pd.DataFrame()

    start = idx.min().date().isoformat()
    end = idx.max().date().isoformat()

    out = pd.DataFrame(index=idx)
    for sid in series_ids:
        s = fetch_macro_series(sid, start=start, end=end, provider_name=provider_name)
        if s.empty:
            out[sid] = pd.Series(index=idx, dtype=float)
            continue
        out[sid] = s.reindex(idx, method="ffill")
    return out
