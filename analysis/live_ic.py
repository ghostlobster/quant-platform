"""
analysis/live_ic.py — Live Information Coefficient (IC) estimator.

Persists every scored (ticker, model) row to the ``live_predictions`` table
in ``quant.db``, back-fills the realized forward-return after the horizon
expires, and computes a rolling Spearman rank-IC that the
``KnowledgeAdaptionAgent`` consumes as ``context["live_ic"]``.

Closes the feedback loop that the agent has been missing — without this
module the agent's IC-degradation branch never fires, so a fresh pickle
with collapsed alpha slips through unnoticed.

ENV vars
--------
    KNOWLEDGE_RECORD_PREDICTIONS  1 → persist scored rows (default)
                                  0 → no-op writer (for tests / dry-runs)
"""
from __future__ import annotations

import math
import os
import time

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

# Rolling-IC cache — keyed on (model_name, window, horizon_d).
_IC_CACHE_TTL_SEC = 300.0
_ic_cache: dict[tuple[str, int, int], tuple[float, float | None]] = {}


def _enabled() -> bool:
    """Return True when the writer is active. Default on."""
    raw = os.environ.get("KNOWLEDGE_RECORD_PREDICTIONS")
    if raw is None:
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _invalidate_ic_cache(model_name: str | None = None) -> None:
    if model_name is None:
        _ic_cache.clear()
        return
    for key in [k for k in _ic_cache if k[0] == model_name]:
        _ic_cache.pop(key, None)


# ── Writer ───────────────────────────────────────────────────────────────────

def record_prediction(
    ticker: str,
    model_name: str,
    score: float,
    horizon_d: int = 5,
    ts: float | None = None,
) -> None:
    """Upsert a single scored row into ``live_predictions``.

    Silent no-op when ``KNOWLEDGE_RECORD_PREDICTIONS=0`` or when the DB
    is unreachable — live-IC plumbing must never break the trading path.
    """
    if not _enabled():
        return
    record_predictions({ticker: float(score)}, model_name=model_name,
                       horizon_d=horizon_d, ts=ts)


def record_predictions(
    scores: dict[str, float],
    model_name: str = "lgbm_alpha",
    horizon_d: int = 5,
    ts: float | None = None,
) -> int:
    """Batch wrapper — one transaction for N tickers. Returns rows written.

    Fails silently on any DB error (logged at WARNING) so the caller can
    stay oblivious to persistence outages.
    """
    if not _enabled() or not scores:
        return 0
    stamp = float(ts) if ts is not None else time.time()
    try:
        from data.db import get_connection
    except Exception:
        return 0
    try:
        conn = get_connection()
    except Exception as exc:
        logger.warning("live_ic: DB unavailable for record", error=str(exc))
        return 0
    try:
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO live_predictions "
                "(ts, ticker, model_name, score, horizon_d) "
                "VALUES (?, ?, ?, ?, ?)",
                [
                    (stamp, ticker, model_name, float(score), int(horizon_d))
                    for ticker, score in scores.items()
                ],
            )
        return len(scores)
    except Exception as exc:
        logger.warning("live_ic: record_predictions failed", error=str(exc))
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── Backfill ─────────────────────────────────────────────────────────────────

def _fetch_realized_for_ticker(
    ticker: str,
    period: str = "3mo",
):
    """Thin wrapper around :func:`data.fetcher.fetch_ohlcv` — test seam."""
    from data.fetcher import fetch_ohlcv
    return fetch_ohlcv(ticker, period=period)


def _realized_return(
    df,
    ts: float,
    horizon_d: int,
) -> float | None:
    """Return the realized pct-return between ``ts`` and ``ts+horizon_d`` days,
    using the closest on-or-after bars in ``df``.

    Returns ``None`` when either anchor is missing (e.g., weekend, fresh
    row whose data hasn't landed yet)."""
    if df is None or len(df) == 0 or "Close" not in df.columns:
        return None
    import pandas as pd

    # fetch_ohlcv returns a DatetimeIndex with naive timestamps. Compare in
    # UTC seconds so ``ts`` (unix epoch) and index values speak the same units.
    ts0 = pd.Timestamp(ts, unit="s", tz="UTC").tz_localize(None)
    ts1 = ts0 + pd.Timedelta(days=int(horizon_d))
    try:
        idx_before = df.index.searchsorted(ts0, side="left")
        idx_after = df.index.searchsorted(ts1, side="left")
    except Exception:
        return None
    # searchsorted returns len(index) when the key is past the end.
    if idx_before >= len(df) or idx_after >= len(df):
        return None
    try:
        p0 = float(df["Close"].iloc[idx_before])
        p1 = float(df["Close"].iloc[idx_after])
    except Exception:
        return None
    if p0 == 0 or math.isnan(p0) or math.isnan(p1):
        return None
    return (p1 / p0) - 1.0


def backfill_realized(
    model_name: str | None = None,
    now: float | None = None,
    max_rows: int = 1000,
) -> int:
    """Fill ``realized`` for every row whose horizon has expired.

    One ``fetch_ohlcv`` call per distinct ticker in the candidate set —
    the fetcher's own cache layer dedups across runs. Bounded by
    ``max_rows`` so a catch-up run after downtime cannot exhaust memory.
    Returns the number of rows updated.
    """
    now = float(now) if now is not None else time.time()
    try:
        from data.db import get_connection
    except Exception:
        return 0
    try:
        conn = get_connection()
    except Exception as exc:
        logger.warning("live_ic: DB unavailable for backfill", error=str(exc))
        return 0
    try:
        query = (
            "SELECT ts, ticker, model_name, horizon_d FROM live_predictions "
            "WHERE realized IS NULL AND (ts + horizon_d * 86400.0) < ?"
        )
        params: list = [now]
        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY ts ASC LIMIT ?"
        params.append(int(max_rows))
        rows = conn.execute(query, params).fetchall()
    except Exception as exc:
        logger.warning("live_ic: backfill select failed", error=str(exc))
        try:
            conn.close()
        except Exception:
            pass
        return 0

    if not rows:
        try:
            conn.close()
        except Exception:
            pass
        return 0

    # One OHLCV fetch per unique ticker.
    tickers = sorted({row["ticker"] for row in rows})
    ohlcv_cache: dict[str, object] = {}
    for ticker in tickers:
        try:
            ohlcv_cache[ticker] = _fetch_realized_for_ticker(ticker)
        except Exception as exc:
            logger.warning(
                "live_ic: fetch_ohlcv failed", ticker=ticker, error=str(exc),
            )
            ohlcv_cache[ticker] = None

    updated = 0
    try:
        with conn:
            for row in rows:
                realized = _realized_return(
                    ohlcv_cache.get(row["ticker"]),
                    ts=row["ts"],
                    horizon_d=row["horizon_d"],
                )
                if realized is None:
                    continue
                conn.execute(
                    "UPDATE live_predictions SET realized = ? "
                    "WHERE ts = ? AND ticker = ? "
                    "AND model_name = ? AND horizon_d = ?",
                    (
                        float(realized),
                        row["ts"], row["ticker"],
                        row["model_name"], row["horizon_d"],
                    ),
                )
                updated += 1
    except Exception as exc:
        logger.warning("live_ic: backfill update failed", error=str(exc))
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if updated:
        _invalidate_ic_cache(model_name)
    return updated


# ── Estimator ────────────────────────────────────────────────────────────────

def rolling_live_ic(
    model_name: str,
    window: int = 60,
    horizon_d: int = 5,
) -> float | None:
    """Rolling Spearman rank-IC over the last ``window`` realized rows.

    Returns ``None`` when fewer than ``max(10, window // 2)`` rows are
    available — warm-up protection so a just-deployed model doesn't
    immediately report a misleading IC computed on 2-3 points.
    Cached per ``(model_name, window, horizon_d)`` for
    ``_IC_CACHE_TTL_SEC`` seconds (override in tests).
    """
    key = (model_name, int(window), int(horizon_d))
    now = time.time()
    cached = _ic_cache.get(key)
    if cached is not None and (now - cached[0]) < _IC_CACHE_TTL_SEC:
        return cached[1]

    try:
        from data.db import get_connection
    except Exception:
        return None
    try:
        conn = get_connection()
    except Exception:
        return None
    try:
        rows = conn.execute(
            "SELECT score, realized FROM live_predictions "
            "WHERE model_name = ? AND horizon_d = ? AND realized IS NOT NULL "
            "ORDER BY ts DESC LIMIT ?",
            (model_name, int(horizon_d), int(window)),
        ).fetchall()
    except Exception as exc:
        logger.warning("live_ic: rolling select failed", error=str(exc))
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    min_rows = max(10, window // 2)
    if len(rows) < min_rows:
        _ic_cache[key] = (now, None)
        return None

    scores = np.asarray([float(r["score"]) for r in rows], dtype=float)
    realized = np.asarray([float(r["realized"]) for r in rows], dtype=float)

    from analysis.factor_ic import _spearman_corr
    ic = _spearman_corr(scores, realized)
    value = None if (ic is None or (isinstance(ic, float) and math.isnan(ic))) else float(ic)
    _ic_cache[key] = (now, value)
    return value
