"""
cron/polygon_backfill.py — one-shot cache warmup against Polygon.io.

Usage
-----
    python -m cron.polygon_backfill [--tickers AAPL,SPY] [--timeframe 1Day]
                                    [--days 365] [--period 1y]

Behaviour
---------
Pulls ``--days`` of bars for each ticker at the requested ``--timeframe``
resolution via :class:`adapters.market_data.polygon_adapter.PolygonAdapter`
and, for daily bars, persists them into the ``data/fetcher.py`` SQLite
``price_cache`` table under ``(ticker, --period)``. Subsequent calls to
:func:`data.fetcher.fetch_ohlcv(ticker, period)` will hit the cache without
round-tripping yfinance.

Non-daily timeframes (``5Min`` / ``1Hour`` / ...) are fetched and counted
but not cached — the cache schema stores daily OHLCV keyed by yfinance
period strings; intraday caching is a follow-up.

Falls back silently when ``POLYGON_API_KEY`` is absent so the cron is safe
to run on any host.

ENV vars
--------
    POLYGON_BACKFILL_TICKERS   comma-separated tickers (default: WF_TICKERS env or SPY,QQQ)
    POLYGON_BACKFILL_DAYS      history window in days (default: 365)
    POLYGON_BACKFILL_TIMEFRAME timeframe string (default: 1Day)
    POLYGON_BACKFILL_PERIOD    cache-key period string (default: derived from --days)
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

import structlog

logger = structlog.get_logger(__name__)

DEFAULT_TIMEFRAME = "1Day"
DEFAULT_DAYS = 365
DEFAULT_TICKERS = "SPY,QQQ"

# yfinance-style period strings in ascending order of coverage.
_PERIOD_BUCKETS: tuple[tuple[int, str], ...] = (
    (1,    "1d"),
    (5,    "5d"),
    (30,   "1mo"),
    (90,   "3mo"),
    (180,  "6mo"),
    (365,  "1y"),
    (730,  "2y"),
    (1825, "5y"),
)


def _period_for_days(days: int) -> str:
    """Pick the smallest yfinance period string that covers ``days``.

    This keeps the cache key compatible with :func:`data.fetcher.fetch_ohlcv`
    so a subsequent call with the same ``period`` argument hits the cache.
    """
    for threshold, label in _PERIOD_BUCKETS:
        if days <= threshold:
            return label
    return "5y"


def _default_ticker_list() -> str:
    return (
        os.environ.get("POLYGON_BACKFILL_TICKERS")
        or os.environ.get("WF_TICKERS")
        or DEFAULT_TICKERS
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cron.polygon_backfill",
        description="Warm the OHLCV cache against Polygon.io.",
    )
    parser.add_argument(
        "--tickers",
        default=_default_ticker_list(),
        help="Comma-separated tickers. Falls back to WF_TICKERS then SPY,QQQ.",
    )
    parser.add_argument(
        "--timeframe",
        default=os.environ.get("POLYGON_BACKFILL_TIMEFRAME", DEFAULT_TIMEFRAME),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=int(os.environ.get("POLYGON_BACKFILL_DAYS", DEFAULT_DAYS)),
    )
    parser.add_argument(
        "--period",
        default=os.environ.get("POLYGON_BACKFILL_PERIOD"),
        help=(
            "yfinance period string used as the cache key "
            "(default: derived from --days). Must match the period later "
            "passed to data.fetcher.fetch_ohlcv for a cache hit."
        ),
    )
    return parser


def _bars_to_dataframe(bars: list[dict]):
    """Convert Polygon bar dicts to the cache's OHLCV DataFrame shape."""
    import pandas as pd

    if not bars:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    rows = []
    idx = []
    for bar in bars:
        t_ms = bar.get("t")
        if t_ms is None:
            continue
        ts = pd.Timestamp(int(t_ms), unit="ms", tz="UTC").tz_convert(None)
        idx.append(ts)
        rows.append(
            {
                "Open": bar.get("o"),
                "High": bar.get("h"),
                "Low": bar.get("l"),
                "Close": bar.get("c"),
                "Volume": bar.get("v"),
            }
        )
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
    df = df.dropna(subset=["Close"])
    return df


def backfill(
    tickers: list[str],
    timeframe: str,
    days: int,
    period: str | None = None,
) -> dict:
    """Pull bars for each ticker, cache the daily ones, and return counts.

    Returns an empty dict when ``POLYGON_API_KEY`` is not set so callers can
    run this cron unconditionally and no-op in local dev.
    """
    if not os.environ.get("POLYGON_API_KEY"):
        logger.warning("polygon_backfill: POLYGON_API_KEY unset — skipping")
        return {}

    from adapters.market_data.polygon_adapter import PolygonAdapter
    from data.db import init_db
    from data.fetcher import _cache_write

    init_db()  # ensure price_cache table exists before we upsert
    cache_period = period or _period_for_days(days)
    cacheable = timeframe.strip() == "1Day"

    end = date.today()
    start = end - timedelta(days=days)
    start_s = start.isoformat()
    end_s = end.isoformat()

    adapter = PolygonAdapter()
    counts: dict[str, int] = {}
    for raw in tickers:
        ticker = raw.strip().upper()
        if not ticker:
            continue
        try:
            bars = adapter.get_bars(ticker, timeframe, start_s, end_s)
        except Exception as exc:
            logger.warning(
                "polygon_backfill: fetch failed",
                ticker=ticker,
                error=str(exc),
            )
            counts[ticker] = 0
            continue

        counts[ticker] = len(bars)
        cache_hit = False
        if cacheable and bars:
            try:
                df = _bars_to_dataframe(bars)
                if not df.empty:
                    _cache_write(ticker, cache_period, df)
                    cache_hit = True
            except Exception as exc:
                logger.warning(
                    "polygon_backfill: cache write failed",
                    ticker=ticker,
                    error=str(exc),
                )
        logger.info(
            "polygon_backfill: pulled",
            ticker=ticker,
            bars=len(bars),
            start=start_s,
            end=end_s,
            cache_period=cache_period if cacheable else None,
            cached=cache_hit,
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    counts = backfill(tickers, args.timeframe, args.days, period=args.period)
    if not counts:
        logger.info("polygon_backfill: nothing to do")
        return 0
    logger.info("polygon_backfill: complete", **counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
