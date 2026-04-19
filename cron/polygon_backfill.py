"""
cron/polygon_backfill.py — one-shot cache warmup against Polygon.io.

Usage
-----
    python -m cron.polygon_backfill [--tickers AAPL,SPY] [--timeframe 1Day]
                                    [--days 365]

Behaviour
---------
Pulls ``--days`` of bars for each ticker at the requested ``--timeframe``
resolution via :class:`adapters.market_data.polygon_adapter.PolygonAdapter`
and stores them in the ``data/fetcher.py`` SQLite cache. Falls back
silently when ``POLYGON_API_KEY`` is absent so the cron is safe to run
on any host.

ENV vars
--------
    POLYGON_BACKFILL_TICKERS   comma-separated tickers (default: WF_TICKERS env or SPY,QQQ)
    POLYGON_BACKFILL_DAYS      history window in days (default: 365)
    POLYGON_BACKFILL_TIMEFRAME timeframe string (default: 1Day)
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
    return parser


def backfill(tickers: list[str], timeframe: str, days: int) -> dict:
    """Pull bars for each ticker and return a per-ticker row-count dict.

    Returns an empty dict when ``POLYGON_API_KEY`` is not set so callers can
    run this cron unconditionally and no-op in local dev.
    """
    if not os.environ.get("POLYGON_API_KEY"):
        logger.warning("polygon_backfill: POLYGON_API_KEY unset — skipping")
        return {}

    from adapters.market_data.polygon_adapter import PolygonAdapter

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
        logger.info(
            "polygon_backfill: pulled",
            ticker=ticker,
            bars=len(bars),
            start=start_s,
            end=end_s,
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    counts = backfill(tickers, args.timeframe, args.days)
    if not counts:
        logger.info("polygon_backfill: nothing to do")
        return 0
    logger.info("polygon_backfill: complete", **counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
