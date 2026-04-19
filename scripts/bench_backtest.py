"""
scripts/bench_backtest.py — SQLite vs DuckDB OHLCV fetch benchmark.

Warms each cache with a shared ticker set, then times a bulk read of every
ticker at a given period. Designed to validate the #143 speedup without
booting the full backtester. Run:

    python scripts/bench_backtest.py [--tickers SPY,QQQ,...] [--period 1y]
                                     [--repeats 3]

The script calls :func:`data.fetcher.fetch_ohlcv` under each provider so
the comparison includes the real code path. The DuckDB run is skipped
when the ``duckdb`` package is not importable.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_TICKERS = "SPY,QQQ,IWM,AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META"
DEFAULT_PERIOD = "1y"
DEFAULT_REPEATS = 3


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.bench_backtest",
        description="Benchmark the OHLCV cache against SQLite and DuckDB.",
    )
    parser.add_argument("--tickers", default=DEFAULT_TICKERS)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    return parser


def _time_reads(tickers: list[str], period: str, repeats: int) -> list[float]:
    """Return wall-clock seconds for each repeat."""
    from data.fetcher import fetch_ohlcv

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for ticker in tickers:
            fetch_ohlcv(ticker, period)
        timings.append(time.perf_counter() - start)
    return timings


def _duckdb_available() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    # Warmup: make sure both caches have the data so we measure read speed,
    # not network fetch time.
    os.environ["TSDB_PROVIDER"] = "sqlite"
    _time_reads(tickers, args.period, repeats=1)

    sqlite_times = _time_reads(tickers, args.period, args.repeats)

    duckdb_times: list[float] = []
    if _duckdb_available():
        os.environ["TSDB_PROVIDER"] = "duckdb"
        # Warm DuckDB on the first pass.
        _time_reads(tickers, args.period, repeats=1)
        duckdb_times = _time_reads(tickers, args.period, args.repeats)

    def _stats(label: str, samples: list[float]) -> None:
        if not samples:
            print(f"{label:<10s} — skipped (backend unavailable)")
            return
        mean = statistics.mean(samples)
        median = statistics.median(samples)
        print(
            f"{label:<10s} repeats={len(samples)} "
            f"mean={mean * 1000:8.1f}ms  median={median * 1000:8.1f}ms  "
            f"per-ticker={median * 1000 / max(len(tickers), 1):6.1f}ms"
        )

    print(f"tickers={len(tickers)}  period={args.period}  repeats={args.repeats}")
    _stats("SQLite", sqlite_times)
    _stats("DuckDB", duckdb_times)

    if sqlite_times and duckdb_times:
        speedup = statistics.median(sqlite_times) / statistics.median(duckdb_times)
        print(f"\nDuckDB speedup over SQLite (median): {speedup:6.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
