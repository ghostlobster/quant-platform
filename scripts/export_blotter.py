"""
scripts/export_blotter.py — export the trade blotter as CSV.

Reads from :func:`journal.trading_journal.get_journal` (the canonical
trade record) and emits a CSV row per trade. Filters by date range so
operators can produce monthly / quarterly blotters for tax + compliance.

Usage
-----
    python scripts/export_blotter.py [--from 2026-01-01] [--to 2026-12-31]
                                     [--ticker AAPL] [--out blotter.csv]

When ``--out`` is omitted the CSV is written to stdout.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_COLUMNS = (
    "id", "ticker", "side", "qty",
    "entry_price", "entry_time",
    "exit_price",  "exit_time",
    "pnl", "exit_reason",
    "signal_source", "regime",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.export_blotter",
        description="Export the trading journal as a CSV blotter.",
    )
    parser.add_argument("--from", dest="start_date", default=None,
                        help="Inclusive lower bound on entry_time (ISO date).")
    parser.add_argument("--to", dest="end_date", default=None,
                        help="Inclusive upper bound on entry_time (ISO date).")
    parser.add_argument("--ticker", default=None,
                        help="Filter to a single ticker symbol.")
    parser.add_argument("--out", default=None,
                        help="Output CSV path. Default: stdout.")
    return parser


def _load_journal(start_date, end_date, ticker):
    from journal.trading_journal import get_journal

    return get_journal(start_date=start_date, end_date=end_date, ticker=ticker)


def export(start_date, end_date, ticker, out_path) -> int:
    df = _load_journal(start_date, end_date, ticker)
    rows = df.to_dict(orient="records") if not df.empty else []

    if out_path:
        target = open(out_path, "w", encoding="utf-8", newline="")
    else:
        target = sys.stdout
    try:
        writer = csv.DictWriter(target, fieldnames=_COLUMNS,
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if out_path:
            target.close()
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    n = export(args.start_date, args.end_date, args.ticker, args.out)
    print(f"# exported {n} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
