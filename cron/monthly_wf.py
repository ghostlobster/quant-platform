"""
Monthly walk-forward backtest runner.
Run manually: python -m cron.monthly_wf
Schedule via cron: 0 6 1 * * /path/to/venv/bin/python -m cron.monthly_wf

Results saved to: data/wf_history.db (SQLite)
Table: wf_results (run_date, ticker, consistency_score, total_return, n_windows)
"""
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from backtester.walk_forward import walk_forward
from utils.logger import get_logger

logger = get_logger("cron.monthly_wf")

DB_PATH = Path(__file__).parent.parent / "data" / "wf_history.db"
DEFAULT_TICKERS = "SPY,QQQ,AAPL,MSFT,TSLA"


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS wf_results (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date  TEXT    NOT NULL,
            ticker    TEXT    NOT NULL,
            consistency_score REAL,
            total_return      REAL,
            n_windows         INTEGER,
            UNIQUE(run_date, ticker)
        )
    """)
    conn.commit()


def _upsert_result(
    conn: sqlite3.Connection,
    run_date: str,
    ticker: str,
    consistency_score: float,
    total_return: float,
    n_windows: int,
) -> None:
    conn.execute(
        """
        INSERT INTO wf_results (run_date, ticker, consistency_score, total_return, n_windows)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(run_date, ticker) DO UPDATE SET
            consistency_score = excluded.consistency_score,
            total_return      = excluded.total_return,
            n_windows         = excluded.n_windows
        """,
        (run_date, ticker, consistency_score, total_return, n_windows),
    )
    conn.commit()


def run() -> int:
    tickers = [
        t.strip().upper()
        for t in os.getenv("WF_TICKERS", DEFAULT_TICKERS).split(",")
        if t.strip()
    ]
    run_date = date.today().isoformat()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _init_db(conn)

    rows = []
    failed: list[str] = []

    for ticker in tickers:
        try:
            logger.info("Fetching 2y OHLCV for %s", ticker)
            df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            df = df.rename(columns=str.lower)

            logger.info("Running walk_forward for %s (%d bars)", ticker, len(df))
            wf = walk_forward(df, ticker=ticker)

            n_windows = len(wf.windows)
            _upsert_result(
                conn,
                run_date,
                ticker,
                wf.consistency_score,
                wf.avg_return,
                n_windows,
            )

            rows.append((ticker, wf.consistency_score, wf.avg_return, n_windows))
            logger.info(
                "%s  consistency=%.2f  avg_return=%.4f  windows=%d",
                ticker,
                wf.consistency_score,
                wf.avg_return,
                n_windows,
            )
        except Exception as exc:
            logger.error("FAILED %s: %s", ticker, exc)
            failed.append(ticker)

    conn.close()

    # Optional: retrain RL position sizer
    if os.getenv("RL_SIZER_RETRAIN", "0") == "1":
        try:
            from analysis.rl_trainer import train as train_rl_sizer
            from journal.trading_journal import get_trades
            trades = get_trades()
            if trades is not None and not trades.empty and "realised_pnl" in trades.columns:
                closes = trades[trades.get("action", pd.Series()).str.upper() == "SELL"] \
                    if "action" in trades.columns else trades
                if len(closes) >= 10:
                    logger.info("Retraining RL position sizer on %d closed trades", len(closes))
                    saved = train_rl_sizer(closes)
                    logger.info("RL sizer retrained: %s", saved)
                else:
                    logger.info("RL sizer skipped — fewer than 10 closed trades available")
        except ImportError:
            logger.info("RL sizer retraining skipped (stable-baselines3/gymnasium not installed)")
        except Exception as exc:
            logger.error("RL sizer retraining failed: %s", exc)

    # Summary table
    col = "{:<8} {:>14} {:>14} {:>10}"
    print("\n" + "=" * 52)
    print(f"Walk-Forward Results — {run_date}")
    print("=" * 52)
    print(col.format("Ticker", "Consistency", "Avg Return", "Windows"))
    print("-" * 52)
    for ticker, cs, tr, nw in rows:
        print(col.format(ticker, f"{cs:.2%}", f"{tr:.4f}", nw))
    if failed:
        print(f"\nFailed tickers: {', '.join(failed)}")
    print("=" * 52 + "\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
