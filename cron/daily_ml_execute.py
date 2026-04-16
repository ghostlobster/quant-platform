"""
Daily ML signal execution cron job.

Loads the trained LightGBM alpha model, scores the configured ticker universe,
and translates the resulting scores into broker orders via the configured
``BROKER_PROVIDER`` (paper by default). Intended to run after the US market
close on weekdays.

Run manually:
    python -m cron.daily_ml_execute

Schedule (16:05 ET on weekdays):
    5 16 * * 1-5 cd /path/to/quant-platform && \\
        .venv/bin/python -m cron.daily_ml_execute

ENV vars
--------
    WF_TICKERS             Comma-separated tickers (default: SPY,QQQ,AAPL,MSFT,TSLA)
    ML_SCORE_THRESHOLD     Minimum |score| required to act (default: 0.3)
    ML_MAX_POSITIONS       Maximum simultaneous longs (default: 5)
    BROKER_PROVIDER        paper | alpaca | ibkr | schwab (default: paper)
    LGBM_ALPHA_MODEL_PATH  Path to the trained model checkpoint
"""
from __future__ import annotations

import os
import sys

from strategies.ml_execution import execute_ml_signals
from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
from utils.logger import get_logger

log = get_logger("cron.daily_ml_execute")

DEFAULT_TICKERS = "SPY,QQQ,AAPL,MSFT,TSLA"


def main() -> None:
    tickers_str = os.environ.get("WF_TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    threshold = float(os.environ.get("ML_SCORE_THRESHOLD", "0.3"))
    max_positions = int(os.environ.get("ML_MAX_POSITIONS", "5"))
    broker_name = os.environ.get("BROKER_PROVIDER", "paper")

    log.info(
        "daily_ml_execute: starting",
        n_tickers=len(tickers),
        threshold=threshold,
        max_positions=max_positions,
        broker=broker_name,
    )

    if not _LGBM_AVAILABLE:
        log.error("daily_ml_execute: lightgbm is not installed — aborting")
        sys.exit(1)

    try:
        model = MLSignal()
        if model._model is None:
            log.error(
                "daily_ml_execute: no trained baseline model found — "
                "run cron.monthly_ml_retrain first"
            )
            sys.exit(1)

        scores = model.predict(tickers, period="6mo")
        log.info("daily_ml_execute: scored", n_scores=len(scores))

        actions = execute_ml_signals(
            scores,
            threshold=threshold,
            max_positions=max_positions,
        )

        log.info(
            "daily_ml_execute: complete",
            n_actions=len(actions),
            actions=actions,
        )

    except Exception as exc:
        log.error("daily_ml_execute: unexpected failure", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
