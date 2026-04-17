"""
Monthly ML model retraining cron job.

Retrains the LightGBM alpha model on the configured ticker universe and saves
a fresh checkpoint to the path specified by LGBM_ALPHA_MODEL_PATH.

Run manually:
    python -m cron.monthly_ml_retrain

Schedule (runs after monthly walk-forward at 06:00):
    0 7 1 * * cd /path/to/quant-platform && .venv/bin/python -m cron.monthly_ml_retrain

ENV vars
-------
    WF_TICKERS           Comma-separated tickers (default: SPY,QQQ,AAPL,MSFT,TSLA)
    ML_TRAIN_PERIOD      yfinance period string for training data (default: 2y)
    LGBM_ALPHA_MODEL_PATH  Checkpoint path (default: models/lgbm_alpha.pkl)
"""
from __future__ import annotations

import os
import sys

from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
from utils.logger import get_logger

log = get_logger("cron.monthly_ml_retrain")

DEFAULT_TICKERS = "SPY,QQQ,AAPL,MSFT,TSLA"


def main() -> None:
    tickers_str = os.environ.get("WF_TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    period = os.environ.get("ML_TRAIN_PERIOD", "2y")

    log.info("ml_retrain: starting", n_tickers=len(tickers), period=period)

    try:
        if not _LGBM_AVAILABLE:
            log.error("ml_retrain: lightgbm is not installed — aborting")
            sys.exit(1)

        # Pick up tuned hyperparameters from the last Optuna run, if any.
        from strategies.ml_tuning import load_best_params

        best_params = load_best_params("lgbm_alpha")
        if best_params:
            log.info("ml_retrain: using tuned params", n_params=len(best_params))
        else:
            log.info("ml_retrain: no tuned params found, using defaults")

        result = MLSignal().train(tickers, period=period, lgbm_params=best_params)

        log.info(
            "ml_retrain: complete",
            train_ic=round(result["train_ic"], 4),
            test_ic=round(result["test_ic"], 4),
            train_icir=round(result["train_icir"], 4),
            test_icir=round(result["test_icir"], 4),
            n_train=result["n_train_samples"],
            n_test=result["n_test_samples"],
        )

    except RuntimeError as exc:
        log.error("ml_retrain: runtime error", error=str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error("ml_retrain: unexpected failure", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
