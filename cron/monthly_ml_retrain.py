"""
Monthly ML model retraining cron job.

Retrains the LightGBM alpha model on the configured ticker universe, saves
a fresh checkpoint to the path specified by ``LGBM_ALPHA_MODEL_PATH``, and
(optionally) logs the checkpoint + IC metrics to the MLflow tracking
registry via :func:`providers.model_registry.get_model_registry`.

Run manually:
    python -m cron.monthly_ml_retrain [--log-to-mlflow | --no-log-to-mlflow]

Schedule (runs after monthly walk-forward at 06:00):
    0 7 1 * * cd /path/to/quant-platform && .venv/bin/python -m cron.monthly_ml_retrain

ENV vars
-------
    WF_TICKERS             Comma-separated tickers (default: SPY,QQQ,AAPL,MSFT,TSLA)
    ML_TRAIN_PERIOD        yfinance period string for training data (default: 2y)
    LGBM_ALPHA_MODEL_PATH  Checkpoint path (default: models/lgbm_alpha.pkl)
    MODEL_REGISTRY_ENABLED 0 | 1 — opt in to the MLflow log step (default: 1)
    MLFLOW_REGISTERED_MODEL name under which the run is registered
                            (default: lgbm_alpha)
"""
from __future__ import annotations

import argparse
import os
import sys

from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
from utils.logger import get_logger

log = get_logger("cron.monthly_ml_retrain")

DEFAULT_TICKERS = "SPY,QQQ,AAPL,MSFT,TSLA"
_REGISTERED_MODEL_DEFAULT = "lgbm_alpha"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cron.monthly_ml_retrain",
        description="Monthly retrain of the LightGBM alpha model.",
    )
    parser.add_argument(
        "--log-to-mlflow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Log the fresh checkpoint + IC metrics to MLflow after training. "
            "When unset, falls back to MODEL_REGISTRY_ENABLED (default: on). "
            "Disable with --no-log-to-mlflow for dry runs / offline dev."
        ),
    )
    return parser


def _resolve_registry_flag(flag: bool | None) -> bool:
    if flag is not None:
        return flag
    raw = os.environ.get("MODEL_REGISTRY_ENABLED", "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _log_to_registry(model_path: str, train_result: dict, tickers: list[str], period: str) -> str | None:
    """Best-effort MLflow log. Returns the run_id on success, ``None`` otherwise.

    Any exception here is swallowed with a warning so a broken registry
    never halts the nightly retrain — the local checkpoint has already
    been written at this point.
    """
    try:
        from providers.model_registry import get_model_registry

        registry = get_model_registry()
    except Exception as exc:
        log.warning("ml_retrain: model_registry unavailable", error=str(exc))
        return None

    registered_name = os.environ.get(
        "MLFLOW_REGISTERED_MODEL", _REGISTERED_MODEL_DEFAULT,
    )
    metrics = {
        "train_ic":   float(train_result.get("train_ic", 0.0) or 0.0),
        "test_ic":    float(train_result.get("test_ic", 0.0) or 0.0),
        "train_icir": float(train_result.get("train_icir", 0.0) or 0.0),
        "test_icir":  float(train_result.get("test_icir", 0.0) or 0.0),
        "n_train":    float(train_result.get("n_train_samples", 0) or 0),
        "n_test":     float(train_result.get("n_test_samples", 0) or 0),
    }
    tags = {
        "model_name": registered_name,
        "period": period,
        "n_tickers": str(len(tickers)),
        "tickers": ",".join(tickers),
        "source": "cron.monthly_ml_retrain",
    }
    try:
        run_id = registry.log_model(
            run_name=f"{registered_name}_retrain",
            model_path=model_path,
            metrics=metrics,
            tags=tags,
        )
    except Exception as exc:
        log.warning("ml_retrain: registry.log_model failed", error=str(exc))
        return None

    if not run_id:
        log.info("ml_retrain: registry logged nothing (adapter returned empty run_id)")
        return None

    # Best-effort promotion: every successful retrain moves the new version
    # into `Staging`. Operators manually promote to `Production`.
    try:
        registry.promote(registered_name, run_id, "Staging")
    except Exception as exc:
        log.warning("ml_retrain: registry.promote failed", error=str(exc))

    log.info(
        "ml_retrain: mlflow logged",
        run_id=run_id,
        registered_model=registered_name,
        **{k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
    )
    return run_id


def main(argv: list[str] | None = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    log_to_registry = _resolve_registry_flag(args.log_to_mlflow)

    tickers_str = os.environ.get("WF_TICKERS", DEFAULT_TICKERS)
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    period = os.environ.get("ML_TRAIN_PERIOD", "2y")

    log.info(
        "ml_retrain: starting",
        n_tickers=len(tickers), period=period,
        log_to_registry=log_to_registry,
    )

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

        signal = MLSignal()
        result = signal.train(tickers, period=period, lgbm_params=best_params)

        log.info(
            "ml_retrain: complete",
            train_ic=round(result["train_ic"], 4),
            test_ic=round(result["test_ic"], 4),
            train_icir=round(result["train_icir"], 4),
            test_icir=round(result["test_icir"], 4),
            n_train=result["n_train_samples"],
            n_test=result["n_test_samples"],
        )

        if log_to_registry:
            _log_to_registry(signal._model_path, result, tickers, period)

    except RuntimeError as exc:
        log.error("ml_retrain: runtime error", error=str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error("ml_retrain: unexpected failure", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
