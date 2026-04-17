"""
strategies/ml_tuning.py — Bayesian hyperparameter tuning for the LightGBM alpha model.

Uses Optuna's Tree-structured Parzen Estimator (TPE) sampler to search the
hyperparameter space far more efficiently than grid or random search. The
objective is mean out-of-fold Information Coefficient (IC), evaluated via
``purged_walk_forward`` to avoid look-ahead bias.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading*, Ch 6 (Bayesian
    hyperparameter optimization); Ch 12 (LightGBM tuning in practice).

Optional deps
-------------
    optuna  (``pip install optuna>=3.5.0``)
    lightgbm (already required by ``strategies.ml_signal``)
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from data.features import _FEATURE_COLS, build_feature_matrix
from strategies.ml_signal import _LGBM_AVAILABLE, _TARGET_COL
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import optuna  # type: ignore[import]
    _OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None  # type: ignore[assignment]
    _OPTUNA_AVAILABLE = False


def _purged_splits(
    feature_matrix: pd.DataFrame,
    n_splits: int,
    embargo: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate ``(train_idx, test_idx)`` positional tuples with an embargo gap.

    Uses integer positions in the feature_matrix row ordering; the caller is
    expected to pass a MultiIndex frame ordered by date.
    """
    dates = feature_matrix.index.get_level_values("date").unique().sort_values()
    if len(dates) < n_splits + 1:
        return []

    fold_size = len(dates) // (n_splits + 1)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        test_start = train_end + embargo
        test_end = test_start + fold_size
        if test_end > len(dates):
            break
        train_dates = dates[:train_end]
        test_dates = dates[test_start:test_end]
        train_mask = feature_matrix.index.get_level_values("date").isin(train_dates)
        test_mask = feature_matrix.index.get_level_values("date").isin(test_dates)
        splits.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    return splits


def _objective_ic(
    trial,
    feature_matrix: pd.DataFrame,
    feature_cols: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Objective: maximise mean Spearman IC across purged folds."""
    import lightgbm as lgb  # type: ignore[import]

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "verbose": -1,
    }

    ics: list[float] = []
    X = feature_matrix[feature_cols].fillna(0.0).to_numpy()
    y = feature_matrix[_TARGET_COL].fillna(0.0).to_numpy()

    for train_idx, test_idx in splits:
        model = lgb.LGBMRegressor(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        if len(preds) < 3 or np.std(preds) == 0:
            continue
        ic, _ = spearmanr(preds, y[test_idx])
        if not np.isnan(ic):
            ics.append(float(ic))

    return float(np.mean(ics)) if ics else 0.0


def tune_lgbm_hyperparams(
    tickers: list[str],
    period: str = "2y",
    n_trials: int = 30,
    n_splits: int = 3,
    embargo: int = 5,
    seed: int = 42,
) -> dict:
    """Run a Bayesian search over the LightGBM hyperparameter space.

    Parameters
    ----------
    tickers   : universe used to build the feature matrix.
    period    : yfinance period string passed to ``build_feature_matrix``.
    n_trials  : number of Optuna trials (default 30).
    n_splits  : purged CV folds (default 3).
    embargo   : number of days held out between train and test folds.

    Returns
    -------
    dict with keys ``best_params``, ``best_ic``, ``n_trials`` (actual),
    ``n_samples``.

    Raises
    ------
    RuntimeError if optuna or lightgbm is not installed.
    """
    if not _OPTUNA_AVAILABLE:
        raise RuntimeError(
            "optuna is required for tune_lgbm_hyperparams; "
            "run `pip install optuna>=3.5.0`"
        )
    if not _LGBM_AVAILABLE:
        raise RuntimeError(
            "lightgbm is required for tune_lgbm_hyperparams"
        )

    fm = build_feature_matrix(tickers, period=period)
    if fm.empty:
        raise RuntimeError("tune_lgbm_hyperparams: feature matrix is empty")

    feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
    splits = _purged_splits(fm, n_splits=n_splits, embargo=embargo)
    if not splits:
        raise RuntimeError(
            "tune_lgbm_hyperparams: not enough data for "
            f"{n_splits} purged folds (have {len(fm)} rows)"
        )

    log.info(
        "tune_lgbm_hyperparams: starting",
        n_trials=n_trials, n_splits=len(splits), n_samples=len(fm),
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _objective_ic(trial, fm, feature_cols, splits),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    result = {
        "best_params": dict(study.best_params),
        "best_ic": float(study.best_value),
        "n_trials": len(study.trials),
        "n_samples": int(len(fm)),
    }
    log.info("tune_lgbm_hyperparams: complete", **{k: v for k, v in result.items()
                                                    if k != "best_params"})
    return result


# ── Persistence helpers for tuned params ─────────────────────────────────────

def save_best_params(
    model_name: str,
    params: dict,
    best_ic: float,
) -> None:
    """Upsert tuned hyperparameters into quant.db model_best_params.

    Logs and swallows any DB error so calling code (typically a UI button
    handler) does not abort after a successful tune just because persistence
    failed.
    """
    from data.db import get_connection

    try:
        conn = get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO model_best_params (model_name, updated_at, params_json, best_ic)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(model_name) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    params_json = excluded.params_json,
                    best_ic = excluded.best_ic
                """,
                (model_name, time.time(), json.dumps(params), float(best_ic)),
            )
        conn.close()
        log.info("save_best_params: persisted", model_name=model_name, best_ic=best_ic)
    except Exception as exc:
        log.warning("save_best_params: failed", model_name=model_name, error=str(exc))


def load_best_params(model_name: str) -> dict | None:
    """Return persisted tuned hyperparameters for ``model_name`` or None."""
    from data.db import get_connection

    try:
        conn = get_connection()
        row = conn.execute(
            "SELECT params_json FROM model_best_params WHERE model_name = ?",
            (model_name,),
        ).fetchone()
        conn.close()
        if row is None:
            return None
        return json.loads(row["params_json"])
    except Exception as exc:
        log.warning("load_best_params: failed", model_name=model_name, error=str(exc))
        return None
