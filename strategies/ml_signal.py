"""
strategies/ml_signal.py — LightGBM alpha model for cross-sectional return prediction.

Trains a gradient-boosting regressor to predict 5-day forward returns across
a ticker universe.  Produces a ranked signal in [-1, 1] per ticker, suitable
for long/short or momentum-overlay strategies.

Falls back to a momentum-composite score when lightgbm is not installed or
no trained model checkpoint is present.

ENV vars
--------
    LGBM_ALPHA_MODEL_PATH   path to pickle checkpoint (default: models/lgbm_alpha.pkl)

Optional dependencies (guarded — app loads without them):
    lightgbm >= 4.0.0
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data.features import _FEATURE_COLS, build_feature_matrix
from utils.logger import get_logger

log = get_logger(__name__)

# ── Optional LightGBM import ──────────────────────────────────────────────────
try:
    import lightgbm as lgb  # type: ignore[import]
    _LGBM_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _LGBM_AVAILABLE = False

_DEFAULT_MODEL_PATH = os.environ.get("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")
_TARGET_COL = "fwd_ret_5d"

_DEFAULT_LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}


class MLSignal:
    """
    LightGBM cross-sectional alpha model.

    Falls back to momentum score when lightgbm is unavailable or no
    checkpoint has been trained yet.

    Attributes
    ----------
    _model      : fitted LGBMRegressor or None
    _model_path : filesystem path for checkpoint persistence
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._model = None
        self._load_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        """Load a persisted model checkpoint if it exists (mirrors rl_sizer.py pattern)."""
        path = Path(self._model_path)
        if not path.exists():
            log.info("ml_signal: no checkpoint found, will use fallback", path=str(path))
            return
        try:
            if not _LGBM_AVAILABLE:
                log.info("ml_signal: lightgbm not installed, using momentum fallback")
                return
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("ml_signal: loaded model checkpoint", path=str(path))
        except Exception as exc:
            log.warning("ml_signal: failed to load checkpoint", path=str(path), error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        lgbm_params: dict | None = None,
    ) -> dict[str, float]:
        """
        Build feature matrix, train LightGBM, evaluate on held-out test split,
        persist the model, and write metadata to quant.db.

        Parameters
        ----------
        tickers    : universe of tickers to train on
        period     : yfinance period string for data fetching
        test_size  : fraction of time series reserved for out-of-sample eval
                     (chronological split — latest test_size of dates)
        lgbm_params : override default LGBMRegressor parameters

        Returns
        -------
        dict with: train_ic, test_ic, train_icir, test_icir,
                   n_train_samples, n_test_samples

        Raises
        ------
        RuntimeError if lightgbm is not installed
        """
        if not _LGBM_AVAILABLE:
            raise RuntimeError(
                "lightgbm is not installed. Run: pip install lightgbm>=4.0.0"
            )

        log.info("ml_signal: building feature matrix", tickers=len(tickers), period=period)
        fm = build_feature_matrix(tickers, period=period)

        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")

        # Drop rows with missing target
        fm = fm.dropna(subset=[_TARGET_COL])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{_TARGET_COL}'")

        # Chronological train/test split on unique dates
        all_dates = sorted(fm.index.get_level_values("date").unique())
        split_idx = int(len(all_dates) * (1 - test_size))
        train_dates = set(all_dates[:split_idx])
        test_dates = set(all_dates[split_idx:])

        feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]

        train_df = fm[fm.index.get_level_values("date").isin(train_dates)]
        test_df = fm[fm.index.get_level_values("date").isin(test_dates)]

        X_train = train_df[feature_cols].values
        y_train = train_df[_TARGET_COL].values
        X_test = test_df[feature_cols].values
        y_test = test_df[_TARGET_COL].values

        log.info(
            "ml_signal: training",
            n_train=len(X_train),
            n_test=len(X_test),
            features=len(feature_cols),
        )

        params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        self._model = model

        # Evaluate IC on train and test sets
        train_ic, train_icir = self._eval_ic(model, X_train, y_train)
        test_ic, test_icir = self._eval_ic(model, X_test, y_test)

        log.info(
            "ml_signal: training complete",
            train_ic=round(train_ic, 4),
            test_ic=round(test_ic, 4),
        )

        # Persist checkpoint
        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("ml_signal: checkpoint saved", path=self._model_path)

        # Write metadata to quant.db
        self._write_metadata(
            n_tickers=len(tickers),
            period=period,
            train_ic=train_ic,
            test_ic=test_ic,
        )

        return {
            "train_ic": train_ic,
            "test_ic": test_ic,
            "train_icir": train_icir,
            "test_icir": test_icir,
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
        }

    @staticmethod
    def _eval_ic(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return (IC mean, ICIR) from a predictions array vs actuals."""
        preds = model.predict(X)
        # Spearman via rank correlation
        from analysis.factor_ic import _spearman_corr
        ic = _spearman_corr(preds, y)
        # Single-split IC: return IC as mean, ICIR undefined (set to IC)
        return float(ic), float(ic)

    def _write_metadata(
        self,
        n_tickers: int,
        period: str,
        train_ic: float,
        test_ic: float,
    ) -> None:
        """Persist training metadata to quant.db model_metadata table."""
        try:
            from data.db import get_connection
            conn = get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO model_metadata
                        (model_name, trained_at, train_ic, test_ic, n_tickers, period)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("lgbm_alpha", time.time(), train_ic, test_ic, n_tickers, period),
                )
            conn.close()
        except Exception as exc:
            log.warning("ml_signal: could not write metadata", error=str(exc))

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        tickers: list[str],
        period: str = "6mo",
    ) -> dict[str, float]:
        """
        Return ranked alpha scores in [-1, 1] for each ticker.

        Uses the most recent date's rows from the feature matrix.
        Scores are z-scored raw predictions clipped to [-1, 1].

        Falls back to momentum score when no model is loaded.

        Parameters
        ----------
        tickers : universe to score
        period  : data window for feature computation

        Returns
        -------
        dict mapping ticker → score in [-1.0, 1.0]
        """
        if self._model is None:
            return self._momentum_fallback(tickers, period)

        try:
            fm = build_feature_matrix(tickers, period=period)
            if fm.empty:
                return self._momentum_fallback(tickers, period)

            feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]

            # Take the most recent date's rows for each ticker
            last_date = fm.index.get_level_values("date").max()
            latest = fm.xs(last_date, level="date")[feature_cols]

            if latest.empty:
                return self._momentum_fallback(tickers, period)

            raw_preds = self._model.predict(latest.values)

            # Z-score and clip to [-1, 1]
            std = raw_preds.std()
            if std == 0:
                scores = np.zeros(len(raw_preds))
            else:
                scores = (raw_preds - raw_preds.mean()) / std

            scores = np.clip(scores, -1.0, 1.0)
            return {ticker: float(scores[i]) for i, ticker in enumerate(latest.index)}

        except Exception as exc:
            log.warning("ml_signal.predict: error, falling back to momentum", error=str(exc))
            return self._momentum_fallback(tickers, period)

    @staticmethod
    def _momentum_fallback(tickers: list[str], period: str) -> dict[str, float]:
        """Score each ticker using composite momentum when the ML model is unavailable."""
        from data.fetcher import fetch_ohlcv
        from strategies.momentum import compute_momentum_score

        scores: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, period)
                if df is not None and not df.empty:
                    mom = compute_momentum_score(df)
                    last_val = mom.dropna().iloc[-1] if not mom.dropna().empty else 0.0
                    scores[ticker] = float(np.clip(last_val, -1.0, 1.0))
                else:
                    scores[ticker] = 0.0
            except Exception:
                scores[ticker] = 0.0
        return scores

    # ── Feature importance ────────────────────────────────────────────────────

    def feature_importance(self) -> pd.DataFrame:
        """
        Return a DataFrame of feature importances sorted descending.

        Returns
        -------
        pd.DataFrame with columns: feature, importance
        Empty DataFrame if no model is loaded.
        """
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])

        try:
            feature_cols = _FEATURE_COLS
            importances = self._model.feature_importances_
            df = pd.DataFrame({
                "feature": feature_cols[:len(importances)],
                "importance": importances,
            })
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception as exc:
            log.warning("ml_signal.feature_importance: error", error=str(exc))
            return pd.DataFrame(columns=["feature", "importance"])
