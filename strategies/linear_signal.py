"""
strategies/linear_signal.py — Ridge regression linear alpha model.

Provides a linear factor model complement to the LightGBM model in ml_signal.py.
Ridge regression (L2-regularised OLS) produces interpretable factor coefficients
that show which features drive the signal — useful for understanding and auditing
the model's behaviour.

Same public interface as MLSignal: train(), predict(), feature_coefficients().

ENV vars
--------
    RIDGE_ALPHA_MODEL_PATH   path to model pickle (default: models/ridge_alpha.pkl)
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.factor_ic import _spearman_corr
from data.features import _FEATURE_COLS, build_feature_matrix
from utils.logger import get_logger

log = get_logger(__name__)

# ── Optional scikit-learn import ──────────────────────────────────────────────
try:
    from sklearn.linear_model import Ridge
    _SKLEARN_AVAILABLE = True
except ImportError:
    Ridge = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

_DEFAULT_MODEL_PATH = os.environ.get("RIDGE_ALPHA_MODEL_PATH", "models/ridge_alpha.pkl")
_TARGET_COL = "fwd_ret_5d"


class LinearSignal:
    """
    Ridge regression cross-sectional alpha model.

    Attributes
    ----------
    _model      : fitted Ridge or None
    _model_path : filesystem path for checkpoint
    _feature_cols : feature columns used at train time
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._model = None  # Ridge | None
        self._feature_cols: list[str] = list(_FEATURE_COLS)
        self._load_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            log.info("linear_signal: no checkpoint found, will use momentum fallback", path=str(path))
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("linear_signal: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning("linear_signal: failed to load checkpoint", path=str(path), error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        alpha: float = 1.0,
    ) -> dict[str, float]:
        """
        Build feature matrix, train Ridge regression, evaluate IC on held-out
        test split, and persist the model.

        Parameters
        ----------
        tickers   : universe of tickers to train on
        period    : yfinance period string for data fetching
        test_size : fraction of time series reserved for out-of-sample eval
                    (chronological split — latest test_size of dates)
        alpha     : Ridge regularisation strength (L2 penalty)

        Returns
        -------
        dict with: train_ic, test_ic, train_icir, test_icir,
                   n_train_samples, n_test_samples
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"
            )

        log.info("linear_signal: building feature matrix", tickers=len(tickers), period=period)
        fm = build_feature_matrix(tickers, period=period)

        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")

        fm = fm.dropna(subset=[_TARGET_COL])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{_TARGET_COL}'")

        # Chronological train/test split on unique dates
        all_dates = sorted(fm.index.get_level_values("date").unique())
        split_idx = int(len(all_dates) * (1 - test_size))
        train_dates = set(all_dates[:split_idx])
        test_dates = set(all_dates[split_idx:])

        feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
        self._feature_cols = feature_cols

        train_df = fm[fm.index.get_level_values("date").isin(train_dates)]
        test_df = fm[fm.index.get_level_values("date").isin(test_dates)]

        X_train = train_df[feature_cols].fillna(0.0).values
        y_train = train_df[_TARGET_COL].values
        X_test = test_df[feature_cols].fillna(0.0).values
        y_test = test_df[_TARGET_COL].values

        log.info(
            "linear_signal: training Ridge",
            n_train=len(X_train),
            n_test=len(X_test),
            features=len(feature_cols),
            alpha=alpha,
        )

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        self._model = model

        train_ic_val = _spearman_corr(model.predict(X_train), y_train)
        test_ic_val = _spearman_corr(model.predict(X_test), y_test)

        log.info(
            "linear_signal: training complete",
            train_ic=round(train_ic_val, 4),
            test_ic=round(test_ic_val, 4),
        )

        # Persist checkpoint
        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("linear_signal: checkpoint saved", path=self._model_path)

        self._write_metadata(
            n_tickers=len(tickers),
            period=period,
            train_ic=train_ic_val,
            test_ic=test_ic_val,
        )

        return {
            "train_ic": float(train_ic_val),
            "test_ic": float(test_ic_val),
            "train_icir": float(train_ic_val),
            "test_icir": float(test_ic_val),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, tickers: list[str], period: str = "6mo") -> dict[str, float]:
        """
        Return ranked alpha scores in [-1, 1] for each ticker.

        Uses the most recent date's rows from the feature matrix.
        Falls back to momentum composite when no model is available.
        """
        if self._model is None:
            return self._momentum_fallback(tickers, period)

        try:
            fm = build_feature_matrix(tickers, period=period)
            if fm.empty:
                return self._momentum_fallback(tickers, period)

            feature_cols = [c for c in self._feature_cols if c in fm.columns]

            last_date = fm.index.get_level_values("date").max()
            latest = fm.xs(last_date, level="date")[feature_cols].fillna(0.0)

            if latest.empty:
                return self._momentum_fallback(tickers, period)

            raw_preds = self._model.predict(latest.values)

            std = raw_preds.std()
            if std == 0:
                scores = np.zeros(len(raw_preds))
            else:
                scores = (raw_preds - raw_preds.mean()) / std

            scores = np.clip(scores, -1.0, 1.0)
            return {ticker: float(scores[i]) for i, ticker in enumerate(latest.index)}

        except Exception as exc:
            log.warning("linear_signal.predict: error, falling back to momentum", error=str(exc))
            return self._momentum_fallback(tickers, period)

    @staticmethod
    def _momentum_fallback(tickers: list[str], period: str) -> dict[str, float]:
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

    # ── Feature coefficients ──────────────────────────────────────────────────

    def feature_coefficients(self) -> pd.DataFrame:
        """
        Return a DataFrame of Ridge coefficients sorted by absolute magnitude.

        Returns
        -------
        pd.DataFrame with columns: feature, coefficient
        Empty DataFrame if no model is loaded.
        """
        if self._model is None:
            return pd.DataFrame(columns=["feature", "coefficient"])

        try:
            coefs = self._model.coef_
            df = pd.DataFrame({
                "feature": self._feature_cols[:len(coefs)],
                "coefficient": coefs,
            })
            return df.reindex(df["coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        except Exception as exc:
            log.warning("linear_signal.feature_coefficients: error", error=str(exc))
            return pd.DataFrame(columns=["feature", "coefficient"])

    # ── Metadata ──────────────────────────────────────────────────────────────

    def _write_metadata(
        self,
        n_tickers: int,
        period: str,
        train_ic: float,
        test_ic: float,
    ) -> None:
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
                    ("ridge_alpha", time.time(), train_ic, test_ic, n_tickers, period),
                )
            conn.close()
        except Exception as exc:
            log.warning("linear_signal: could not write metadata", error=str(exc))
