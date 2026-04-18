"""
strategies/mlp_signal.py — Feed-forward MLP alpha (Jansen ML4T Ch 17).

Mirror of :mod:`strategies.linear_signal` but with ``MLPRegressor``
instead of Ridge.  Same train/predict surface so the ensemble blender,
the ml_signals UI, and the daily_ml_execute cron path can pick it up
without any other changes.

Backed by ``sklearn.neural_network.MLPRegressor`` (already a dep);
torch is reserved for the deeper Ch 18 / Ch 19 / Ch 20 / Ch 21 modules
(``cnn_signal.py``, ``dl_signal.py``, ``risk_autoencoder.py``,
``synthetic_paths.py``).

ENV vars
--------
    MLP_ALPHA_MODEL_PATH    path to model pickle (default: models/mlp_alpha.pkl)
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

try:
    from sklearn.neural_network import MLPRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:
    MLPRegressor = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False


_DEFAULT_MODEL_PATH = os.environ.get("MLP_ALPHA_MODEL_PATH", "models/mlp_alpha.pkl")
_TARGET_COL = "fwd_ret_5d"


class MLPSignal:
    """Feed-forward MLP cross-sectional alpha model."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._model = None  # MLPRegressor | None
        self._feature_cols: list[str] = list(_FEATURE_COLS)
        self._load_if_available()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            log.info("mlp_signal: no checkpoint found", path=str(path))
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("mlp_signal: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning("mlp_signal: failed to load checkpoint", path=str(path), error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        alpha: float = 1e-3,
        max_iter: int = 200,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Fit MLPRegressor on the cross-sectional feature matrix.

        Returns the standard
        ``{train_ic, test_ic, train_icir, test_icir,
            n_train_samples, n_test_samples}`` schema.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"
            )

        log.info("mlp_signal: building feature matrix", tickers=len(tickers), period=period)
        fm = build_feature_matrix(tickers, period=period)
        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")

        fm = fm.dropna(subset=[_TARGET_COL])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{_TARGET_COL}'")

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
            "mlp_signal: training MLPRegressor",
            n_train=len(X_train), n_test=len(X_test),
            features=len(feature_cols),
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha, max_iter=max_iter,
        )

        model = MLPRegressor(
            hidden_layer_sizes=tuple(int(h) for h in hidden_layer_sizes),
            alpha=float(alpha),
            max_iter=int(max_iter),
            random_state=int(random_state),
            early_stopping=False,
        )
        model.fit(X_train, y_train)
        self._model = model

        train_ic = _spearman_corr(model.predict(X_train), y_train)
        test_ic = _spearman_corr(model.predict(X_test), y_test)

        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)

        log.info(
            "mlp_signal: training complete",
            train_ic=round(train_ic, 4), test_ic=round(test_ic, 4),
            path=self._model_path,
        )

        self._write_metadata(
            n_tickers=len(tickers), period=period,
            train_ic=train_ic, test_ic=test_ic,
        )

        return {
            "train_ic": float(train_ic),
            "test_ic": float(test_ic),
            "train_icir": float(train_ic),
            "test_icir": float(test_ic),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, tickers: list[str], period: str = "6mo") -> dict[str, float]:
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

            raw = self._model.predict(latest.values)
            std = raw.std()
            scores = np.zeros_like(raw) if std == 0 else (raw - raw.mean()) / std
            scores = np.clip(scores, -1.0, 1.0)
            return {ticker: float(scores[i]) for i, ticker in enumerate(latest.index)}
        except Exception as exc:
            log.warning("mlp_signal.predict: error, falling back to momentum", error=str(exc))
            return self._momentum_fallback(tickers, period)

    # ── Feature importances (|first-layer weights| per feature) ───────────────

    def feature_importances(self) -> pd.DataFrame:
        """Per-feature importance proxy from the first layer's |weight| sum.

        ``MLPRegressor`` exposes ``coefs_[0]`` of shape
        ``(n_features, hidden_layer_sizes[0])``; row-wise L1 norms give a
        cheap interpretable importance signal.
        """
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        try:
            first_layer = self._model.coefs_[0]
            importances = np.abs(first_layer).sum(axis=1)
            df = pd.DataFrame({
                "feature": self._feature_cols[: len(importances)],
                "importance": importances,
            })
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception as exc:
            log.warning("mlp_signal.feature_importances: error", error=str(exc))
            return pd.DataFrame(columns=["feature", "importance"])

    # ── Fallback & metadata ───────────────────────────────────────────────────

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
                    ("mlp_alpha", time.time(), train_ic, test_ic, n_tickers, period),
                )
            conn.close()
        except Exception as exc:
            log.warning("mlp_signal: could not write metadata", error=str(exc))
