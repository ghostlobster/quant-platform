"""
strategies/rf_long_short.py — Random-Forest cross-sectional long-short alpha.

Implements the long-short trading strategy from Jansen, *Machine Learning
for Algorithmic Trading* (2nd ed.) Chapter 11: predict forward returns
with a RandomForestRegressor over the same engineered feature matrix
used by ml_signal/linear_signal, then form a portfolio that goes long
the top-quantile names and short the bottom-quantile names.

Same public surface as MLSignal / LinearSignal: ``train()``, ``predict()``,
plus an extra :meth:`long_short_portfolio` helper that turns the
per-ticker alpha scores into ±1 / 0 cross-sectional weights.

ENV vars
--------
    RF_LS_MODEL_PATH      path to model pickle (default: models/rf_long_short.pkl)
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
    from sklearn.ensemble import RandomForestRegressor
    _SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestRegressor = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

_DEFAULT_MODEL_PATH = os.environ.get("RF_LS_MODEL_PATH", "models/rf_long_short.pkl")
_TARGET_COL = "fwd_ret_5d"


class RFLongShortSignal:
    """Random-Forest regressor + cross-sectional long-short portfolio builder."""

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._model = None  # RandomForestRegressor | None
        self._feature_cols: list[str] = list(_FEATURE_COLS)
        self._load_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            log.info("rf_long_short: no checkpoint found", path=str(path))
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("rf_long_short: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning("rf_long_short: failed to load checkpoint", path=str(path), error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        n_estimators: int = 200,
        max_depth: int | None = 6,
        random_state: int = 42,
    ) -> dict[str, float]:
        """Fit a RandomForestRegressor on the cross-sectional feature matrix.

        Parameters mirror :class:`strategies.linear_signal.LinearSignal.train`
        except for the RF-specific ``n_estimators``, ``max_depth``,
        ``random_state``. Returns a dict with train/test IC + ICIR + sample
        counts.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"
            )

        log.info("rf_long_short: building feature matrix", tickers=len(tickers), period=period)
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
            "rf_long_short: training RandomForest",
            n_train=len(X_train), n_test=len(X_test),
            features=len(feature_cols), n_estimators=n_estimators, max_depth=max_depth,
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self._model = model

        train_ic = _spearman_corr(model.predict(X_train), y_train)
        test_ic = _spearman_corr(model.predict(X_test), y_test)

        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(
            "rf_long_short: training complete",
            train_ic=round(train_ic, 4), test_ic=round(test_ic, 4), path=self._model_path,
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
        """Per-ticker alpha scores in ``[-1, 1]`` for the latest date."""
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
            log.warning("rf_long_short.predict: error, falling back to momentum", error=str(exc))
            return self._momentum_fallback(tickers, period)

    # ── Long-short portfolio (Ch 11 specialty) ────────────────────────────────

    def long_short_portfolio(
        self,
        tickers: list[str],
        period: str = "6mo",
        top_pct: float = 0.2,
    ) -> dict[str, int]:
        """Return ``+1`` (long), ``-1`` (short), ``0`` (flat) per ticker.

        Cross-sectional dollar-neutral construction: long the top
        ``top_pct`` of tickers by predicted score, short the bottom
        ``top_pct``, flat on the rest. ``top_pct`` is clipped to
        ``(0, 0.5]``.
        """
        top_pct = max(min(float(top_pct), 0.5), 1e-6)
        scores = self.predict(tickers, period=period)
        if not scores:
            return {t: 0 for t in tickers}

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        n = len(ordered)
        k = max(1, int(round(n * top_pct)))

        weights: dict[str, int] = {t: 0 for t in scores}
        for ticker, _ in ordered[:k]:
            weights[ticker] = 1
        for ticker, _ in ordered[-k:]:
            # Avoid double-assigning a single-ticker universe to both legs
            if weights[ticker] == 0:
                weights[ticker] = -1
        return weights

    # ── Fallback & metadata (unchanged from sibling strategies) ───────────────

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

    def feature_importances(self) -> pd.DataFrame:
        """RF feature importances sorted descending (empty if no model)."""
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        try:
            imp = self._model.feature_importances_
            df = pd.DataFrame({
                "feature": self._feature_cols[:len(imp)],
                "importance": imp,
            })
            return df.sort_values("importance", ascending=False).reset_index(drop=True)
        except Exception as exc:
            log.warning("rf_long_short.feature_importances: error", error=str(exc))
            return pd.DataFrame(columns=["feature", "importance"])

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
                    ("rf_long_short", time.time(), train_ic, test_ic, n_tickers, period),
                )
            conn.close()
        except Exception as exc:
            log.warning("rf_long_short: could not write metadata", error=str(exc))
