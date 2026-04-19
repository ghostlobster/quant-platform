"""
strategies/bayesian_signal.py — Bayesian linear-regression alpha model.

The existing ``LinearSignal`` (Ridge) gives a point estimate; it can't
express parameter uncertainty.  Bayesian linear regression retains a
full posterior so we can report a per-sample predictive ``sigma`` — the
execution layer can then scale Kelly by confidence (shrink bets when
the posterior is wide) and the UI can surface error bars.

``BayesianSignal`` mirrors :class:`~strategies.linear_signal.LinearSignal`
in every public method so the ensemble / prediction pipelines pick it
up without special-casing.  The only extra surface is
:meth:`predict_with_uncertainty` which returns a ``(mean, sigma)`` dict
pair.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 10.3-10.5.
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
    from sklearn.linear_model import BayesianRidge
    _SKLEARN_AVAILABLE = True
except ImportError:
    BayesianRidge = None  # type: ignore[assignment,misc]
    _SKLEARN_AVAILABLE = False

_DEFAULT_MODEL_PATH = os.environ.get(
    "BAYES_ALPHA_MODEL_PATH", "models/bayesian_alpha.pkl"
)
_TARGET_COL = "fwd_ret_5d"

from agents.knowledge_registry import ModelEntry  # noqa: E402

MODEL_ENTRY = ModelEntry(
    name="bayesian_alpha",
    artefact_env="BAYES_ALPHA_MODEL_PATH",
    artefact_default="models/bayesian_alpha.pkl",
    metadata_name="bayesian_alpha",
)


class BayesianSignal:
    """Bayesian Ridge alpha model — same interface as :class:`LinearSignal`.

    Internally wraps ``sklearn.linear_model.BayesianRidge`` which gives
    an analytical posterior over the coefficients and exposes a
    per-sample predictive std.  The posterior std is the quantity the
    execution layer can consume to shrink Kelly sizes.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._model = None  # BayesianRidge | None
        self._feature_cols: list[str] = list(_FEATURE_COLS)
        self._load_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            log.info(
                "bayesian_signal: no checkpoint found, will use momentum fallback",
                path=str(path),
            )
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("bayesian_signal: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning(
                "bayesian_signal: failed to load checkpoint",
                path=str(path), error=str(exc),
            )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
    ) -> dict[str, float]:
        """Fit BayesianRidge on the feature matrix and persist the model.

        Same return schema as :meth:`LinearSignal.train` — the caller
        code in the ensemble / UI layers can treat both classes
        interchangeably.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is not installed. Run: pip install scikit-learn>=1.3.0"
            )

        log.info(
            "bayesian_signal: building feature matrix",
            tickers=len(tickers), period=period,
        )
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
            "bayesian_signal: training BayesianRidge",
            n_train=len(X_train),
            n_test=len(X_test),
            features=len(feature_cols),
        )

        model = BayesianRidge()
        model.fit(X_train, y_train)
        self._model = model

        train_ic = _spearman_corr(model.predict(X_train), y_train)
        test_ic = _spearman_corr(model.predict(X_test), y_test)

        log.info(
            "bayesian_signal: training complete",
            train_ic=round(train_ic, 4),
            test_ic=round(test_ic, 4),
        )

        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("bayesian_signal: checkpoint saved", path=self._model_path)

        self._write_metadata(
            n_tickers=len(tickers),
            period=period,
            train_ic=train_ic,
            test_ic=test_ic,
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

    def predict(
        self, tickers: list[str], period: str = "6mo",
    ) -> dict[str, float]:
        """Return ranked mean alpha scores in ``[-1, 1]`` per ticker.

        The per-sample ``sigma`` is discarded here — callers who want it
        should call :meth:`predict_with_uncertainty` instead.
        """
        mean, _ = self.predict_with_uncertainty(tickers, period=period)
        return mean

    def predict_with_uncertainty(
        self, tickers: list[str], period: str = "6mo",
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return ``(mean_scores, sigma_scores)`` dicts keyed by ticker.

        Mean scores are z-scored + clipped to ``[-1, 1]`` exactly like
        :meth:`LinearSignal.predict`.  Sigma is the raw BayesianRidge
        predictive std — smaller ⇒ more confident.  When no model is
        loaded, falls back to the momentum composite and reports
        sigma = 1.0 for every ticker (max uncertainty).
        """
        fallback_sigma = {t: 1.0 for t in tickers}
        if self._model is None:
            return self._momentum_fallback(tickers, period), fallback_sigma

        try:
            fm = build_feature_matrix(tickers, period=period)
            if fm.empty:
                return self._momentum_fallback(tickers, period), fallback_sigma

            feature_cols = [c for c in self._feature_cols if c in fm.columns]
            last_date = fm.index.get_level_values("date").max()
            latest = fm.xs(last_date, level="date")[feature_cols].fillna(0.0)

            if latest.empty:
                return self._momentum_fallback(tickers, period), fallback_sigma

            raw_mean, raw_std = self._model.predict(latest.values, return_std=True)

            std = raw_mean.std()
            if std == 0:
                scores = np.zeros(len(raw_mean))
            else:
                scores = (raw_mean - raw_mean.mean()) / std
            scores = np.clip(scores, -1.0, 1.0)

            mean_dict = {
                ticker: float(scores[i]) for i, ticker in enumerate(latest.index)
            }
            sigma_dict = {
                ticker: float(raw_std[i]) for i, ticker in enumerate(latest.index)
            }
            return mean_dict, sigma_dict

        except Exception as exc:
            log.warning(
                "bayesian_signal.predict: error, falling back to momentum",
                error=str(exc),
            )
            return self._momentum_fallback(tickers, period), fallback_sigma

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
                    last_val = (
                        mom.dropna().iloc[-1] if not mom.dropna().empty else 0.0
                    )
                    scores[ticker] = float(np.clip(last_val, -1.0, 1.0))
                else:
                    scores[ticker] = 0.0
            except Exception:
                scores[ticker] = 0.0
        return scores

    # ── Coefficients ─────────────────────────────────────────────────────────

    def feature_coefficients(self) -> pd.DataFrame:
        """Return a DataFrame of posterior-mean coefficients with std.

        Columns: ``feature``, ``coefficient``, ``std``.  Empty if no
        model is loaded.  The ``std`` column is the *posterior* std of
        each coefficient (sqrt of the diagonal of ``sigma_``) — a
        direct indicator of which factors the model believes in.
        """
        if self._model is None:
            return pd.DataFrame(columns=["feature", "coefficient", "std"])
        try:
            coefs = np.asarray(self._model.coef_)
            # ``sigma_`` is the posterior covariance matrix of the coefs
            sigma = getattr(self._model, "sigma_", None)
            if sigma is not None and sigma.shape[0] == len(coefs):
                coef_std = np.sqrt(np.maximum(np.diag(sigma), 0.0))
            else:
                coef_std = np.zeros_like(coefs)
            df = pd.DataFrame({
                "feature": self._feature_cols[:len(coefs)],
                "coefficient": coefs,
                "std": coef_std,
            })
            return df.reindex(
                df["coefficient"].abs().sort_values(ascending=False).index
            ).reset_index(drop=True)
        except Exception as exc:
            log.warning("bayesian_signal.feature_coefficients: error", error=str(exc))
            return pd.DataFrame(columns=["feature", "coefficient", "std"])

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
                    (
                        "bayesian_alpha", time.time(), train_ic, test_ic,
                        n_tickers, period,
                    ),
                )
            conn.close()
        except Exception as exc:
            log.warning(
                "bayesian_signal: could not write metadata", error=str(exc),
            )
