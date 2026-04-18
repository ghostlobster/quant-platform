"""
strategies/ml_signal.py — LightGBM alpha model for cross-sectional return prediction.

Trains a gradient-boosting regressor to predict 5-day forward returns across
a ticker universe.  Produces a ranked signal in [-1, 1] per ticker, suitable
for long/short or momentum-overlay strategies.

Falls back to a momentum-composite score when lightgbm is not installed or
no trained model checkpoint is present.

Threat model (pickle path confinement)
--------------------------------------
The paths read from ``LGBM_ALPHA_MODEL_PATH`` / ``LGBM_REGIME_MODELS_PATH``
feed ``pickle.load``, which executes arbitrary code during deserialisation.
Environment variables are therefore treated as **operator-trusted** — they
should only be set by the human who deploys the platform. As
defence-in-depth every load site runs the candidate path through
``agents.knowledge_agent._confine_pickle_path`` and refuses any path that
escapes the repo root or system temp dir.

ENV vars
--------
    LGBM_ALPHA_MODEL_PATH       path to baseline model pickle
                                (default: models/lgbm_alpha.pkl)
    LGBM_REGIME_MODELS_PATH     path to regime-conditioned models dict pickle
                                (default: models/lgbm_regime_models.pkl)

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

from agents.knowledge_agent import _confine_pickle_path
from analysis.regime import get_live_regime
from data.features import _FEATURE_COLS, build_feature_matrix
from data.fetcher import fetch_ohlcv
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
_DEFAULT_REGIME_MODELS_PATH = os.environ.get(
    "LGBM_REGIME_MODELS_PATH", "models/lgbm_regime_models.pkl"
)
_TARGET_COL = "fwd_ret_5d"
_TB_TARGET_COL = "tb_bin"
_TB_RET_COL = "tb_ret"
# Minimum (date, ticker) rows per regime required to train a regime-specific model
_MIN_REGIME_SAMPLES = 100

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
    LightGBM cross-sectional alpha model with optional regime conditioning.

    Maintains two model tiers:
      1. Baseline model  (_model)          — trained on all regimes combined
      2. Regime models   (_regime_models)  — one model per market regime,
                                             trained only on dates matching
                                             that regime's market state

    predict() automatically selects the regime-specific model for the current
    market state (via analysis.regime.get_live_regime), falling back to the
    baseline model, then to momentum composite scores.

    Attributes
    ----------
    _model             : fitted LGBMRegressor or None  (baseline)
    _regime_models     : dict[str, LGBMRegressor]      (regime-conditioned)
    _model_path        : filesystem path for baseline checkpoint
    _regime_model_path : filesystem path for regime models dict checkpoint
    """

    def __init__(
        self,
        model_path: str | None = None,
        regime_model_path: str | None = None,
    ) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._regime_model_path: str = regime_model_path or _DEFAULT_REGIME_MODELS_PATH
        self._model = None
        self._is_classifier: bool = False
        self._regime_models: dict = {}
        self._load_if_available()
        self._load_regime_models_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        """Load a persisted baseline model checkpoint if it exists."""
        try:
            path = _confine_pickle_path(self._model_path)
        except ValueError as exc:
            log.error(
                "ml_signal: refusing unsafe baseline path",
                path=str(self._model_path), error=str(exc),
            )
            return
        if not path.exists():
            log.info("ml_signal: no baseline checkpoint found, will use fallback", path=str(path))
            return
        try:
            if not _LGBM_AVAILABLE:
                log.info("ml_signal: lightgbm not installed, using momentum fallback")
                return
            with open(path, "rb") as f:
                payload = pickle.load(f)
            # Newer checkpoints are {"model": model, "is_classifier": bool}.
            # Older checkpoints are the raw estimator — treat as regressor.
            if isinstance(payload, dict) and "model" in payload:
                self._model = payload["model"]
                self._is_classifier = bool(payload.get("is_classifier", False))
            else:
                self._model = payload
                self._is_classifier = False
            log.info(
                "ml_signal: loaded baseline model checkpoint",
                path=str(path),
                classifier=self._is_classifier,
            )
        except Exception as exc:
            log.warning("ml_signal: failed to load baseline checkpoint", path=str(path), error=str(exc))

    def _load_regime_models_if_available(self) -> None:
        """Load persisted regime-conditioned model checkpoints if they exist."""
        try:
            path = _confine_pickle_path(self._regime_model_path)
        except ValueError as exc:
            log.error(
                "ml_signal: refusing unsafe regime-models path",
                path=str(self._regime_model_path), error=str(exc),
            )
            return
        if not path.exists():
            return
        try:
            if not _LGBM_AVAILABLE:
                return
            with open(path, "rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and "models" in payload:
                self._regime_models = payload["models"]
                # Preserve classifier flag if baseline hasn't set it.
                self._is_classifier = self._is_classifier or bool(
                    payload.get("is_classifier", False)
                )
            else:
                # Legacy format: bare dict of {regime: model}
                self._regime_models = payload
            log.info(
                "ml_signal: loaded regime models",
                path=str(path),
                regimes=list(self._regime_models.keys()),
            )
        except Exception as exc:
            log.warning("ml_signal: failed to load regime models", path=str(path), error=str(exc))

    # ── Training — baseline ───────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        lgbm_params: dict | None = None,
        label_type: str = "fwd_ret",
        pt_sl: tuple[float, float] = (1.0, 1.0),
        num_days: int = 5,
        use_sample_weights: bool = False,
    ) -> dict[str, float]:
        """
        Build feature matrix, train baseline LightGBM, evaluate on held-out
        test split, persist the model, and write metadata to quant.db.

        Parameters
        ----------
        tickers    : universe of tickers to train on
        period     : yfinance period string for data fetching
        test_size  : fraction of time series reserved for out-of-sample eval
                     (chronological split — latest test_size of dates)
        lgbm_params : override default LGBMRegressor / LGBMClassifier params
        label_type : "fwd_ret" (regressor on fwd_ret_5d) or "triple_barrier"
                     (classifier on the tb_bin {-1, 0, +1} label)
        pt_sl      : (profit-take, stop-loss) multipliers; triple-barrier only
        num_days   : vertical-barrier horizon in days; triple-barrier only
        use_sample_weights :
                     when True and label_type=="triple_barrier", compute
                     per-event uniqueness weights (López de Prado Ch 4)
                     from ``tb_t1`` and forward them to LightGBM's
                     ``fit(sample_weight=...)``.  Silently ignored for
                     the regressor ``fwd_ret`` path.

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

        is_classifier = label_type == "triple_barrier"
        target_col = _TB_TARGET_COL if is_classifier else _TARGET_COL

        log.info(
            "ml_signal: building feature matrix",
            tickers=len(tickers), period=period, label_type=label_type,
        )
        fm = build_feature_matrix(
            tickers, period=period,
            label_type=label_type, pt_sl=pt_sl, num_days=num_days,
        )

        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")

        # Drop rows with missing target
        fm = fm.dropna(subset=[target_col])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{target_col}'")

        # Chronological train/test split on unique dates
        all_dates = sorted(fm.index.get_level_values("date").unique())
        split_idx = int(len(all_dates) * (1 - test_size))
        train_dates = set(all_dates[:split_idx])
        test_dates = set(all_dates[split_idx:])

        feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]

        train_df = fm[fm.index.get_level_values("date").isin(train_dates)]
        test_df = fm[fm.index.get_level_values("date").isin(test_dates)]

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values
        if is_classifier:
            y_train = train_df[target_col].astype(int).values
            y_test = test_df[target_col].astype(int).values
            # Evaluate IC against the realised return, not the {-1,0,+1} label
            ic_eval_train = train_df[_TB_RET_COL].astype(float).values
            ic_eval_test = test_df[_TB_RET_COL].astype(float).values
        else:
            y_train = train_df[target_col].values
            y_test = test_df[target_col].values
            ic_eval_train = y_train
            ic_eval_test = y_test

        log.info(
            "ml_signal: training",
            n_train=len(X_train),
            n_test=len(X_test),
            features=len(feature_cols),
            classifier=is_classifier,
        )

        params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        if is_classifier:
            model = lgb.LGBMClassifier(**params)
        else:
            model = lgb.LGBMRegressor(**params)

        fit_kwargs: dict = {}
        if use_sample_weights and is_classifier and "tb_t1" in train_df.columns:
            from analysis.sample_weights import weights_for_train_index

            events = train_df["tb_t1"].dropna()
            # Collapse MultiIndex to date-only event series (one entry per
            # event start date; duplicate dates share the same t1).
            events_by_date = (
                events.reset_index()
                .drop_duplicates("date")
                .set_index("date")["tb_t1"]
            )
            close_idx = pd.DatetimeIndex(
                sorted(fm.index.get_level_values("date").unique())
            )
            sw = weights_for_train_index(train_df.index, events_by_date, close_idx)
            if sw.shape == (len(X_train),):
                fit_kwargs["sample_weight"] = sw
                log.info(
                    "ml_signal: training with AFML Ch 4 sample weights",
                    mean=float(sw.mean()), min=float(sw.min()), max=float(sw.max()),
                )

        model.fit(X_train, y_train, **fit_kwargs)
        self._model = model
        self._is_classifier = is_classifier

        # Evaluate IC on train and test sets (against realised return)
        train_ic, train_icir = self._eval_ic(
            model, X_train, ic_eval_train, classifier=is_classifier,
        )
        test_ic, test_icir = self._eval_ic(
            model, X_test, ic_eval_test, classifier=is_classifier,
        )

        log.info(
            "ml_signal: training complete",
            train_ic=round(train_ic, 4),
            test_ic=round(test_ic, 4),
        )

        # Persist checkpoint (versioned payload preserves is_classifier flag)
        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._model_path, "wb") as f:
            pickle.dump({"model": model, "is_classifier": is_classifier}, f)
        log.info("ml_signal: baseline checkpoint saved", path=self._model_path)

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

    # ── Training — regime-conditioned ─────────────────────────────────────────

    def train_regime_models(
        self,
        tickers: list[str],
        period: str = "2y",
        min_regime_samples: int = _MIN_REGIME_SAMPLES,
        test_size: float = 0.2,
        lgbm_params: dict | None = None,
        label_type: str = "fwd_ret",
        pt_sl: tuple[float, float] = (1.0, 1.0),
        num_days: int = 5,
        use_sample_weights: bool = False,
    ) -> dict[str, dict]:
        """
        Train one LGBMRegressor per market regime.

        Fetches historical SPY + ^VIX data to label each date in the feature
        matrix with its market regime, then trains a separate model on each
        regime's subset.  Only regimes with >= min_regime_samples rows are
        trained; the rest are silently skipped.

        Parameters
        ----------
        tickers            : universe of tickers
        period             : yfinance period string
        min_regime_samples : minimum (date, ticker) rows per regime (default 100)
        test_size          : chronological held-out fraction per regime
        lgbm_params        : LGBMRegressor parameter overrides

        Returns
        -------
        dict mapping regime_label → {train_ic, test_ic, train_icir, test_icir,
                                      n_train, n_test}
        Only regimes that were trained are included.

        Raises
        ------
        RuntimeError if lightgbm is not installed
        ValueError   if feature matrix is empty or no usable target rows
        """
        if not _LGBM_AVAILABLE:
            raise RuntimeError("lightgbm is not installed.")

        is_classifier = label_type == "triple_barrier"
        target_col = _TB_TARGET_COL if is_classifier else _TARGET_COL

        log.info(
            "ml_signal: building feature matrix for regime models",
            tickers=len(tickers),
            period=period,
            label_type=label_type,
        )
        fm = build_feature_matrix(
            tickers, period=period,
            label_type=label_type, pt_sl=pt_sl, num_days=num_days,
        )

        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")

        fm = fm.dropna(subset=[target_col])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{target_col}'")

        # Label each row with its market regime
        regime_series = self._get_historical_regimes(period)
        if regime_series.empty:
            raise ValueError("Could not determine historical regimes — SPY data unavailable")

        dates = fm.index.get_level_values("date")
        regime_labels = regime_series.reindex(dates).fillna("mean_reverting").values

        feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
        params = {**_DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}

        from analysis.regime import REGIME_STATES

        results: dict[str, dict] = {}
        regime_models: dict = {}

        for regime in REGIME_STATES:
            mask = regime_labels == regime
            regime_fm = fm[mask]

            if len(regime_fm) < min_regime_samples:
                log.info(
                    "ml_signal: skipping regime — insufficient samples",
                    regime=regime,
                    n=int(len(regime_fm)),
                    required=min_regime_samples,
                )
                continue

            # Chronological split within this regime's data
            all_dates = sorted(regime_fm.index.get_level_values("date").unique())
            split_idx = int(len(all_dates) * (1 - test_size))
            train_dates = set(all_dates[:split_idx])
            test_dates = set(all_dates[split_idx:])

            train_df = regime_fm[regime_fm.index.get_level_values("date").isin(train_dates)]
            test_df = regime_fm[regime_fm.index.get_level_values("date").isin(test_dates)]

            X_train = train_df[feature_cols].values
            X_test = test_df[feature_cols].values
            if is_classifier:
                y_train = train_df[target_col].astype(int).values
                y_test = test_df[target_col].astype(int).values
                ic_eval_train = train_df[_TB_RET_COL].astype(float).values
                ic_eval_test = test_df[_TB_RET_COL].astype(float).values
                model = lgb.LGBMClassifier(**params)
            else:
                y_train = train_df[target_col].values
                y_test = test_df[target_col].values
                ic_eval_train = y_train
                ic_eval_test = y_test
                model = lgb.LGBMRegressor(**params)

            fit_kwargs: dict = {}
            if use_sample_weights and is_classifier and "tb_t1" in train_df.columns:
                from analysis.sample_weights import weights_for_train_index

                events = train_df["tb_t1"].dropna()
                events_by_date = (
                    events.reset_index()
                    .drop_duplicates("date")
                    .set_index("date")["tb_t1"]
                )
                close_idx = pd.DatetimeIndex(
                    sorted(regime_fm.index.get_level_values("date").unique())
                )
                sw = weights_for_train_index(
                    train_df.index, events_by_date, close_idx,
                )
                if sw.shape == (len(X_train),):
                    fit_kwargs["sample_weight"] = sw

            model.fit(X_train, y_train, **fit_kwargs)

            train_ic, train_icir = self._eval_ic(
                model, X_train, ic_eval_train, classifier=is_classifier,
            )
            test_ic, test_icir = self._eval_ic(
                model, X_test, ic_eval_test, classifier=is_classifier,
            )

            regime_models[regime] = model
            results[regime] = {
                "train_ic": float(train_ic),
                "test_ic": float(test_ic),
                "train_icir": float(train_icir),
                "test_icir": float(test_icir),
                "n_train": int(len(X_train)),
                "n_test": int(len(X_test)),
            }
            log.info(
                "ml_signal: regime model trained",
                regime=regime,
                train_ic=round(train_ic, 4),
                test_ic=round(test_ic, 4),
            )

        self._regime_models = regime_models
        self._is_classifier = is_classifier
        Path(self._regime_model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self._regime_model_path, "wb") as f:
            pickle.dump(
                {"models": regime_models, "is_classifier": is_classifier}, f,
            )
        log.info(
            "ml_signal: regime models saved",
            path=self._regime_model_path,
            regimes=list(regime_models.keys()),
        )

        return results

    def _get_historical_regimes(self, period: str) -> pd.Series:
        """
        Return a date-indexed Series of regime labels for the given period.

        Uses SPY 200-day rolling SMA and ^VIX closing level, vectorised.
        Falls back to 'mean_reverting' for any date with missing VIX data.
        """
        spy_df = fetch_ohlcv("SPY", period)
        vix_df = fetch_ohlcv("^VIX", period)

        if spy_df is None or spy_df.empty:
            log.warning("ml_signal: could not fetch SPY for regime labels")
            return pd.Series(dtype=str)

        spy_close = spy_df["Close"].astype(float)
        sma200 = spy_close.rolling(200, min_periods=1).mean()

        if vix_df is not None and not vix_df.empty:
            vix_aligned = (
                vix_df["Close"].astype(float)
                .reindex(spy_close.index, method="ffill")
                .fillna(20.0)
            )
        else:
            vix_aligned = pd.Series(20.0, index=spy_close.index)

        # Priority: VIX > 30 → high_vol; VIX 20–30 → mean_reverting;
        #           VIX < 20 + SPY > SMA200 → trending_bull; else → trending_bear
        regimes = np.select(
            [vix_aligned > 30, vix_aligned >= 20, spy_close > sma200],
            ["high_vol", "mean_reverting", "trending_bull"],
            default="trending_bear",
        )
        return pd.Series(regimes, index=spy_close.index, name="regime")

    # ── Shared eval & metadata helpers ────────────────────────────────────────

    @staticmethod
    def _classifier_scores(model, X: np.ndarray) -> np.ndarray:
        """Map a {-1, 0, +1} classifier's predict_proba → continuous [-1, 1].

        Score = P(+1) - P(-1), which is monotone in expected directional bet
        and lives naturally in the same band as the regressor output.
        """
        proba = model.predict_proba(X)
        classes = np.asarray(model.classes_)
        p_up = proba[:, classes == 1].sum(axis=1) if (classes == 1).any() else np.zeros(len(X))
        p_dn = proba[:, classes == -1].sum(axis=1) if (classes == -1).any() else np.zeros(len(X))
        return np.asarray(p_up - p_dn, dtype=float)

    @classmethod
    def _eval_ic(
        cls, model, X: np.ndarray, y: np.ndarray, classifier: bool = False,
    ) -> tuple[float, float]:
        """Return (IC mean, ICIR) from a predictions array vs actuals."""
        if classifier:
            preds = cls._classifier_scores(model, X)
        else:
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
        """Persist training metadata to quant.db model_metadata table.

        On retrain #2 onward also computes ``test_ic_delta`` as
        ``this_test_ic - previous_test_ic`` for the same ``model_name`` so
        downstream consumers (``analysis/retrain_roi.py``,
        ``KnowledgeAdaptionAgent``) can detect IC plateaus.
        """
        try:
            from data.db import get_connection
            conn = get_connection()
            with conn:
                prev_row = conn.execute(
                    "SELECT test_ic FROM model_metadata "
                    "WHERE model_name = ? ORDER BY trained_at DESC LIMIT 1",
                    ("lgbm_alpha",),
                ).fetchone()
                prev_ic = None
                if prev_row is not None:
                    raw = prev_row["test_ic"] if hasattr(prev_row, "keys") else prev_row[0]
                    if raw is not None:
                        prev_ic = float(raw)
                delta = None if prev_ic is None else float(test_ic) - prev_ic
                conn.execute(
                    """
                    INSERT INTO model_metadata
                        (model_name, trained_at, train_ic, test_ic,
                         n_tickers, period, test_ic_delta)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "lgbm_alpha", time.time(), train_ic, test_ic,
                        n_tickers, period, delta,
                    ),
                )
            conn.close()
        except Exception as exc:
            log.warning("ml_signal: could not write metadata", error=str(exc))

    # ── Model selection ───────────────────────────────────────────────────────

    def _select_model(self, use_regime_model: bool = True):
        """
        Select the best available model for the current market state.

        Priority: regime-specific model > baseline model > None (→ fallback).
        """
        if use_regime_model and self._regime_models:
            try:
                regime_info = get_live_regime()
                current_regime = regime_info.get("regime")
                if current_regime and current_regime in self._regime_models:
                    log.info("ml_signal: using regime model", regime=current_regime)
                    return self._regime_models[current_regime]
            except Exception as exc:
                log.warning(
                    "ml_signal: regime detection failed, falling back to baseline",
                    error=str(exc),
                )
        return self._model

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        tickers: list[str],
        period: str = "6mo",
        use_regime_model: bool = True,
    ) -> dict[str, float]:
        """
        Return ranked alpha scores in [-1, 1] for each ticker.

        Uses the most recent date's rows from the feature matrix.
        Scores are z-scored raw predictions clipped to [-1, 1].

        Model selection priority: regime-specific > baseline > momentum fallback.

        Parameters
        ----------
        tickers          : universe to score
        period           : data window for feature computation
        use_regime_model : if True (default), prefer regime-conditioned model
                           for the current market regime when one is available

        Returns
        -------
        dict mapping ticker → score in [-1.0, 1.0]
        """
        active_model = self._select_model(use_regime_model)

        if active_model is None:
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

            if self._is_classifier and hasattr(active_model, "predict_proba"):
                raw_preds = self._classifier_scores(active_model, latest.values)
            else:
                raw_preds = active_model.predict(latest.values)

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

    # ── Scoring helper (used by backtest path) ────────────────────────────────

    def score_features(self, X: np.ndarray) -> np.ndarray:
        """
        Return raw continuous scores for a pre-built feature array.

        Dispatches to predict_proba-derived P(+1)-P(-1) for classifier
        checkpoints and to regressor.predict for regressor checkpoints.
        """
        if self._model is None:
            raise RuntimeError("ml_signal.score_features: no trained model loaded")
        if self._is_classifier and hasattr(self._model, "predict_proba"):
            return self._classifier_scores(self._model, X)
        return np.asarray(self._model.predict(X), dtype=float)

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
