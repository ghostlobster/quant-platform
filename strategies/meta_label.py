"""
strategies/meta_label.py — López de Prado meta-labeling wrapper.

Meta-labeling trains a secondary classifier on top of a primary signal to
predict *when the primary is likely to be correct*. The final output
combines primary direction with a confidence score:

    final_score = primary * P(correct | features)

This reduces false positives from any primary model (technical crossovers,
LightGBM, sentiment signals) without changing its directional view.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 3.3.

Optional deps
-------------
    scikit-learn >= 1.3.0  (for ``RandomForestClassifier``). When unavailable,
    ``MetaLabeler`` is still importable but its ``fit``/``predict`` raise a
    clear ``RuntimeError``.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
    _SKLEARN_AVAILABLE = True
except ImportError:
    RandomForestClassifier = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False


class MetaLabeler:
    """
    Secondary classifier that filters a primary trading signal.

    Parameters
    ----------
    n_estimators : number of trees in the RandomForest (default 200).
    random_state : RNG seed for reproducibility.

    Usage
    -----
        primary = pd.Series([+1, -1, +1, ...], index=dates)     # {-1, +1}
        bins    = pd.Series([+1, -1, 0, ...], index=dates)      # triple-barrier
        features = pd.DataFrame({...}, index=dates)

        labeler = MetaLabeler().fit(primary, bins, features)
        final = labeler.predict(primary, features)   # continuous [-1, 1]
    """

    def __init__(self, n_estimators: int = 200, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Optional["RandomForestClassifier"] = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        primary_signals: pd.Series,
        triple_barrier_bins: pd.Series,
        features: pd.DataFrame,
    ) -> dict:
        """Train the secondary classifier.

        A training sample is labeled ``1`` when the primary signal agreed with
        the realised outcome (``primary * bin > 0``) and ``0`` otherwise.
        Neutral bins (``bin == 0``) are dropped — they carry no information
        about whether the primary was correct.

        Returns
        -------
        dict with ``train_accuracy``, ``n_samples``, ``positive_rate`` keys.
        """
        if not _SKLEARN_AVAILABLE:
            raise RuntimeError(
                "scikit-learn is required for MetaLabeler; "
                "run `pip install scikit-learn>=1.3.0`"
            )

        idx = primary_signals.index.intersection(triple_barrier_bins.index).intersection(features.index)
        if len(idx) == 0:
            raise ValueError("meta_label: no common index between primary, bins, features")

        primary = primary_signals.reindex(idx).astype(float)
        bins = triple_barrier_bins.reindex(idx).astype(float)
        X = features.reindex(idx).fillna(0.0)

        # Drop neutral events (bin == 0) — they don't teach the secondary anything.
        keep = bins != 0
        primary, bins, X = primary[keep], bins[keep], X[keep]
        if len(X) == 0:
            raise ValueError("meta_label: all bins were neutral after filtering")

        y = (primary * bins > 0).astype(int)

        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(X.values, y.values)
        self._feature_cols = list(X.columns)

        train_acc = float(self._model.score(X.values, y.values))
        positive_rate = float(y.mean())
        metrics = {
            "train_accuracy": train_acc,
            "n_samples": int(len(y)),
            "positive_rate": positive_rate,
        }
        log.info("meta_label: trained", **metrics)
        return metrics

    def predict(
        self,
        primary_signals: pd.Series,
        features: pd.DataFrame,
    ) -> pd.Series:
        """Return the confidence-weighted final score: ``primary * P(correct)``.

        Scores are clipped to ``[-1, 1]``. When the model is not yet trained or
        sklearn is unavailable, the raw primary signal is returned unchanged —
        this keeps the pipeline functional in CI without optional deps.
        """
        if self._model is None or not _SKLEARN_AVAILABLE:
            return primary_signals.astype(float).clip(-1.0, 1.0)

        idx = primary_signals.index.intersection(features.index)
        primary = primary_signals.reindex(idx).astype(float)
        X = features.reindex(idx)[self._feature_cols].fillna(0.0)

        p_correct = self._model.predict_proba(X.values)[:, 1]
        final = (primary.values * p_correct).clip(-1.0, 1.0)
        return pd.Series(final, index=idx)

    def is_trained(self) -> bool:
        return self._model is not None


def apply_meta_labels(
    primary_signals: pd.Series,
    features: pd.DataFrame,
    triple_barrier_bins: Optional[pd.Series] = None,
) -> pd.Series:
    """Convenience: fit a meta-labeler in-place (if labels provided) and predict.

    When ``triple_barrier_bins`` is None, returns the clipped primary signal
    unchanged. Useful as a one-shot helper inside notebooks / CI pipelines.
    """
    if triple_barrier_bins is None:
        return primary_signals.astype(float).clip(-1.0, 1.0)

    labeler = MetaLabeler()
    try:
        labeler.fit(primary_signals, triple_barrier_bins, features)
    except (RuntimeError, ValueError) as exc:
        log.warning("meta_label: fit failed, returning primary as-is", error=str(exc))
        return primary_signals.astype(float).clip(-1.0, 1.0)

    return labeler.predict(primary_signals, features)


def filter_primary_by_confidence(
    primary_signals: pd.Series,
    final_scores: pd.Series,
    min_confidence: float = 0.5,
) -> pd.Series:
    """Zero out primary signals whose |final_score / primary| falls below threshold.

    Useful for converting a continuous meta-score back into a ``{-1, 0, +1}``
    hard-signal suitable for the event-driven backtester.
    """
    idx = primary_signals.index.intersection(final_scores.index)
    primary = primary_signals.reindex(idx).astype(float)
    final = final_scores.reindex(idx).astype(float)

    # Confidence = |final| / |primary|  (primary is ±1; so this is |final|).
    conf = final.abs()
    out = primary.copy()
    out[conf < min_confidence] = 0.0
    return out
