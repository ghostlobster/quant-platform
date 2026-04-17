"""
analysis/feature_importance.py — López de Prado Ch 8 feature importance.

SHAP values (already available through ``strategies.ml_signal``) tell you
which features the model *uses* on a given prediction.  They don't tell
you which features actually make the model more accurate.  AFML Ch 8
advocates two complementary measures:

  * **Mean Decrease in Accuracy (MDA)** — shuffle one feature's values
    and measure the drop in cross-validated test performance.  Captures
    both substitution effects and feature interactions.
  * **Mean Decrease in Impurity (MDI)** — sum of impurity decreases
    weighted by node sample sizes, averaged across every tree.  Cheap
    to compute because it's already accumulated during training.

Together they corroborate each other and reveal features that SHAP can
over-credit due to collinearity.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 8.3–8.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_Scorer = Callable[[np.ndarray, np.ndarray], float]


def _default_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (IC) — AFML's default financial scorer."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 3 or np.std(y_pred) == 0:
        return 0.0
    rho, _ = spearmanr(y_pred, y_true)
    return 0.0 if np.isnan(rho) else float(rho)


@dataclass
class FeatureImportanceResult:
    """Output of :func:`mda_importance` / :func:`mdi_importance`.

    Attributes
    ----------
    importance :
        Series indexed by feature name.  Higher = more important.
    std :
        Optional per-feature standard deviation across folds / trees.
        ``None`` when the estimator cannot supply one.
    """

    importance: pd.Series
    std: Optional[pd.Series] = None

    def top_k(self, k: int = 20) -> pd.Series:
        return self.importance.sort_values(ascending=False).head(k)


def mda_importance(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: Optional[Iterable[tuple[np.ndarray, np.ndarray]]] = None,
    scorer: _Scorer = _default_scorer,
    random_state: int = 42,
) -> FeatureImportanceResult:
    """Mean-Decrease-in-Accuracy feature importance via permutation.

    For every ``(train_idx, test_idx)`` split the model is fit on the
    training fold and scored on the test fold, then for each feature
    column the values are shuffled and the score re-computed.  The
    drop in score is the feature's importance for that fold.  Results
    are averaged across folds.

    Parameters
    ----------
    model :
        An sklearn-style estimator.  Must implement ``fit(X, y)`` and
        ``predict(X) -> np.ndarray``.  The model is **cloned** by a
        simple no-op (sklearn isn't imported here) — callers should
        pass in a factory-produced fresh instance if refitting matters
        for their estimator.
    X :
        Feature DataFrame.  Column names are preserved in the output.
    y :
        Target array (1-D).  Length must match ``len(X)``.
    cv_splits :
        Iterable of ``(train_idx, test_idx)`` positional arrays
        (compatible with :func:`backtester.combinatorial_cv`).  If
        ``None``, a single 70/30 chronological split is used.
    scorer :
        ``(y_true, y_pred) -> float``.  Defaults to Spearman IC.
    random_state :
        Seed for the permutation RNG.

    Returns
    -------
    :class:`FeatureImportanceResult` indexed by feature name.
    """
    if X is None or len(X) == 0:
        return FeatureImportanceResult(pd.Series(dtype=float))
    y = np.asarray(y)
    if len(y) != len(X):
        raise ValueError("mda_importance: X and y length mismatch")

    features = list(X.columns)
    if cv_splits is None:
        n = len(X)
        cut = int(n * 0.7)
        cv_splits = [(np.arange(cut), np.arange(cut, n))]
    cv_splits = list(cv_splits)
    if not cv_splits:
        return FeatureImportanceResult(pd.Series(0.0, index=features))

    rng = np.random.default_rng(random_state)
    per_fold_drops: list[pd.Series] = []

    for train_idx, test_idx in cv_splits:
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train.values, y_train)
        baseline = scorer(y_test, model.predict(X_test.values))

        drops = {}
        for col in features:
            saved = X_test[col].values.copy()
            X_test[col] = rng.permutation(saved)
            shuffled = scorer(y_test, model.predict(X_test.values))
            drops[col] = baseline - shuffled
            X_test[col] = saved            # restore for the next column

        per_fold_drops.append(pd.Series(drops, index=features))

    stacked = pd.concat(per_fold_drops, axis=1)
    mean = stacked.mean(axis=1)
    std = stacked.std(axis=1) if stacked.shape[1] > 1 else None
    return FeatureImportanceResult(mean.astype(float), std)


def mdi_importance(
    model,
    feature_names: Iterable[str],
) -> FeatureImportanceResult:
    """Mean-Decrease-in-Impurity feature importance.

    For tree ensembles, sklearn / LightGBM expose
    ``feature_importances_`` — the sum of (weighted) impurity
    decreases from every node, averaged across all trees.  This helper
    wraps that into a :class:`FeatureImportanceResult` with a
    per-feature std when ``estimators_`` is available (sklearn
    ensembles).  For LightGBM (no ``estimators_``), std is ``None``.

    Returns an empty result when the model exposes no importances.
    """
    feature_names = list(feature_names)
    if model is None or not hasattr(model, "feature_importances_"):
        return FeatureImportanceResult(pd.Series(dtype=float))

    importances = np.asarray(model.feature_importances_, dtype=float)
    if len(importances) != len(feature_names):
        # Feature arity mismatch: return whatever we can, padded / truncated.
        k = min(len(importances), len(feature_names))
        importances = importances[:k]
        feature_names = feature_names[:k]

    mean = pd.Series(importances, index=feature_names, dtype=float)

    std: Optional[pd.Series] = None
    estimators = getattr(model, "estimators_", None)
    if estimators:
        try:
            per_tree = np.stack(
                [getattr(est, "feature_importances_", np.zeros(len(feature_names)))
                 for est in estimators]
            )
            if per_tree.shape[1] == len(feature_names):
                std = pd.Series(
                    per_tree.std(axis=0),
                    index=feature_names,
                    dtype=float,
                )
        except (ValueError, AttributeError):
            std = None

    return FeatureImportanceResult(mean, std)
