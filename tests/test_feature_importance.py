"""Tests for analysis/feature_importance.py — AFML Ch 8 MDA / MDI."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.feature_importance import (
    FeatureImportanceResult,
    _default_scorer,
    mda_importance,
    mdi_importance,
)

# ── _default_scorer ──────────────────────────────────────────────────────────

def test_default_scorer_returns_zero_for_constant_predictions():
    y = np.arange(10)
    preds = np.zeros(10)
    assert _default_scorer(y, preds) == 0.0


def test_default_scorer_returns_one_for_perfect_rank_correlation():
    y = np.arange(10)
    preds = np.arange(10) * 3.0
    assert _default_scorer(y, preds) == pytest.approx(1.0)


def test_default_scorer_returns_minus_one_for_inverse_rank():
    y = np.arange(10)
    preds = np.arange(10)[::-1] * 3.0
    assert _default_scorer(y, preds) == pytest.approx(-1.0)


# ── mda_importance ───────────────────────────────────────────────────────────

class _LinearModel:
    """Tiny linear regressor — no sklearn dep so tests stay fast."""

    def __init__(self, feature_weights: dict[str, float] | None = None):
        # When weights are provided, ``fit`` is a no-op so tests can verify
        # that only the relevant feature drives predictions.
        self._weights = feature_weights
        self._coef = None
        self._cols: list[str] = []

    def fit(self, X, y):
        X_arr = np.asarray(X)
        # Least-squares coef — closed form.
        self._coef = np.linalg.lstsq(X_arr, y, rcond=None)[0]
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        if self._coef is None:
            return np.zeros(len(X_arr))
        return X_arr @ self._coef


def _synthetic_frame(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Three-feature frame where only ``x0`` predicts y."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x0": rng.normal(size=n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
    })
    y = (2.0 * X["x0"] + 0.1 * rng.normal(size=n)).to_numpy()
    return X, y


def test_mda_ranks_predictive_feature_highest():
    X, y = _synthetic_frame()
    result = mda_importance(_LinearModel(), X, y, random_state=0)
    assert isinstance(result, FeatureImportanceResult)
    assert result.importance["x0"] > result.importance["x1"]
    assert result.importance["x0"] > result.importance["x2"]


def test_mda_importance_preserves_feature_names():
    X, y = _synthetic_frame()
    result = mda_importance(_LinearModel(), X, y, random_state=0)
    assert list(result.importance.index) == list(X.columns)


def test_mda_random_feature_gets_near_zero_importance():
    X, y = _synthetic_frame(seed=1)
    result = mda_importance(_LinearModel(), X, y, random_state=0)
    # x1 and x2 are orthogonal noise — shuffling them barely changes IC.
    assert abs(result.importance["x1"]) < 0.3
    assert abs(result.importance["x2"]) < 0.3


def test_mda_with_custom_cv_splits():
    X, y = _synthetic_frame()
    n = len(X)
    splits = [
        (np.arange(n // 2), np.arange(n // 2, n)),
        (np.arange(n // 2, n), np.arange(n // 2)),
    ]
    result = mda_importance(_LinearModel(), X, y, cv_splits=splits, random_state=0)
    assert result.std is not None         # std exposed when >1 fold
    assert result.std.shape == result.importance.shape


def test_mda_empty_input_returns_empty_series():
    result = mda_importance(_LinearModel(), pd.DataFrame(), np.array([]))
    assert result.importance.empty


def test_mda_raises_on_length_mismatch():
    X, y = _synthetic_frame(n=20)
    with pytest.raises(ValueError, match="length mismatch"):
        mda_importance(_LinearModel(), X, y[:10])


def test_mda_top_k_helper():
    X, y = _synthetic_frame()
    result = mda_importance(_LinearModel(), X, y, random_state=0)
    top_1 = result.top_k(1)
    assert len(top_1) == 1
    assert top_1.index[0] == "x0"


# ── mdi_importance ───────────────────────────────────────────────────────────

class _FakeTree:
    """Sklearn-style tree with a constant ``feature_importances_``."""

    def __init__(self, imp): self.feature_importances_ = np.asarray(imp, dtype=float)


class _FakeEnsemble:
    def __init__(self, per_tree: list[list[float]]):
        self.estimators_ = [_FakeTree(row) for row in per_tree]
        self.feature_importances_ = np.mean(np.asarray(per_tree, dtype=float), axis=0)


def test_mdi_returns_feature_importances_for_lgbm_like_model():
    class _Lgbm:
        feature_importances_ = np.array([5.0, 3.0, 2.0])

    result = mdi_importance(_Lgbm(), ["a", "b", "c"])
    assert list(result.importance.index) == ["a", "b", "c"]
    assert result.importance["a"] == pytest.approx(5.0)
    # LightGBM exposes no per-tree estimators_, so std should be None.
    assert result.std is None


def test_mdi_computes_std_from_sklearn_estimators():
    per_tree = [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2.0, 2.0, 2.0]]
    ensemble = _FakeEnsemble(per_tree)
    result = mdi_importance(ensemble, ["a", "b", "c"])
    assert result.std is not None
    assert result.std["b"] == pytest.approx(0.0)
    assert result.std["a"] > 0


def test_mdi_without_model_returns_empty():
    result = mdi_importance(None, ["a", "b"])
    assert result.importance.empty


def test_mdi_handles_arity_mismatch_gracefully():
    class _Lgbm:
        feature_importances_ = np.array([1.0, 2.0])

    result = mdi_importance(_Lgbm(), ["a", "b", "c"])
    # Truncated to common min length
    assert list(result.importance.index) == ["a", "b"]
