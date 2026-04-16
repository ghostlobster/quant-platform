"""Unit tests for strategies/meta_label.py."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.meta_label import (
    _SKLEARN_AVAILABLE,
    MetaLabeler,
    apply_meta_labels,
    filter_primary_by_confidence,
)

pytestmark = pytest.mark.skipif(
    not _SKLEARN_AVAILABLE, reason="sklearn is required for these tests"
)


def _make_dataset(n: int = 100, signal_strength: float = 0.7, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    primary = pd.Series(rng.choice([-1, 1], size=n), index=idx)
    # Feature that predicts whether the primary will be correct.
    feature = primary.values * signal_strength + rng.randn(n) * 0.3
    features = pd.DataFrame({"f0": feature, "f1": rng.randn(n)}, index=idx)
    # Bins align with primary when feature is strong; random otherwise.
    true_agreement = (feature > 0).astype(int) * 2 - 1
    bins = pd.Series(primary.values * true_agreement, index=idx)
    return primary, bins, features


def test_fit_returns_expected_metrics():
    primary, bins, features = _make_dataset()
    labeler = MetaLabeler(n_estimators=50)
    metrics = labeler.fit(primary, bins, features)

    assert set(metrics) == {"train_accuracy", "n_samples", "positive_rate"}
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert metrics["n_samples"] > 0
    assert labeler.is_trained()


def test_predict_returns_scores_in_unit_range():
    primary, bins, features = _make_dataset()
    labeler = MetaLabeler(n_estimators=50)
    labeler.fit(primary, bins, features)
    scores = labeler.predict(primary, features)

    assert scores.abs().max() <= 1.0
    assert len(scores) == len(primary)


def test_predict_without_fit_returns_primary_unchanged():
    primary, _, features = _make_dataset()
    labeler = MetaLabeler()
    out = labeler.predict(primary, features)
    assert out.equals(primary.astype(float).clip(-1.0, 1.0))


def test_fit_rejects_all_neutral_bins():
    primary, _, features = _make_dataset()
    bins = pd.Series(0, index=primary.index)
    with pytest.raises(ValueError, match="neutral"):
        MetaLabeler().fit(primary, bins, features)


def test_fit_rejects_disjoint_indexes():
    primary, bins, features = _make_dataset()
    shifted = pd.Series(
        primary.values,
        index=pd.date_range("2030-01-01", periods=len(primary), freq="D"),
    )
    with pytest.raises(ValueError, match="common index"):
        MetaLabeler().fit(shifted, bins, features)


def test_apply_meta_labels_without_bins_returns_primary():
    primary, _, features = _make_dataset()
    out = apply_meta_labels(primary, features, triple_barrier_bins=None)
    assert out.equals(primary.astype(float).clip(-1.0, 1.0))


def test_apply_meta_labels_with_bins_attenuates_signal():
    primary, bins, features = _make_dataset()
    out = apply_meta_labels(primary, features, triple_barrier_bins=bins)
    assert out.abs().mean() <= primary.abs().mean()


def test_filter_primary_by_confidence_zeros_low_confidence():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    primary = pd.Series([1, -1, 1, -1, 1], index=idx, dtype=float)
    final = pd.Series([0.9, -0.3, 0.6, -0.1, 0.05], index=idx)

    out = filter_primary_by_confidence(primary, final, min_confidence=0.5)
    assert out.iloc[0] == 1.0      # 0.9 ≥ 0.5 → keep
    assert out.iloc[1] == 0.0      # 0.3 < 0.5 → drop
    assert out.iloc[2] == 1.0
    assert out.iloc[3] == 0.0
    assert out.iloc[4] == 0.0
