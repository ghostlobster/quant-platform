"""Tests for analysis/drift.py — PSI / KS covariate-shift detectors."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.drift import (
    aggregate_drift,
    feature_psi,
    kolmogorov_smirnov,
    summarize_features,
)

# ── summarize_features ───────────────────────────────────────────────────────

def test_summarize_features_baseline():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "x": rng.standard_normal(500),
            "y": rng.standard_normal(500) + 3.0,
        }
    )
    stats = summarize_features(frame, ["x", "y"])
    assert set(stats.keys()) == {"x", "y"}
    for col in ("x", "y"):
        s = stats[col]
        assert set(s.keys()) == {"mean", "std", "q10", "q50", "q90", "n_samples"}
        assert s["n_samples"] == 500
        assert s["std"] > 0
        assert s["q10"] < s["q50"] < s["q90"]
    # Mean captures the shift on y
    assert abs(stats["y"]["mean"] - 3.0) < 0.3


def test_summarize_features_drops_sparse():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"a": rng.standard_normal(500), "b": [float("nan")] * 495 + [1.0, 2.0, 3.0, 4.0, 5.0]})
    stats = summarize_features(frame, ["a", "b"])
    assert "a" in stats
    assert "b" not in stats  # only 5 non-NaN rows — below _MIN_SAMPLES


def test_summarize_features_handles_missing_column():
    frame = pd.DataFrame({"x": np.random.default_rng(2).standard_normal(200)})
    stats = summarize_features(frame, ["x", "missing"])
    assert "x" in stats
    assert "missing" not in stats


def test_summarize_features_constant_column_kept():
    frame = pd.DataFrame({"const": [1.5] * 100})
    stats = summarize_features(frame, ["const"])
    # Constant columns are retained with std == 0 — the live comparator
    # will surface any shift as a large PSI.
    assert "const" in stats
    assert stats["const"]["std"] == 0.0


# ── feature_psi ──────────────────────────────────────────────────────────────

def _gaussian_frame(mean: float, n: int, seed: int, cols: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {c: rng.standard_normal(n) + mean for c in cols}
    return pd.DataFrame(data)


def test_feature_psi_identical_distribution_near_zero():
    cols = ["f1", "f2", "f3"]
    train = _gaussian_frame(0.0, 1000, seed=10, cols=cols)
    live = _gaussian_frame(0.0, 1000, seed=11, cols=cols)
    stats = summarize_features(train, cols)
    psi = feature_psi(stats, live)
    assert set(psi) == set(cols)
    for col in cols:
        assert psi[col] < 0.05, f"{col} PSI={psi[col]}"


def test_feature_psi_shifted_mean_above_retrain():
    cols = ["x"]
    train = _gaussian_frame(0.0, 1000, seed=20, cols=cols)
    live = _gaussian_frame(2.0, 1000, seed=21, cols=cols)  # +2σ shift
    stats = summarize_features(train, cols)
    psi = feature_psi(stats, live)
    assert psi["x"] >= 0.25


def test_feature_psi_handles_insufficient_samples():
    train = _gaussian_frame(0.0, 500, seed=30, cols=["x"])
    live = pd.DataFrame({"x": [0.1, 0.2, 0.3, 0.4, 0.5]})  # 5 rows
    stats = summarize_features(train, ["x"])
    psi = feature_psi(stats, live)
    assert math.isnan(psi["x"])


def test_feature_psi_handles_missing_live_column():
    train = _gaussian_frame(0.0, 500, seed=40, cols=["x"])
    live = pd.DataFrame({"y": [0.1, 0.2, 0.3]})
    stats = summarize_features(train, ["x"])
    psi = feature_psi(stats, live)
    assert math.isnan(psi["x"])


def test_feature_psi_skips_features_without_training_stats():
    # Training fingerprint only has 'x'; live has 'x' and 'y'.
    train = _gaussian_frame(0.0, 500, seed=50, cols=["x"])
    live = _gaussian_frame(0.0, 500, seed=51, cols=["x", "y"])
    stats = summarize_features(train, ["x"])
    psi = feature_psi(stats, live, feature_cols=["x", "y"])
    assert "x" in psi
    assert "y" not in psi  # no training stats for y → silently skipped


def test_feature_psi_floors_zero_mass_bins():
    # Live distribution concentrated in a single bin — extreme but finite PSI.
    train = _gaussian_frame(0.0, 500, seed=60, cols=["x"])
    stats = summarize_features(train, ["x"])
    live = pd.DataFrame({"x": [5.0] * 100})  # all in the far right tail
    psi = feature_psi(stats, live)
    assert psi["x"] > 0
    assert not math.isinf(psi["x"])


# ── kolmogorov_smirnov ───────────────────────────────────────────────────────

def test_kolmogorov_smirnov_identical_large_pvalue():
    rng = np.random.default_rng(100)
    a = rng.standard_normal(500)
    b = rng.standard_normal(500)
    stat, pv = kolmogorov_smirnov(a, b)
    assert 0.0 <= stat <= 1.0
    assert pv > 0.05


def test_kolmogorov_smirnov_shifted_small_pvalue():
    rng = np.random.default_rng(200)
    a = rng.standard_normal(500)
    b = rng.standard_normal(500) + 2.0
    stat, pv = kolmogorov_smirnov(a, b)
    assert stat > 0.3
    assert pv < 0.01


def test_kolmogorov_smirnov_nan_handling():
    a = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
    stat, pv = kolmogorov_smirnov(a, b)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= pv <= 1.0


def test_kolmogorov_smirnov_insufficient_samples_returns_nan():
    stat, pv = kolmogorov_smirnov(np.array([1.0]), np.array([1.0, 2.0]))
    assert math.isnan(stat)
    assert math.isnan(pv)


# ── aggregate_drift ──────────────────────────────────────────────────────────

def test_aggregate_drift_none_tier():
    out = aggregate_drift({"a": 0.01, "b": 0.05})
    assert out["level"] == "none"
    assert out["max_psi"] == pytest.approx(0.05)
    assert out["drifted_features"] == []


def test_aggregate_drift_monitor_tier():
    out = aggregate_drift({"a": 0.05, "b": 0.15, "c": 0.20})
    assert out["level"] == "monitor"
    assert out["max_psi"] == pytest.approx(0.20)
    # c > b > psi_monitor; sorted by descending PSI
    assert out["drifted_features"] == ["c", "b"]


def test_aggregate_drift_retrain_tier():
    out = aggregate_drift({"a": 0.05, "b": 0.30, "c": 0.15})
    assert out["level"] == "retrain"
    assert out["max_psi"] == pytest.approx(0.30)
    assert out["drifted_features"] == ["b", "c"]


def test_aggregate_drift_ignores_nans():
    out = aggregate_drift({"a": float("nan"), "b": 0.30})
    assert out["level"] == "retrain"
    assert out["max_psi"] == pytest.approx(0.30)
    assert out["drifted_features"] == ["b"]


def test_aggregate_drift_all_nan_returns_none_level():
    out = aggregate_drift({"a": float("nan"), "b": float("nan")})
    assert out == {"level": "none", "max_psi": None, "drifted_features": []}


def test_aggregate_drift_empty_dict():
    out = aggregate_drift({})
    assert out == {"level": "none", "max_psi": None, "drifted_features": []}


def test_aggregate_drift_custom_thresholds():
    # Stricter thresholds → a 0.05 PSI now counts as monitor.
    out = aggregate_drift(
        {"a": 0.05, "b": 0.11},
        psi_monitor=0.03, psi_retrain=0.10,
    )
    assert out["level"] == "retrain"
    assert "b" in out["drifted_features"]
    assert "a" in out["drifted_features"]  # now above custom monitor


# ── Smoke: PSI distinguishes shifts that would move a tier ───────────────────

def test_feature_psi_approximation_within_tolerance():
    """The fingerprint-based PSI should match tier-level decisions on a
    simple Gaussian shift even though it's an approximation of the exact
    PSI.
    """
    train = _gaussian_frame(0.0, 2000, seed=500, cols=["x"])
    live_ok = _gaussian_frame(0.0, 2000, seed=501, cols=["x"])
    live_shift_small = _gaussian_frame(0.3, 2000, seed=502, cols=["x"])
    live_shift_big = _gaussian_frame(2.0, 2000, seed=503, cols=["x"])

    stats = summarize_features(train, ["x"])
    psi_ok = feature_psi(stats, live_ok)["x"]
    psi_small = feature_psi(stats, live_shift_small)["x"]
    psi_big = feature_psi(stats, live_shift_big)["x"]

    assert psi_ok < psi_small < psi_big
    assert psi_ok < 0.05
    assert psi_big >= 0.25
