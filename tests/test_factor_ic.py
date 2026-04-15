"""Tests for analysis/factor_ic.py — Information Coefficient analysis."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.factor_ic import (
    MEANINGFUL_ICIR,
    _spearman_corr,
    _ttest_1samp_pvalue,
    compute_ic,
    compute_ic_decay,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_feature_matrix(n_dates: int = 60, n_tickers: int = 10) -> pd.DataFrame:
    """Build a synthetic MultiIndex (date, ticker) feature matrix."""
    np.random.seed(0)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "date": d,
                "ticker": t,
                "ret_5d": np.random.randn(),
                "realised_vol_21d": abs(np.random.randn()) * 0.2,
                "fwd_ret_1d": np.random.randn() * 0.01,
                "fwd_ret_5d": np.random.randn() * 0.02,
                "fwd_ret_10d": np.random.randn() * 0.03,
                "fwd_ret_21d": np.random.randn() * 0.04,
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    return df


def _make_perfect_matrix(n_dates: int = 50, n_tickers: int = 10) -> pd.DataFrame:
    """Feature perfectly predicts fwd_ret_5d → IC ≈ 1."""
    np.random.seed(1)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        values = np.random.randn(n_tickers)
        for i, t in enumerate(tickers):
            rows.append({
                "date": d, "ticker": t,
                "perfect_feature": values[i],
                "fwd_ret_5d": values[i] + np.random.randn() * 0.001,  # near-perfect
            })

    return pd.DataFrame(rows).set_index(["date", "ticker"])


# ── _spearman_corr ─────────────────────────────────────────────────────────────

def test_spearman_corr_perfect_positive():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = _spearman_corr(x, x)
    assert abs(r - 1.0) < 1e-6


def test_spearman_corr_perfect_negative():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = _spearman_corr(x, -x)
    assert abs(r + 1.0) < 1e-6


def test_spearman_corr_too_short():
    r = _spearman_corr(np.array([1.0]), np.array([1.0]))
    assert np.isnan(r)


# ── _ttest_1samp_pvalue ────────────────────────────────────────────────────────

def test_ttest_pvalue_near_zero_for_strong_signal():
    """Large consistent IC series should yield very small p-value."""
    ic_series = np.ones(100) * 0.3  # constant non-zero mean
    p = _ttest_1samp_pvalue(ic_series)
    assert p < 0.01


def test_ttest_pvalue_near_one_for_noise():
    """Near-zero mean IC series should have high p-value."""
    np.random.seed(99)
    ic_series = np.random.randn(200) * 0.001  # mean ≈ 0
    p = _ttest_1samp_pvalue(ic_series)
    assert p > 0.3


# ── compute_ic ─────────────────────────────────────────────────────────────────

def test_compute_ic_returns_dict_keys():
    fm = _make_feature_matrix()
    results = compute_ic(fm, feature_cols=["ret_5d"])
    assert "ret_5d" in results
    result = results["ret_5d"]
    for key in ("ic_mean", "ic_std", "icir", "p_value", "n_dates"):
        assert key in result, f"Missing key: {key}"


def test_compute_ic_range():
    """IC mean must lie in [-1, 1]."""
    fm = _make_feature_matrix()
    results = compute_ic(fm, feature_cols=["ret_5d", "realised_vol_21d"])
    for feat, stats in results.items():
        ic = stats["ic_mean"]
        if not np.isnan(ic):
            assert -1.0 <= ic <= 1.0, f"{feat}: IC {ic} out of [-1, 1]"


def test_compute_ic_icir_formula():
    """ICIR = ic_mean / ic_std where ic_std != 0."""
    fm = _make_feature_matrix(n_dates=80, n_tickers=12)
    results = compute_ic(fm, feature_cols=["ret_5d"])
    stats = results["ret_5d"]
    if stats["ic_std"] != 0 and not np.isnan(stats["ic_std"]):
        expected_icir = stats["ic_mean"] / stats["ic_std"]
        assert abs(stats["icir"] - expected_icir) < 1e-9


def test_compute_ic_perfect_predictor():
    """A feature equal to forward returns should have IC ≈ 1."""
    fm = _make_perfect_matrix(n_dates=60, n_tickers=10)
    results = compute_ic(fm, forward_return_col="fwd_ret_5d",
                         feature_cols=["perfect_feature"])
    ic = results["perfect_feature"]["ic_mean"]
    assert not np.isnan(ic)
    assert ic > 0.90, f"Expected IC ≈ 1 for perfect predictor, got {ic:.4f}"


def test_compute_ic_random_noise():
    """A random feature should have IC ≈ 0 and high p-value."""
    fm = _make_feature_matrix(n_dates=100, n_tickers=15)
    results = compute_ic(fm, feature_cols=["realised_vol_21d"])
    stats = results["realised_vol_21d"]
    if not np.isnan(stats["ic_mean"]):
        assert abs(stats["ic_mean"]) < 0.4, (
            f"Random feature IC too large: {stats['ic_mean']:.4f}"
        )


def test_compute_ic_empty_matrix():
    results = compute_ic(pd.DataFrame())
    assert results == {}


def test_compute_ic_missing_target():
    fm = _make_feature_matrix()
    results = compute_ic(fm, forward_return_col="nonexistent_col")
    assert results == {}


def test_compute_ic_skips_dates_with_few_tickers():
    """Dates with fewer than 5 tickers should not contribute to IC series."""
    # Build a matrix where most dates have < 5 tickers
    dates = pd.date_range("2023-01-01", periods=30, freq="B")
    rows = []
    for d in dates:
        for t in ["T1", "T2"]:  # only 2 tickers per date — below threshold
            rows.append({"date": d, "ticker": t,
                         "ret_5d": np.random.randn(),
                         "fwd_ret_5d": np.random.randn()})
    fm = pd.DataFrame(rows).set_index(["date", "ticker"])

    results = compute_ic(fm, feature_cols=["ret_5d"])
    # n_dates should be 0 since every date has < 5 tickers
    assert results["ret_5d"]["n_dates"] == 0


def test_compute_ic_default_excludes_fwd_ret_columns():
    """When feature_cols=None, fwd_ret_* columns should not appear in results."""
    fm = _make_feature_matrix()
    results = compute_ic(fm)
    for key in results:
        assert not key.startswith("fwd_ret_"), f"fwd_ret column leaked into IC results: {key}"


# ── compute_ic_decay ───────────────────────────────────────────────────────────

def test_compute_ic_decay_horizons():
    fm = _make_feature_matrix(n_dates=80, n_tickers=10)
    horizons = [1, 5, 10, 21]
    decay = compute_ic_decay(fm, feature_col="ret_5d", horizons=horizons)
    assert set(decay.keys()) == set(horizons)
    for h in horizons:
        assert "ic_mean" in decay[h]


def test_compute_ic_decay_default_horizons():
    fm = _make_feature_matrix(n_dates=80, n_tickers=10)
    decay = compute_ic_decay(fm, feature_col="ret_5d")
    assert set(decay.keys()) == {1, 5, 10, 21}


# ── MEANINGFUL_ICIR constant ───────────────────────────────────────────────────

def test_meaningful_icir_constant():
    assert MEANINGFUL_ICIR == 0.5
