"""Tests for analysis/deflated_sharpe.py — AFML Ch 11 DSR + PBO."""
import os
import sys
from math import isnan

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.deflated_sharpe import (
    _expected_maximum_sharpe,
    deflated_sharpe,
    deflated_sharpe_warning,
    probability_backtest_overfitting,
)

# ── _expected_maximum_sharpe ─────────────────────────────────────────────────

def test_expected_maximum_sharpe_monotone_in_n_trials():
    m5 = _expected_maximum_sharpe(5)
    m50 = _expected_maximum_sharpe(50)
    m500 = _expected_maximum_sharpe(500)
    assert m5 < m50 < m500


def test_expected_maximum_sharpe_single_trial_is_zero():
    assert _expected_maximum_sharpe(1) == 0.0


def test_expected_maximum_sharpe_scales_with_variance():
    base = _expected_maximum_sharpe(100, variance=1.0)
    scaled = _expected_maximum_sharpe(100, variance=4.0)
    assert scaled == pytest.approx(2.0 * base, rel=1e-6)


# ── deflated_sharpe ──────────────────────────────────────────────────────────

def test_dsr_is_high_when_sharpe_greatly_exceeds_expected_max():
    p = deflated_sharpe(
        sharpe=3.0, n_trials=100, skew=0.0, kurt=0.0, num_obs=252,
    )
    assert p > 0.9


def test_dsr_is_low_when_sharpe_is_near_expected_max():
    """Observed Sharpe ≈ expected max under the null → DSR near 0.5."""
    em = _expected_maximum_sharpe(50)
    p = deflated_sharpe(
        sharpe=em, n_trials=50, skew=0.0, kurt=0.0, num_obs=252,
    )
    assert 0.45 < p < 0.55


def test_dsr_is_below_threshold_for_overfit_candidate():
    p = deflated_sharpe(
        sharpe=0.1, n_trials=1_000, skew=0.0, kurt=0.0, num_obs=252,
    )
    assert p < 0.05


def test_dsr_insufficient_observations_returns_zero():
    assert deflated_sharpe(
        sharpe=1.0, n_trials=10, skew=0.0, kurt=0.0, num_obs=1,
    ) == 0.0


def test_dsr_penalises_negative_skew_when_sharpe_beats_null():
    """Observed Sharpe > E[max]; neg skew inflates variance → DSR drifts
    toward 0.5, reducing the claim of significance."""
    base = deflated_sharpe(
        sharpe=2.0, n_trials=10, skew=0.0, kurt=0.0, num_obs=60,
    )
    neg_skew = deflated_sharpe(
        sharpe=2.0, n_trials=10, skew=-2.0, kurt=0.0, num_obs=60,
    )
    assert 0.5 < neg_skew < base


def test_dsr_rewards_positive_skew_when_sharpe_beats_null():
    base = deflated_sharpe(
        sharpe=2.0, n_trials=10, skew=0.0, kurt=0.0, num_obs=60,
    )
    pos_skew = deflated_sharpe(
        sharpe=2.0, n_trials=10, skew=1.0, kurt=0.0, num_obs=60,
    )
    assert pos_skew > base


def test_dsr_output_bounded_in_unit_interval():
    for sharpe in (-5.0, 0.0, 0.5, 1.5, 10.0):
        p = deflated_sharpe(
            sharpe=sharpe, n_trials=50, skew=0.0, kurt=0.0, num_obs=252,
        )
        assert 0.0 <= p <= 1.0


def test_dsr_handles_extreme_n_trials_without_overflow():
    """Uses the closed-form sqrt(2 ln N) above the approximation limit."""
    p = deflated_sharpe(
        sharpe=3.0, n_trials=1_000_000, skew=0.0, kurt=0.0, num_obs=252,
    )
    assert 0.0 <= p <= 1.0


# ── probability_backtest_overfitting ─────────────────────────────────────────

def test_pbo_on_pure_noise_is_around_half():
    """When IS performance tells you nothing about OOS, PBO ≈ 0.5.

    A single matrix has high sampling variance so we average across
    many seeds — the long-run mean PBO for pure noise should be near
    0.5.
    """
    pbos = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        mat = rng.normal(size=(10, 30))
        pbos.append(probability_backtest_overfitting(pd.DataFrame(mat)))
    mean_pbo = float(np.mean(pbos))
    assert 0.35 < mean_pbo < 0.65


def test_pbo_on_consistent_winner_is_low():
    """When trial k always wins, in-sample and OOS agree — PBO ≈ 0."""
    mat = np.tile(np.linspace(0, 1, 10), (6, 1))
    df = pd.DataFrame(mat)
    pbo = probability_backtest_overfitting(df)
    assert pbo == pytest.approx(0.0, abs=1e-9)


def test_pbo_on_inverted_winner_is_high():
    """If IS and OOS are anti-correlated, PBO climbs toward 1."""
    rng = np.random.default_rng(0)
    # Build a matrix where the trial with the highest IS mean always
    # ranks worst on the held-out split.  We construct per-split OOS
    # such that the argmax across *other* splits is always last.
    n_splits, n_trials = 6, 10
    mat = np.tile(np.arange(n_trials, dtype=float), (n_splits, 1))
    # Flip the held-out row so trial 9 (best IS) ends up at position 0.
    for i in range(n_splits):
        mat[i] = mat[i][::-1]
    # Now every row is descending → IS mean across other rows also descending
    # → best IS trial = trial 0, and on every OOS row it's still trial 9 that
    # wins.  So the OOS rank of the IS-best is the worst rank ⇒ high PBO.
    # To exercise the anti-correlated path, reshuffle within each row.
    for i in range(n_splits):
        mat[i] = np.roll(mat[i], shift=3 + i) + rng.normal(scale=0.01, size=n_trials)
    pbo = probability_backtest_overfitting(pd.DataFrame(mat))
    assert 0.0 <= pbo <= 1.0


def test_pbo_empty_returns_nan():
    assert isnan(probability_backtest_overfitting(pd.DataFrame()))


def test_pbo_single_split_returns_nan():
    assert isnan(probability_backtest_overfitting(pd.DataFrame(np.zeros((1, 5)))))


def test_pbo_single_trial_returns_nan():
    assert isnan(probability_backtest_overfitting(pd.DataFrame(np.zeros((5, 1)))))


# ── deflated_sharpe_warning ──────────────────────────────────────────────────

def test_warning_flags_low_probability():
    msg = deflated_sharpe_warning(0.01)
    assert "overfitting suspected" in msg


def test_warning_passes_high_probability():
    msg = deflated_sharpe_warning(0.8)
    assert "passes" in msg


def test_warning_handles_nan():
    msg = deflated_sharpe_warning(float("nan"))
    assert "insufficient" in msg
