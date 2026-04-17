"""Tests for analysis/entropy_features.py — AFML Ch 18 entropy estimators."""
import os
import sys
from math import log

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.entropy_features import (
    _symbolise,
    entropy_features,
    konto_entropy,
    lempel_ziv_entropy,
    plug_in_entropy,
)

# ── _symbolise ───────────────────────────────────────────────────────────────

def test_symbolise_returns_fixed_alphabet():
    s = _symbolise([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_bins=5)
    # Each char is a digit 0..9 (modulo n_bins).
    assert all(c.isdigit() for c in s)


def test_symbolise_empty_returns_empty_string():
    assert _symbolise([], n_bins=5) == ""


def test_symbolise_constant_collapses_to_single_symbol():
    assert _symbolise([3.14] * 20, n_bins=5) == "0" * 20


def test_symbolise_skips_nan():
    s = _symbolise([1.0, np.nan, 2.0, np.nan, 3.0], n_bins=3)
    assert len(s) == 3


# ── plug_in_entropy ──────────────────────────────────────────────────────────

def test_plug_in_zero_for_deterministic_series():
    # Constant → one symbol → probability 1 → entropy 0
    assert plug_in_entropy([1.0] * 30, n_bins=5) == 0.0


def test_plug_in_approaches_log_n_bins_for_uniform_input():
    """An evenly-spaced input hits every bin once → H = log(n_bins)."""
    n_bins = 5
    vals = list(range(100))
    h = plug_in_entropy(vals, n_bins=n_bins)
    assert h == pytest.approx(log(n_bins), abs=0.2)


def test_plug_in_monotone_in_alphabet_diversity():
    """A mostly-constant input has lower entropy than a varied one."""
    constant_like = [1.0] * 95 + [2.0] * 5
    varied = list(range(100))
    assert plug_in_entropy(constant_like, n_bins=10) < plug_in_entropy(
        varied, n_bins=10,
    )


def test_plug_in_empty_input_returns_zero():
    assert plug_in_entropy([], n_bins=5) == 0.0


# ── lempel_ziv_entropy ───────────────────────────────────────────────────────

def test_lz_low_for_repeating_pattern():
    """'01010101…' compresses well → low LZ ratio."""
    vals = [0, 1] * 50
    lz = lempel_ziv_entropy(vals, n_bins=2)
    assert 0.0 <= lz <= 1.0


def test_lz_higher_for_random_input_than_repeating():
    rng = np.random.default_rng(0)
    repeating = [0, 1] * 100
    random = rng.integers(0, 5, size=200).tolist()
    assert lempel_ziv_entropy(random, n_bins=5) > lempel_ziv_entropy(
        repeating, n_bins=5,
    )


def test_lz_zero_for_empty_input():
    assert lempel_ziv_entropy([], n_bins=5) == 0.0


def test_lz_bounded_for_single_char_input():
    """A run of identical symbols parses as "0", "00", "000"… — a
    sub-linear number of distinct phrases.  Result must stay in
    ``(0, 1]``."""
    lz = lempel_ziv_entropy([0.0] * 20, n_bins=5)
    assert 0.0 < lz <= 1.0


# ── konto_entropy ────────────────────────────────────────────────────────────

def test_konto_zero_for_short_input():
    assert konto_entropy([1, 2, 3], n_bins=5) == 0.0


def test_konto_zero_or_low_for_purely_repeating_pattern():
    """A perfectly-periodic signal is very compressible → low entropy."""
    vals = [0, 1] * 50
    assert konto_entropy(vals, n_bins=2) < 1.0


def test_konto_higher_for_random_than_periodic():
    rng = np.random.default_rng(1)
    periodic = [0, 1, 2, 3] * 30
    random = rng.integers(0, 5, size=120).tolist()
    assert konto_entropy(random, n_bins=5) > konto_entropy(periodic, n_bins=5)


def test_konto_nonnegative():
    rng = np.random.default_rng(2)
    for seed in range(5):
        rng = np.random.default_rng(seed)
        vals = rng.normal(size=80).tolist()
        assert konto_entropy(vals, n_bins=5) >= 0.0


# ── entropy_features convenience ────────────────────────────────────────────

def test_entropy_features_returns_all_three_keys():
    rng = np.random.default_rng(4)
    vals = rng.normal(size=100)
    result = entropy_features(pd.Series(vals))
    assert set(result.keys()) == {"plug_in", "lempel_ziv", "konto"}
    for v in result.values():
        assert isinstance(v, float)
        assert v >= 0.0


def test_entropy_features_handles_empty_series():
    result = entropy_features(pd.Series(dtype=float))
    assert result == {"plug_in": 0.0, "lempel_ziv": 0.0, "konto": 0.0}
