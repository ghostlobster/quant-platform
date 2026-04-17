"""Tests for backtester/combinatorial_cv.py — AFML Ch 12."""
import os
import sys
from math import comb

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.combinatorial_cv import (
    _apply_embargo,
    _group_ranges,
    combinatorial_purged_cv,
    combinatorial_purged_splits,
    num_combinatorial_paths,
    paths_dataframe,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _fake_feature_matrix(n_samples: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")
    return pd.DataFrame({"x": np.arange(n_samples)}, index=dates)


# ── num_combinatorial_paths ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    "n, k, expected",
    [
        (10, 2, 45),          # C(10, 2) = 45
        (6, 3, 20),
        (4, 1, 4),
        (5, 5, 1),
        (5, 6, 0),            # k > n → 0
        (0, 1, 0),
        (5, 0, 0),
    ],
)
def test_num_combinatorial_paths(n, k, expected):
    assert num_combinatorial_paths(n, k) == expected


# ── _group_ranges ────────────────────────────────────────────────────────────

def test_group_ranges_equal_split():
    ranges = _group_ranges(10, 5)
    assert ranges == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


def test_group_ranges_uneven_split_covers_all():
    ranges = _group_ranges(11, 3)
    # Every sample must be covered exactly once
    covered = np.zeros(11, dtype=bool)
    for start, end in ranges:
        covered[start:end] = True
    assert covered.all()
    assert len(ranges) == 3


def test_group_ranges_zero_samples():
    assert _group_ranges(0, 5) == []


# ── combinatorial_purged_splits ──────────────────────────────────────────────

def test_emits_expected_number_of_paths():
    splits = list(combinatorial_purged_splits(
        n_samples=100, n_splits=6, n_test_splits=2, embargo=0,
    ))
    assert len(splits) == comb(6, 2)


def test_test_and_train_are_disjoint_without_embargo():
    splits = list(combinatorial_purged_splits(
        n_samples=60, n_splits=6, n_test_splits=2, embargo=0,
    ))
    for train, test in splits:
        assert set(train).isdisjoint(set(test))


def test_every_combination_covers_all_samples_without_embargo():
    """Without embargo the train and test sets together cover every bar."""
    splits = list(combinatorial_purged_splits(
        n_samples=60, n_splits=6, n_test_splits=2, embargo=0,
    ))
    for train, test in splits:
        assert len(train) + len(test) == 60


def test_test_set_has_expected_size():
    """Each test set covers n_test_splits contiguous groups."""
    n_samples = 60
    n_splits = 6
    n_test = 2
    group_size = n_samples / n_splits  # 10 per group
    splits = list(combinatorial_purged_splits(
        n_samples=n_samples, n_splits=n_splits, n_test_splits=n_test, embargo=0,
    ))
    for _, test in splits:
        assert len(test) == int(group_size * n_test)


def test_embargo_removes_train_bars_adjacent_to_test():
    n = 40
    splits = list(combinatorial_purged_splits(
        n_samples=n, n_splits=4, n_test_splits=1, embargo=3,
    ))
    # Take the fold whose test group is [10, 20)
    for train, test in splits:
        if test.min() == 10 and test.max() == 19:
            # Train should exclude 7, 8, 9 (embargo before) and 20, 21, 22 (embargo after)
            forbidden = {7, 8, 9, 20, 21, 22}
            assert forbidden.isdisjoint(set(train.tolist()))
            break
    else:
        pytest.fail("expected fold with test range [10, 20) not found")


def test_every_sample_appears_in_test_across_all_paths():
    """Union of test indices across all paths = full sample range."""
    n_samples = 50
    splits = list(combinatorial_purged_splits(
        n_samples=n_samples, n_splits=5, n_test_splits=2, embargo=0,
    ))
    seen = set()
    for _, test in splits:
        seen.update(test.tolist())
    assert seen == set(range(n_samples))


def test_each_group_appears_in_k_over_n_fraction_of_tests():
    """Combinatorial symmetry: each group is in test exactly
    C(n-1, k-1) times out of C(n, k)."""
    n_splits, n_test = 5, 2
    splits = list(combinatorial_purged_splits(
        n_samples=50, n_splits=n_splits, n_test_splits=n_test, embargo=0,
    ))
    group_ranges = _group_ranges(50, n_splits)
    # For each group range, count how many paths have that group in test
    for group_idx, (gs, ge) in enumerate(group_ranges):
        count = sum(
            1 for _, test in splits
            if ((test >= gs) & (test < ge)).any() and
               ((test >= gs) & (test < ge)).sum() == (ge - gs)
        )
        assert count == comb(n_splits - 1, n_test - 1)


def test_invalid_n_splits_raises():
    with pytest.raises(ValueError, match="n_splits"):
        list(combinatorial_purged_splits(n_samples=10, n_splits=1))


def test_invalid_n_test_splits_raises():
    with pytest.raises(ValueError, match="n_test_splits"):
        list(combinatorial_purged_splits(
            n_samples=10, n_splits=5, n_test_splits=6,
        ))


def test_negative_embargo_raises():
    with pytest.raises(ValueError, match="embargo"):
        list(combinatorial_purged_splits(
            n_samples=10, n_splits=5, n_test_splits=2, embargo=-1,
        ))


def test_n_splits_larger_than_samples_raises():
    with pytest.raises(ValueError):
        list(combinatorial_purged_splits(
            n_samples=4, n_splits=10, n_test_splits=2,
        ))


def test_zero_samples_emits_nothing():
    assert list(combinatorial_purged_splits(n_samples=0, n_splits=5)) == []


# ── combinatorial_purged_cv (materialised helper) ────────────────────────────

def test_cv_accepts_dataframe():
    fm = _fake_feature_matrix(60)
    splits = combinatorial_purged_cv(fm, n_splits=5, n_test_splits=2)
    assert len(splits) == comb(5, 2)


def test_cv_accepts_integer_n_samples():
    splits = combinatorial_purged_cv(60, n_splits=5, n_test_splits=2)
    assert len(splits) == comb(5, 2)


def test_cv_returns_empty_for_empty_input():
    empty = pd.DataFrame()
    assert combinatorial_purged_cv(empty, n_splits=5, n_test_splits=2) == []


def test_cv_returns_empty_for_zero_samples_int():
    assert combinatorial_purged_cv(0, n_splits=5, n_test_splits=2) == []


def test_cv_returns_empty_for_none_input():
    assert combinatorial_purged_cv(None, n_splits=5, n_test_splits=2) == []  # type: ignore[arg-type]


# ── paths_dataframe ──────────────────────────────────────────────────────────

def test_paths_dataframe_indexes_by_path_id():
    results = [{"sharpe": 1.0 + i, "ret": 0.01 * i} for i in range(comb(5, 2))]
    df = paths_dataframe(results, n_splits=5, n_test_splits=2)
    assert list(df.columns) == ["sharpe", "ret"]
    assert df.index.name == "path"
    assert len(df) == comb(5, 2)
    assert df.iloc[3]["sharpe"] == pytest.approx(4.0)


def test_paths_dataframe_wrong_length_raises():
    results = [{"sharpe": 1.0}] * 5  # wrong count for C(5, 2) = 10
    with pytest.raises(ValueError, match="expected 10 fold results"):
        paths_dataframe(results, n_splits=5, n_test_splits=2)


def test_paths_dataframe_empty_returns_empty_dataframe():
    df = paths_dataframe([], n_splits=5, n_test_splits=2)
    assert df.empty


# ── _apply_embargo ───────────────────────────────────────────────────────────

def test_apply_embargo_zero_is_noop():
    mask = np.ones(10, dtype=bool)
    out = _apply_embargo(mask, [(3, 6)], embargo=0, n_samples=10)
    np.testing.assert_array_equal(out, mask)


def test_apply_embargo_is_symmetric():
    mask = np.ones(20, dtype=bool)
    out = _apply_embargo(mask, [(10, 14)], embargo=2, n_samples=20)
    # Bars 8-9 (before) and 14-15 (after) purged; so true train bars
    # are 0..7 and 16..19.
    assert out[7] and not out[8] and not out[9]
    assert not out[10:14].any()
    assert not out[14] and not out[15]
    assert out[16]
