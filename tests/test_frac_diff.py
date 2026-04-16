"""Unit tests for data/frac_diff.py."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.frac_diff import find_min_d, frac_diff_ffd, frac_diff_weights


def _series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_weights_first_entry_is_one():
    w = frac_diff_weights(0.5)
    assert w[0] == 1.0
    assert len(w) > 1


def test_weights_decay_toward_zero():
    w = frac_diff_weights(0.4, thresh=1e-5)
    assert abs(w[-1]) < 1e-3


def test_weights_d_zero_is_identity():
    # d=0 means no differentiation; the recursion terminates at w=[1.0].
    w = frac_diff_weights(0.0)
    assert len(w) == 1
    assert w[0] == 1.0


def test_frac_diff_d_one_matches_first_difference_shape():
    # d=1 produces weights [1, -1] so the result should match diff().
    prices = _series(np.arange(1, 21, dtype=float) * 10.0)
    ffd = frac_diff_ffd(prices, d=1.0, thresh=1e-5)
    expected = prices.diff()
    pd.testing.assert_series_equal(
        ffd.dropna(), expected.dropna(), check_names=False, rtol=1e-6, atol=1e-6,
    )


def test_frac_diff_preserves_memory_for_small_d():
    # For small d the output should still correlate with the original level.
    rng = np.random.RandomState(42)
    prices = _series(100 + np.cumsum(rng.randn(500)))
    ffd = frac_diff_ffd(prices, d=0.3, thresh=1e-3)
    valid = ffd.dropna()
    assert len(valid) > 20
    corr = valid.corr(prices.reindex(valid.index))
    assert corr > 0.3  # non-trivial memory retained


def test_frac_diff_empty_series_returns_empty():
    empty = pd.Series(dtype=float)
    ffd = frac_diff_ffd(empty, d=0.5)
    assert ffd.empty


def test_frac_diff_short_series_returns_all_nan():
    # Series shorter than the weight window → every output NaN.
    short = _series([100.0, 101.0])
    ffd = frac_diff_ffd(short, d=0.9, thresh=1e-10)
    assert ffd.isna().all()


def test_find_min_d_returns_value_in_unit_interval():
    rng = np.random.RandomState(0)
    prices = _series(100 + np.cumsum(rng.randn(200)))
    d = find_min_d(prices, d_values=[0.0, 0.5, 1.0])
    assert 0.0 <= d <= 1.0


def test_negative_d_raises():
    import pytest
    with pytest.raises(ValueError):
        frac_diff_weights(-0.1)
