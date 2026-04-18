"""Tests for analysis/chart_images.py — GAF + OHLC rasterisation."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.chart_images import ohlc_to_pixels, to_gramian_angular_field


def test_gaf_returns_correct_shape_and_range():
    series = pd.Series(np.linspace(100, 110, 32))
    img = to_gramian_angular_field(series, window=16)
    assert img.shape == (16, 16)
    assert img.dtype == np.float32
    assert (img >= -1.0).all() and (img <= 1.0).all()


def test_gaf_short_window_returns_zero_image():
    img = to_gramian_angular_field(pd.Series([1.0, 2.0]), window=8)
    assert img.shape == (8, 8)
    assert (img == 0.0).all()


def test_gaf_constant_series_returns_ones():
    img = to_gramian_angular_field(pd.Series([5.0] * 10), window=10)
    assert img.shape == (10, 10)
    assert np.allclose(img, 1.0)


def test_gaf_is_symmetric():
    series = pd.Series(np.sin(np.linspace(0, 2 * np.pi, 20)))
    img = to_gramian_angular_field(series, window=20)
    np.testing.assert_allclose(img, img.T, atol=1e-6)


def test_ohlc_to_pixels_shape_and_value_range():
    n = 16
    df = pd.DataFrame({
        "Open":  np.linspace(100, 110, n),
        "High":  np.linspace(101, 111, n),
        "Low":   np.linspace(99, 109, n),
        "Close": np.linspace(100.5, 110.5, n),
    })
    img = ohlc_to_pixels(df, window=n, height=24)
    assert img.shape == (24, n)
    assert img.dtype == np.float32
    assert (img >= 0.0).all() and (img <= 1.0).all()
    # At least some pixels are lit (we drew something)
    assert (img > 0).sum() > 0


def test_ohlc_to_pixels_short_input_returns_zero_image():
    df = pd.DataFrame({
        "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5],
    })
    img = ohlc_to_pixels(df, window=10, height=20)
    assert img.shape == (20, 10)
    assert (img == 0.0).all()


def test_ohlc_to_pixels_missing_columns_returns_zero():
    df = pd.DataFrame({"Open": [1, 2, 3], "Close": [1, 2, 3]})
    img = ohlc_to_pixels(df, window=3, height=10)
    assert (img == 0.0).all()


def test_ohlc_to_pixels_constant_prices_returns_zero():
    """Hi == Lo across the whole window collapses to a degenerate image."""
    df = pd.DataFrame({
        "Open": [5.0] * 8, "High": [5.0] * 8,
        "Low": [5.0] * 8, "Close": [5.0] * 8,
    })
    img = ohlc_to_pixels(df, window=8, height=10)
    assert (img == 0.0).all()
