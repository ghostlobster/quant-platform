"""
tests/test_ensemble_signal.py — Unit tests for strategies/ensemble_signal.py.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.ensemble_signal import blend_signals


def test_equal_weights_default():
    a = {"AAPL": 0.6, "MSFT": -0.4}
    b = {"AAPL": 0.2, "MSFT": 0.8}
    result = blend_signals(a, b)
    assert abs(result["AAPL"] - 0.4) < 1e-9
    assert abs(result["MSFT"] - 0.2) < 1e-9


def test_custom_weights_normalised():
    a = {"AAPL": 1.0}
    b = {"AAPL": 0.0}
    # 75% weight on a, 25% on b
    result = blend_signals(a, b, weights=[3.0, 1.0])
    assert abs(result["AAPL"] - 0.75) < 1e-9


def test_missing_ticker_treated_as_zero():
    a = {"AAPL": 0.8}
    b = {"MSFT": 0.6}
    result = blend_signals(a, b)
    # AAPL: (0.8 + 0.0) / 2 = 0.4
    assert abs(result["AAPL"] - 0.4) < 1e-9
    # MSFT: (0.0 + 0.6) / 2 = 0.3
    assert abs(result["MSFT"] - 0.3) < 1e-9


def test_output_clipped_to_minus_one_one():
    a = {"X": 0.9}
    b = {"X": 0.9}
    result = blend_signals(a, b)
    assert result["X"] <= 1.0

    c = {"X": -0.9}
    d = {"X": -0.9}
    result2 = blend_signals(c, d)
    assert result2["X"] >= -1.0


def test_empty_input_returns_empty():
    result = blend_signals()
    assert result == {}


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        blend_signals({"A": 0.5}, {"B": 0.3}, weights=[1.0])


def test_all_zero_weights_raises():
    with pytest.raises(ValueError, match="zero"):
        blend_signals({"A": 0.5}, {"B": 0.3}, weights=[0.0, 0.0])


def test_three_models_union_tickers():
    a = {"AAPL": 0.6}
    b = {"MSFT": 0.4}
    c = {"GOOG": 0.2}
    result = blend_signals(a, b, c)
    assert set(result.keys()) == {"AAPL", "MSFT", "GOOG"}
    # Each model covers only its own ticker; others get 0.0
    assert abs(result["AAPL"] - 0.6 / 3) < 1e-9
