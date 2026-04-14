"""Tests for analysis/rl_sizer.py (Issue #28)."""
from __future__ import annotations

import numpy as np
import pytest


def test_obs_to_array_shape_and_one_hot():
    from analysis.rl_sizer import RLSizerObservation, _obs_to_array

    obs = RLSizerObservation(
        regime="trending_bull",
        volatility=0.20,
        win_rate=0.55,
        drawdown=0.03,
    )
    arr = _obs_to_array(obs)
    assert arr.shape == (7,)
    assert arr.dtype == np.float32
    # First element is 1.0 (trending_bull one-hot)
    assert arr[0] == pytest.approx(1.0)
    # Remaining regime positions are 0
    assert arr[1] == pytest.approx(0.0)
    assert arr[2] == pytest.approx(0.0)
    assert arr[3] == pytest.approx(0.0)
    # Feature slots
    assert arr[4] == pytest.approx(0.20)
    assert arr[5] == pytest.approx(0.55)
    assert arr[6] == pytest.approx(0.03)


def test_obs_to_array_unknown_regime_all_zeros():
    from analysis.rl_sizer import RLSizerObservation, _obs_to_array

    obs = RLSizerObservation(regime="unknown_regime", volatility=0.15, win_rate=0.5, drawdown=0.0)
    arr = _obs_to_array(obs)
    # No one-hot element should be 1.0 for an unknown regime
    assert sum(arr[:4]) == pytest.approx(0.0)


def test_kelly_fallback_returns_valid_range():
    from analysis.rl_sizer import RLPositionSizer, RLSizerObservation

    sizer = RLPositionSizer(model_path="/nonexistent/path/model.zip")
    obs = RLSizerObservation(regime="trending_bull", volatility=0.20, win_rate=0.60, drawdown=0.05)
    result = sizer.predict(obs)
    assert 0.0 <= result <= 2.0


def test_kelly_fallback_low_win_rate():
    from analysis.rl_sizer import RLPositionSizer, RLSizerObservation

    sizer = RLPositionSizer(model_path="/nonexistent/path/model.zip")
    obs = RLSizerObservation(regime="trending_bear", volatility=0.35, win_rate=0.20, drawdown=0.15)
    result = sizer.predict(obs)
    # Low win rate + bear regime should give a small or zero multiplier
    assert 0.0 <= result <= 2.0


def test_predict_with_missing_model_uses_kelly():
    from analysis.rl_sizer import RLPositionSizer, RLSizerObservation

    sizer = RLPositionSizer(model_path="/no/such/file.zip")
    assert sizer._model is None  # no model loaded
    obs = RLSizerObservation(regime="high_vol", volatility=0.40, win_rate=0.45, drawdown=0.10)
    mult = sizer.predict(obs)
    assert isinstance(mult, float)
    assert 0.0 <= mult <= 2.0


def test_predict_all_regime_states_in_range():
    from analysis.rl_sizer import REGIME_STATES, RLPositionSizer, RLSizerObservation

    sizer = RLPositionSizer(model_path="/no/model")
    for regime in REGIME_STATES:
        obs = RLSizerObservation(regime=regime, volatility=0.20, win_rate=0.5, drawdown=0.0)
        mult = sizer.predict(obs)
        assert 0.0 <= mult <= 2.0, f"Out of range for regime={regime}: {mult}"


def test_sizer_model_load_failure_falls_back_gracefully(tmp_path):
    """A corrupt checkpoint file should not crash predict()."""
    corrupt = tmp_path / "bad_model.zip"
    corrupt.write_bytes(b"not-a-valid-zip")

    from analysis.rl_sizer import RLPositionSizer, RLSizerObservation

    sizer = RLPositionSizer(model_path=str(corrupt))
    obs = RLSizerObservation(regime="mean_reverting", volatility=0.18, win_rate=0.52, drawdown=0.02)
    result = sizer.predict(obs)
    assert 0.0 <= result <= 2.0
