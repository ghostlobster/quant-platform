"""Tests for analysis/synthetic_paths.py — minimal LSTM-GAN."""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch  # noqa: F401
    _TORCH = True
except ImportError:
    _TORCH = False

skip_no_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")


def _ar1(n: int = 500, phi: float = 0.3, sigma: float = 0.01, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n) * sigma
    out = np.empty(n)
    out[0] = eps[0]
    for t in range(1, n):
        out[t] = phi * out[t - 1] + eps[t]
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(out, index=idx, name="ret")


# ── Gating ────────────────────────────────────────────────────────────────────

def test_train_generator_raises_without_torch():
    from analysis import synthetic_paths as mod
    with patch.object(mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            mod.train_generator(_ar1())


def test_sample_paths_raises_without_torch():
    from analysis import synthetic_paths as mod
    with patch.object(mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            mod.sample_paths(None, 5)  # type: ignore[arg-type]


def test_train_generator_rejects_too_short_series():
    from analysis.synthetic_paths import train_generator
    if not _TORCH:
        pytest.skip("torch not installed")
    with pytest.raises(ValueError, match="≥|need"):
        train_generator(pd.Series([0.01, 0.02, 0.03]), horizon=10)


# ── Torch happy paths ────────────────────────────────────────────────────────

@skip_no_torch
def test_train_returns_correct_artefact_shape():
    from analysis.synthetic_paths import train_generator
    series = _ar1(n=200, seed=1)
    result = train_generator(series, horizon=8, latent_dim=4, hidden=8, epochs=2, batch_size=16)
    assert result.horizon == 8
    assert result.latent_dim == 4
    assert isinstance(result.train_mean, float)
    assert isinstance(result.train_std, float)
    assert result.train_std > 0


@skip_no_torch
def test_sample_paths_returns_correct_shape():
    from analysis.synthetic_paths import sample_paths, train_generator
    series = _ar1(n=200, seed=2)
    result = train_generator(series, horizon=8, latent_dim=4, hidden=8, epochs=1, batch_size=16)
    paths = sample_paths(result, n_paths=12, seed=0)
    assert paths.shape == (12, 8)
    assert np.isfinite(paths).all()


@skip_no_torch
def test_sample_paths_reproducible_with_seed():
    from analysis.synthetic_paths import sample_paths, train_generator
    series = _ar1(n=200, seed=3)
    result = train_generator(series, horizon=6, latent_dim=4, hidden=8, epochs=1, batch_size=16)
    a = sample_paths(result, n_paths=4, seed=42)
    b = sample_paths(result, n_paths=4, seed=42)
    np.testing.assert_array_equal(a, b)


@skip_no_torch
def test_sample_paths_supports_shorter_horizon_than_training():
    from analysis.synthetic_paths import sample_paths, train_generator
    series = _ar1(n=200, seed=4)
    result = train_generator(series, horizon=10, latent_dim=4, hidden=8, epochs=1, batch_size=16)
    paths = sample_paths(result, n_paths=3, horizon=4, seed=1)
    assert paths.shape == (3, 4)


@skip_no_torch
def test_sample_paths_match_input_scale_order_of_magnitude():
    """Generator outputs should sit on the same numerical scale as training data."""
    from analysis.synthetic_paths import sample_paths, train_generator
    series = _ar1(n=400, sigma=0.02, seed=5)
    result = train_generator(series, horizon=10, latent_dim=4, hidden=8, epochs=10, batch_size=32)
    paths = sample_paths(result, n_paths=200, seed=10)
    # Std-of-stds rather than mean-of-stds — adversarial training is noisy
    # but the output should still live within an order of magnitude of input.
    assert 0.1 * series.std() <= paths.std() <= 10 * series.std()
