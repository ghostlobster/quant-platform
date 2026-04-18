"""Tests for analysis/risk_autoencoder.py — autoencoder risk factors."""
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


def _low_rank_panel(
    n_dates: int = 200,
    n_assets: int = 6,
    rank: int = 2,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic returns whose true latent rank is ``rank`` (plus noise)."""
    rng = np.random.default_rng(seed)
    factors = rng.standard_normal((n_dates, rank)) * 0.02
    loadings = rng.standard_normal((rank, n_assets)) * 0.5
    noise = rng.standard_normal((n_dates, n_assets)) * 0.001
    panel = factors @ loadings + noise
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    cols = [f"T{i}" for i in range(n_assets)]
    return pd.DataFrame(panel, index=idx, columns=cols)


# ── Gating ────────────────────────────────────────────────────────────────────

def test_train_autoencoder_raises_without_torch():
    from analysis import risk_autoencoder as mod
    with patch.object(mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            mod.train_autoencoder(_low_rank_panel())


def test_latent_factors_raises_without_torch():
    from analysis import risk_autoencoder as mod
    with patch.object(mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            mod.latent_factors(None, _low_rank_panel())  # type: ignore[arg-type]


def test_reconstruction_error_raises_without_torch():
    from analysis import risk_autoencoder as mod
    with patch.object(mod, "_TORCH_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="PyTorch"):
            mod.reconstruction_error(None, _low_rank_panel())  # type: ignore[arg-type]


def test_train_autoencoder_rejects_empty_panel():
    from analysis.risk_autoencoder import train_autoencoder
    if not _TORCH:
        pytest.skip("torch not installed")
    with pytest.raises(ValueError, match="empty"):
        train_autoencoder(pd.DataFrame())


def test_train_autoencoder_rejects_too_large_latent_dim():
    from analysis.risk_autoencoder import train_autoencoder
    if not _TORCH:
        pytest.skip("torch not installed")
    panel = _low_rank_panel(n_assets=3)
    with pytest.raises(ValueError, match="latent_dim"):
        train_autoencoder(panel, latent_dim=10)


# ── Torch happy paths ────────────────────────────────────────────────────────

@skip_no_torch
def test_train_returns_correct_artefact_shape():
    from analysis.risk_autoencoder import latent_factors, train_autoencoder
    panel = _low_rank_panel(n_dates=120, n_assets=6, rank=2)
    result = train_autoencoder(panel, latent_dim=2, hidden=8, epochs=20)
    assert result.latent_dim == 2
    assert result.feature_names == tuple(panel.columns)

    z = latent_factors(result, panel)
    assert z.shape == (len(panel), 2)
    assert list(z.columns) == ["f1", "f2"]


@skip_no_torch
def test_reconstruction_error_decreases_after_training():
    """A trained autoencoder must reconstruct better than a random init."""
    from analysis import risk_autoencoder as mod

    panel = _low_rank_panel(n_dates=200, n_assets=6, rank=2, seed=7)
    init = mod.train_autoencoder(panel, latent_dim=2, hidden=8, epochs=1, seed=7)
    init_err = mod.reconstruction_error(init, panel).mean()

    trained = mod.train_autoencoder(panel, latent_dim=2, hidden=8, epochs=200, seed=7)
    trained_err = mod.reconstruction_error(trained, panel).mean()

    assert trained_err < init_err


@skip_no_torch
def test_reconstruction_error_index_matches_input():
    from analysis import risk_autoencoder as mod
    panel = _low_rank_panel(n_dates=50, n_assets=4, rank=2)
    result = mod.train_autoencoder(panel, latent_dim=2, hidden=8, epochs=10)
    err = mod.reconstruction_error(result, panel)
    assert err.shape == (len(panel),)
    assert (err.index == panel.index).all()


@skip_no_torch
def test_latent_factors_align_with_input_index():
    from analysis import risk_autoencoder as mod
    panel = _low_rank_panel(n_dates=40, n_assets=4, rank=2)
    result = mod.train_autoencoder(panel, latent_dim=2, hidden=8, epochs=10)
    z = mod.latent_factors(result, panel)
    assert (z.index == panel.index).all()


@skip_no_torch
def test_train_autoencoder_handles_nan_inputs_via_fill():
    """NaNs in the panel should not crash; they're zero-filled internally."""
    from analysis.risk_autoencoder import train_autoencoder
    panel = _low_rank_panel(n_dates=60, n_assets=4, rank=2)
    panel.iloc[0, 0] = np.nan
    panel.iloc[5, 2] = np.nan
    # Should not raise
    result = train_autoencoder(panel, latent_dim=2, hidden=8, epochs=5)
    assert result.latent_dim == 2
