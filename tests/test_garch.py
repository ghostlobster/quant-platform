"""Tests for analysis/garch.py — volatility forecasting (Jansen Ch 9.2).

The heavy GARCH fit is expensive so we keep the fixtures small but
long enough (>= 100 obs) for the optimiser to converge.  All tests
that actually call ``fit_garch`` are guarded by ``_ARCH_AVAILABLE``.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.garch import (
    _ARCH_AVAILABLE,
    GarchForecast,
    fit_garch,
    forecast_next_sigma,
)

# ── Unavailable-dep paths (always run) ───────────────────────────────────────

def test_fit_garch_raises_when_arch_missing(monkeypatch):
    monkeypatch.setattr("analysis.garch._ARCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="arch"):
        fit_garch(pd.Series(np.zeros(100)))


def test_forecast_next_sigma_returns_none_when_arch_missing(monkeypatch):
    monkeypatch.setattr("analysis.garch._ARCH_AVAILABLE", False)
    assert forecast_next_sigma(pd.Series(np.zeros(100))) is None


# ── Degenerate inputs (always run; hit early-exit path without arch) ─────────

def test_fit_garch_empty_returns_empty_forecast():
    if not _ARCH_AVAILABLE:
        pytest.skip("arch not installed")
    out = fit_garch(pd.Series(dtype=float))
    assert isinstance(out, GarchForecast)
    assert out.sigma_next is None
    assert out.converged is False


def test_fit_garch_too_short_returns_empty_forecast():
    if not _ARCH_AVAILABLE:
        pytest.skip("arch not installed")
    out = fit_garch(pd.Series(np.random.randn(10) * 0.01))
    assert out.sigma_next is None
    assert out.converged is False


def test_fit_garch_all_nan_returns_empty_forecast():
    if not _ARCH_AVAILABLE:
        pytest.skip("arch not installed")
    out = fit_garch(pd.Series([np.nan] * 100))
    assert out.sigma_next is None


# ── Happy-path fit (skipped when arch is absent) ─────────────────────────────

def _synth_returns(n: int = 300, seed: int = 0) -> pd.Series:
    """Quasi-realistic low-vol daily returns with mild conditional vol."""
    rng = np.random.default_rng(seed)
    vol = 0.005 + 0.002 * np.abs(np.sin(np.arange(n) / 30))
    eps = rng.normal(size=n)
    return pd.Series(vol * eps)


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch not installed")
def test_fit_garch_produces_positive_sigma_next():
    out = fit_garch(_synth_returns())
    assert out.converged
    assert out.sigma_next is not None
    assert out.sigma_next > 0.0


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch not installed")
def test_fit_garch_sigma_next_in_reasonable_range_vs_realised():
    """GARCH sigma forecast should be on the same order as realised std."""
    r = _synth_returns(n=400, seed=1)
    realised = float(r.std())
    out = fit_garch(r)
    assert out.sigma_next is not None
    # Within a 10x factor either way — loose but catches gross errors.
    assert realised / 10 < out.sigma_next < realised * 10


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch not installed")
def test_forecast_next_sigma_matches_fit_garch():
    r = _synth_returns(n=300, seed=2)
    full = fit_garch(r)
    shortcut = forecast_next_sigma(r)
    assert shortcut is not None
    assert shortcut == pytest.approx(full.sigma_next, rel=1e-6)


@pytest.mark.skipif(not _ARCH_AVAILABLE, reason="arch not installed")
def test_fit_garch_handles_nan_in_series():
    """fit_garch must drop NaNs rather than propagating them."""
    r = _synth_returns(n=300, seed=3).copy()
    r.iloc[::20] = np.nan
    out = fit_garch(r)
    assert out.converged
