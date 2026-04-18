"""Tests for analysis/microstructure.py — VPIN and Kyle's lambda."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.microstructure import bvc_buy_fraction, kyle_lambda, vpin


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2022-01-01", periods=n, freq="B")


# ── bvc_buy_fraction ───────────────────────────────────────────────────────────

def test_bvc_buy_fraction_bounds():
    np.random.seed(0)
    rets = pd.Series(np.random.randn(200) * 0.01, index=_index(200))
    frac = bvc_buy_fraction(rets, window=50)
    valid = frac.dropna()
    assert len(valid) > 100
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_bvc_buy_fraction_monotone_in_return():
    """Larger positive return → larger estimated buy fraction."""
    np.random.seed(1)
    rets = pd.Series(np.random.randn(200) * 0.01, index=_index(200))
    # Replace last bar with a very large positive shock
    rets.iloc[-1] = 0.10
    frac = bvc_buy_fraction(rets, window=50)
    # That bar's fraction should be close to 1 and strictly above the median
    assert frac.iloc[-1] > frac.dropna().median()
    assert frac.iloc[-1] > 0.9


def test_bvc_buy_fraction_nan_warmup():
    rets = pd.Series(np.zeros(30), index=_index(30))
    frac = bvc_buy_fraction(rets, window=50)
    assert frac.isna().all()


# ── vpin ──────────────────────────────────────────────────────────────────────

def test_vpin_in_unit_interval():
    np.random.seed(2)
    n = 300
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=_index(n))
    volume = pd.Series(
        np.random.randint(1_000_000, 5_000_000, n).astype(float), index=_index(n)
    )
    out = vpin(close, volume, window=50)
    valid = out.dropna()
    assert len(valid) > 100
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_vpin_trending_exceeds_noise():
    """A strongly trending series has higher VPIN than pure noise."""
    n = 300
    idx = _index(n)
    np.random.seed(3)

    # Trending series: deterministic +0.5 per bar with tiny noise
    trend_close = pd.Series(100 + 0.5 * np.arange(n) + np.random.randn(n) * 0.05, index=idx)
    # Noisy series: zero-drift Gaussian walk
    np.random.seed(4)
    noise_close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=idx)

    vol = pd.Series(1_000_000.0, index=idx)

    v_trend = vpin(trend_close, vol, window=50).dropna().mean()
    v_noise = vpin(noise_close, vol, window=50).dropna().mean()
    assert v_trend > v_noise


def test_vpin_warmup_is_nan():
    n = 60
    close = pd.Series(100 + np.arange(n, dtype=float), index=_index(n))
    vol = pd.Series(1_000_000.0, index=_index(n))
    out = vpin(close, vol, window=50)
    # First (window - 1) + 1 rows involve warmup of pct_change → std → rolling sum
    assert out.iloc[:50].isna().all()


# ── kyle_lambda ───────────────────────────────────────────────────────────────

def test_kyle_lambda_recovers_planted_slope():
    """Plant r_t = λ · sgn · V · p and verify the rolling estimator recovers λ."""
    n = 250
    idx = _index(n)
    np.random.seed(5)
    shocks = np.random.choice([-1.0, 1.0], size=n)
    vol = pd.Series(np.random.uniform(1e5, 1e6, n), index=idx)
    price = 100.0
    planted_lambda = 1e-10
    returns = planted_lambda * shocks * vol.to_numpy() * price
    close = pd.Series(price * np.cumprod(1.0 + returns), index=idx)

    out = kyle_lambda(close, vol, window=50).dropna()
    # Estimator should recover a positive λ close to the planted value.
    assert out.mean() > 0
    # Within one order of magnitude of the plant.
    ratio = out.median() / planted_lambda
    assert 0.1 < ratio < 10.0, f"recovered λ ratio out of range: {ratio}"


def test_kyle_lambda_zero_variance_regressor_is_nan():
    """Constant price (zero returns) → signed_dv identically zero → NaN, not DivByZero."""
    n = 60
    close = pd.Series(100.0, index=_index(n))
    vol = pd.Series(1_000_000.0, index=_index(n))
    out = kyle_lambda(close, vol, window=21)
    # No exception; result is all NaN
    assert out.isna().all()


def test_kyle_lambda_warmup_is_nan():
    n = 30
    close = pd.Series(100 + np.arange(n, dtype=float), index=_index(n))
    vol = pd.Series(1_000_000.0, index=_index(n))
    out = kyle_lambda(close, vol, window=21)
    assert out.iloc[:20].isna().all()
