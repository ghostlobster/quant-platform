"""Tests for ``tests/factories.py`` — synthetic-data factory library.

Locks the canonical factory contract so future drift between
in-file and library helpers is caught at PR time, not when a test
silently changes seed and goes flaky.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.factories import (
    make_feature_matrix,
    make_ohlcv,
    make_prices,
    make_returns,
)

# ── make_ohlcv ──────────────────────────────────────────────────────────────


class TestMakeOHLCV:
    def test_default_shape_and_columns(self) -> None:
        df = make_ohlcv()
        assert len(df) == 100
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_seed_is_deterministic(self) -> None:
        a = make_ohlcv(n=50, seed=7)
        b = make_ohlcv(n=50, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_produce_different_data(self) -> None:
        a = make_ohlcv(n=50, seed=7)
        b = make_ohlcv(n=50, seed=8)
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(a, b)

    def test_ohlc_invariant(self) -> None:
        """High ≥ Close ≥ Low for every bar."""
        df = make_ohlcv(n=100)
        assert (df["High"] >= df["Close"]).all()
        assert (df["Close"] >= df["Low"]).all()

    def test_volume_in_realistic_range(self) -> None:
        df = make_ohlcv(n=100)
        assert (df["Volume"] >= 1_000_000).all()
        assert (df["Volume"] < 5_000_000).all()

    def test_freq_business_days_skips_weekends(self) -> None:
        df = make_ohlcv(n=10, freq="B", start="2024-01-08")
        # Monday Jan 8 + 9 business days = Friday Jan 19; no weekends in there
        assert all(d.weekday() < 5 for d in df.index)

    def test_drift_increases_endpoint_on_average(self) -> None:
        """Positive drift → end price systematically higher than base
        across many seeds."""
        n = 200
        endpoints = [
            make_ohlcv(n=n, seed=s, drift=0.5)["Close"].iloc[-1]
            for s in range(20)
        ]
        # Average endpoint should be > base + n * drift / 4 (rough lower bound)
        assert np.mean(endpoints) > 100.0 + n * 0.5 / 4


# ── make_prices ─────────────────────────────────────────────────────────────


class TestMakePrices:
    def test_constant_when_last_unset(self) -> None:
        s = make_prices(n=10, base=100.0)
        assert (s == 100.0).all()

    def test_last_overrides_endpoint(self) -> None:
        s = make_prices(n=10, base=100.0, last=110.0)
        assert (s.iloc[:-1] == 100.0).all()
        assert s.iloc[-1] == 110.0

    def test_sma_recoverable_from_constant_prices(self) -> None:
        """A constant-then-shift series has a known mean — important
        for the regime tests' SMA-200 calculations."""
        s = make_prices(n=200, base=100.0, last=110.0)
        expected_mean = (199 * 100.0 + 110.0) / 200
        assert s.mean() == pytest.approx(expected_mean)


# ── make_returns ────────────────────────────────────────────────────────────


class TestMakeReturns:
    def test_default_shape(self) -> None:
        r = make_returns()
        assert len(r) == 252
        assert isinstance(r, pd.Series)

    def test_seed_is_deterministic(self) -> None:
        a = make_returns(n=50, seed=7)
        b = make_returns(n=50, seed=7)
        pd.testing.assert_series_equal(a, b)

    def test_sigma_controls_dispersion(self) -> None:
        """Higher sigma → higher empirical std (within ~10 % at n=10000)."""
        low = make_returns(n=10_000, sigma=0.005, seed=0)
        high = make_returns(n=10_000, sigma=0.020, seed=0)
        assert high.std() > low.std() * 3  # ~4x in expectation

    def test_mu_controls_mean(self) -> None:
        """Negative mu → negative empirical mean at large n."""
        r = make_returns(n=10_000, mu=-0.001, sigma=0.005, seed=0)
        assert r.mean() < 0


# ── make_feature_matrix ─────────────────────────────────────────────────────


class TestMakeFeatureMatrix:
    def test_shape_and_columns(self) -> None:
        from data.features import _FEATURE_COLS

        df = make_feature_matrix(n_dates=20, n_tickers=4, seed=0)
        assert len(df) == 80  # 20 * 4
        assert "date" in df.columns
        assert "ticker" in df.columns
        assert "fwd_ret_5d" in df.columns
        assert all(c in df.columns for c in _FEATURE_COLS)

    def test_seed_is_deterministic(self) -> None:
        a = make_feature_matrix(n_dates=10, n_tickers=3, seed=7)
        b = make_feature_matrix(n_dates=10, n_tickers=3, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_signal_strength_drives_correlation(self) -> None:
        """Higher fwd_signal_strength → higher correlation between
        feature[0] and fwd_ret_5d."""
        from data.features import _FEATURE_COLS

        col = _FEATURE_COLS[0]
        weak = make_feature_matrix(
            n_dates=80, n_tickers=10, seed=0, fwd_signal_strength=0.05
        )
        strong = make_feature_matrix(
            n_dates=80, n_tickers=10, seed=0, fwd_signal_strength=2.0
        )
        weak_corr = weak[col].corr(weak["fwd_ret_5d"])
        strong_corr = strong[col].corr(strong["fwd_ret_5d"])
        assert strong_corr > weak_corr
