"""Tests for analysis/regime.py — market regime classification."""
import numpy as np
import pandas as pd

from analysis.regime import (
    REGIME_METADATA,
    detect_regime,
    kelly_regime_multiplier,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 250, base: float = 100.0, last: float = 100.0) -> pd.Series:
    """Return a price series of length n where the final value is `last`.

    All values except the last are `base`, so the 200d SMA ≈ base.
    Setting `last` above or below `base` controls whether SPY is above/below SMA.
    """
    prices = np.full(n, base, dtype=float)
    prices[-1] = last
    return pd.Series(prices)


# ── Regime classification ─────────────────────────────────────────────────────

class TestDetectRegime:
    def test_trending_bull(self):
        # SPY above 200d SMA, VIX < 20
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bull"

    def test_trending_bear(self):
        # SPY below 200d SMA, VIX < 20
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bear"

    def test_mean_reverting_spy_above_sma(self):
        # VIX in [20, 30] — SPY position doesn't matter
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=25.0) == "mean_reverting"

    def test_mean_reverting_spy_below_sma(self):
        # VIX in [20, 30] — SPY position doesn't matter
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=25.0) == "mean_reverting"

    def test_high_vol_spy_above_sma(self):
        # VIX > 30 always wins regardless of SPY position
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=35.0) == "high_vol"

    def test_high_vol_spy_below_sma(self):
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=35.0) == "high_vol"


# ── Boundary conditions ───────────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_vix_exactly_20_is_mean_reverting(self):
        # VIX = 20 triggers mean_reverting, not trending_bull even if SPY > SMA
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=20.0) == "mean_reverting"

    def test_vix_exactly_30_is_mean_reverting(self):
        # VIX = 30 is still within [20, 30], not high_vol (which requires > 30)
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=30.0) == "mean_reverting"

    def test_vix_just_above_30_is_high_vol(self):
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=30.01) == "high_vol"

    def test_vix_just_below_20_uses_sma_bull(self):
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=19.99) == "trending_bull"

    def test_vix_just_below_20_uses_sma_bear(self):
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=19.99) == "trending_bear"

    def test_spy_equal_to_sma_is_trending_bear(self):
        # price == SMA is NOT strictly above, so classified as trending_bear
        prices = _make_prices(n=200, base=100.0, last=100.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bear"


# ── Kelly multiplier ──────────────────────────────────────────────────────────

class TestKellyRegimeMultiplier:
    def test_high_vol_halves_kelly(self):
        assert kelly_regime_multiplier("high_vol") == 0.5

    def test_trending_bull_full_kelly(self):
        assert kelly_regime_multiplier("trending_bull") == 1.0

    def test_trending_bear_full_kelly(self):
        assert kelly_regime_multiplier("trending_bear") == 1.0

    def test_mean_reverting_full_kelly(self):
        assert kelly_regime_multiplier("mean_reverting") == 1.0


# ── REGIME_METADATA ───────────────────────────────────────────────────────────

class TestRegimeMetadata:
    def test_all_four_regimes_present(self):
        for regime in ("trending_bull", "trending_bear", "mean_reverting", "high_vol"):
            assert regime in REGIME_METADATA, f"Missing regime: {regime}"

    def test_metadata_has_description(self):
        for regime, meta in REGIME_METADATA.items():
            assert "description" in meta, f"{regime} missing description"
            assert isinstance(meta["description"], str)
            assert len(meta["description"]) > 0

    def test_metadata_has_recommended_strategies(self):
        for regime, meta in REGIME_METADATA.items():
            assert "recommended_strategies" in meta, f"{regime} missing recommended_strategies"
            assert isinstance(meta["recommended_strategies"], list)
            assert len(meta["recommended_strategies"]) > 0
