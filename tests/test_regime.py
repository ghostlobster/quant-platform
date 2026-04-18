"""Tests for analysis/regime.py — market regime classification."""
import numpy as np
import pandas as pd

from analysis.regime import (
    REGIME_METADATA,
    detect_regime,
    is_regime_at_risk,
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

class TestIsRegimeAtRisk:
    def test_spy_at_sma_is_at_risk(self):
        # SPY exactly equal to SMA200 → within tolerance → at_risk=True
        assert is_regime_at_risk(spy_price=100.0, spy_sma200=100.0, vix=15.0) is True

    def test_spy_within_2pct_of_sma_is_at_risk(self):
        # +1% of SMA is within ±2% tolerance
        assert is_regime_at_risk(spy_price=101.0, spy_sma200=100.0, vix=15.0) is True

    def test_spy_far_from_sma_low_vix_not_at_risk(self):
        # +10% SMA deviation and VIX far from 20 → safe
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=15.0) is False

    def test_vix_within_10pct_of_20_is_at_risk(self):
        # VIX=19 is within ±10% of 20 → [18, 22] → at_risk=True even if SPY is far from SMA
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=19.0) is True

    def test_vix_at_22_is_at_risk(self):
        # Upper boundary of [18, 22] — still at risk
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=22.0) is True

    def test_vix_at_25_not_at_risk(self):
        # VIX=25 is outside ±10% of 20 and SPY is far from SMA
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=25.0) is False

    def test_zero_sma_returns_false(self):
        # Guard against division by zero; fail-open
        assert is_regime_at_risk(spy_price=100.0, spy_sma200=0.0, vix=15.0) is False


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
