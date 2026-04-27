"""
tests/test_options_sizing.py — delta-neutral / max-vega sizing.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.options_sizing import (
    LegMarket,
    cap_by_max_vega,
    delta_neutral_qty,
    scale_legs,
)
from strategies.options_legs import iron_condor, straddle

# ── delta_neutral_qty ────────────────────────────────────────────────────────

def test_delta_neutral_caps_directional_long_call():
    """A standalone long call is highly directional — large delta. The
    sizer truncates the multiplier so |delta × m × 100| stays under cap."""
    legs = straddle("SPY", "2026-06-19", strike=450, long=True)
    # Both legs at-the-money: delta is roughly +0.5 call, -0.5 put — net flat.
    # Substitute a clearly directional structure: just the long call leg.
    legs = [legs[0]]  # one buy_to_open call
    market = {0: LegMarket(S=450.0, T=0.25, sigma=0.20)}
    qty = delta_neutral_qty(legs, market, max_abs_delta_shares=100)
    assert qty >= 1
    # With ~50 share-deltas per contract, qty=2 gives ~100 — at the cap.
    assert qty <= 2


def test_delta_neutral_returns_zero_when_one_contract_breaches():
    """Strict caps below one contract's delta are honoured by truncation
    to zero — caller must not place the order."""
    from strategies.options_legs import OptionLeg

    legs = [
        OptionLeg(
            underlying="SPY", expiry="2026-06-19", strike=450,
            option_type="call", side="buy_to_open", qty=1,
        ),
    ]
    market = {0: LegMarket(S=450.0, T=0.25, sigma=0.20)}
    # 10 share-deltas cap is below the ~50-delta of an ATM call → 0.
    qty = delta_neutral_qty(legs, market, max_abs_delta_shares=10)
    assert qty == 0


def test_delta_neutral_handles_neutral_structure():
    """A balanced long straddle is delta-near-zero — sizer returns the
    base qty without dividing by ~zero."""
    legs = straddle("SPY", "2026-06-19", strike=450, long=True, qty=1)
    market = {
        0: LegMarket(S=450.0, T=0.25, sigma=0.20),
        1: LegMarket(S=450.0, T=0.25, sigma=0.20),
    }
    qty = delta_neutral_qty(legs, market, max_abs_delta_shares=50)
    # Net delta ≈ 0 → sizer falls back to the base leg qty (1).
    assert qty >= 1


def test_delta_neutral_rejects_non_positive_cap():
    legs = straddle("SPY", "2026-06-19", strike=450)
    market = {
        0: LegMarket(S=450.0, T=0.25, sigma=0.20),
        1: LegMarket(S=450.0, T=0.25, sigma=0.20),
    }
    with pytest.raises(ValueError, match="max_abs_delta"):
        delta_neutral_qty(legs, market, max_abs_delta_shares=0)


# ── cap_by_max_vega ──────────────────────────────────────────────────────────

def test_cap_by_max_vega_truncates_to_cap():
    """An iron condor is short vega; cap_by_max_vega picks the largest m."""
    legs = iron_condor(
        "SPY", "2026-06-19",
        put_long_strike=440, put_short_strike=445,
        call_short_strike=455, call_long_strike=460,
    )
    market = {
        i: LegMarket(S=450.0, T=0.25, sigma=0.20)
        for i in range(len(legs))
    }
    qty = cap_by_max_vega(legs, market, max_vega_dollars=200.0)
    assert qty >= 1


def test_cap_by_max_vega_zero_when_one_contract_breaches():
    legs = iron_condor(
        "SPY", "2026-06-19",
        put_long_strike=440, put_short_strike=445,
        call_short_strike=455, call_long_strike=460,
    )
    market = {
        i: LegMarket(S=450.0, T=0.25, sigma=0.20)
        for i in range(len(legs))
    }
    qty = cap_by_max_vega(legs, market, max_vega_dollars=0.001)
    assert qty == 0


def test_cap_by_max_vega_rejects_non_positive_cap():
    legs = iron_condor(
        "SPY", "2026-06-19",
        put_long_strike=440, put_short_strike=445,
        call_short_strike=455, call_long_strike=460,
    )
    market = {
        i: LegMarket(S=450.0, T=0.25, sigma=0.20)
        for i in range(len(legs))
    }
    with pytest.raises(ValueError, match="max_vega"):
        cap_by_max_vega(legs, market, max_vega_dollars=0)


# ── scale_legs ───────────────────────────────────────────────────────────────

def test_scale_legs_multiplies_qty():
    legs = straddle("SPY", "2026-06-19", strike=450, qty=1)
    scaled = scale_legs(legs, multiplier=4)
    assert all(leg.qty == 4 for leg in scaled)
    # Original leg list is untouched.
    assert all(leg.qty == 1 for leg in legs)


def test_scale_legs_zero_returns_empty():
    legs = straddle("SPY", "2026-06-19", strike=450, qty=1)
    assert scale_legs(legs, multiplier=0) == []
    assert scale_legs(legs, multiplier=-3) == []
