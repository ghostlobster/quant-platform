"""
tests/test_options_legs.py — multi-leg builder validation.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.options_legs import (
    OptionLeg,
    calendar,
    closing_legs,
    iron_condor,
    straddle,
    vertical_spread,
)

# ── OptionLeg validation ─────────────────────────────────────────────────────

def test_option_leg_validates_strike():
    with pytest.raises(ValueError, match="strike"):
        OptionLeg(
            underlying="SPY", expiry="2026-06-19", strike=0,
            option_type="call", side="buy_to_open", qty=1,
        )


def test_option_leg_validates_qty():
    with pytest.raises(ValueError, match="qty"):
        OptionLeg(
            underlying="SPY", expiry="2026-06-19", strike=100,
            option_type="call", side="buy_to_open", qty=0,
        )


def test_option_leg_validates_side():
    with pytest.raises(ValueError, match="side"):
        OptionLeg(
            underlying="SPY", expiry="2026-06-19", strike=100,
            option_type="call", side="long", qty=1,
        )


def test_option_leg_validates_option_type():
    with pytest.raises(ValueError, match="option_type"):
        OptionLeg(
            underlying="SPY", expiry="2026-06-19", strike=100,
            option_type="future", side="buy_to_open", qty=1,
        )


def test_option_leg_validates_expiry_format():
    with pytest.raises(ValueError, match="expiry"):
        OptionLeg(
            underlying="SPY", expiry="june", strike=100,
            option_type="call", side="buy_to_open", qty=1,
        )


def test_signed_qty_long_short():
    long_leg = OptionLeg(
        underlying="SPY", expiry="2026-06-19", strike=100,
        option_type="call", side="buy_to_open", qty=3,
    )
    short_leg = OptionLeg(
        underlying="SPY", expiry="2026-06-19", strike=100,
        option_type="call", side="sell_to_open", qty=3,
    )
    assert long_leg.signed_qty == 3
    assert long_leg.is_long is True
    assert short_leg.signed_qty == -3
    assert short_leg.is_long is False


# ── Vertical spread ──────────────────────────────────────────────────────────

def test_vertical_spread_two_legs_opposite_sides():
    legs = vertical_spread(
        "SPY", "2026-06-19",
        long_strike=450.0, short_strike=455.0,
    )
    assert len(legs) == 2
    assert legs[0].side == "buy_to_open"
    assert legs[0].strike == 450.0
    assert legs[1].side == "sell_to_open"
    assert legs[1].strike == 455.0
    assert {leg.option_type for leg in legs} == {"call"}


def test_vertical_spread_rejects_equal_strikes():
    with pytest.raises(ValueError, match="differ"):
        vertical_spread("SPY", "2026-06-19", 450, 450)


# ── Iron condor ──────────────────────────────────────────────────────────────

def test_iron_condor_four_legs_correct_sides():
    legs = iron_condor(
        "SPY", "2026-06-19",
        put_long_strike=440, put_short_strike=445,
        call_short_strike=455, call_long_strike=460,
        qty=2,
    )
    assert len(legs) == 4
    assert all(leg.qty == 2 for leg in legs)

    # Legs in canonical order: long put, short put, short call, long call.
    assert legs[0].option_type == "put"  and legs[0].side == "buy_to_open"
    assert legs[1].option_type == "put"  and legs[1].side == "sell_to_open"
    assert legs[2].option_type == "call" and legs[2].side == "sell_to_open"
    assert legs[3].option_type == "call" and legs[3].side == "buy_to_open"

    # Net signed_qty should be zero (two long, two short).
    assert sum(leg.signed_qty for leg in legs) == 0


def test_iron_condor_rejects_misordered_strikes():
    with pytest.raises(ValueError, match="strikes must satisfy"):
        iron_condor(
            "SPY", "2026-06-19",
            put_long_strike=445, put_short_strike=440,  # swapped
            call_short_strike=455, call_long_strike=460,
        )


# ── Straddle ─────────────────────────────────────────────────────────────────

def test_long_straddle_two_legs_buy_to_open():
    legs = straddle("SPY", "2026-06-19", strike=450, long=True)
    assert len(legs) == 2
    assert {leg.option_type for leg in legs} == {"call", "put"}
    assert all(leg.side == "buy_to_open" for leg in legs)


def test_short_straddle_two_legs_sell_to_open():
    legs = straddle("SPY", "2026-06-19", strike=450, long=False)
    assert all(leg.side == "sell_to_open" for leg in legs)


# ── Calendar ─────────────────────────────────────────────────────────────────

def test_calendar_two_legs_short_near_long_far():
    legs = calendar("SPY", "2026-06-19", "2026-09-18", strike=450)
    assert len(legs) == 2
    assert legs[0].expiry == "2026-06-19" and legs[0].side == "sell_to_open"
    assert legs[1].expiry == "2026-09-18" and legs[1].side == "buy_to_open"


def test_calendar_rejects_inverted_expiry():
    with pytest.raises(ValueError, match="earlier than"):
        calendar("SPY", "2026-09-18", "2026-06-19", strike=450)


# ── closing_legs ─────────────────────────────────────────────────────────────

def test_closing_legs_mirrors_open_sides():
    open_legs = vertical_spread("SPY", "2026-06-19", 450, 455)
    closes = closing_legs(open_legs)
    assert closes[0].side == "sell_to_close"
    assert closes[1].side == "buy_to_close"
    # Strikes / expiries / qtys preserved.
    for o, c in zip(open_legs, closes):
        assert o.strike == c.strike
        assert o.expiry == c.expiry
        assert o.qty == c.qty


def test_closing_legs_rejects_already_closed():
    leg = OptionLeg(
        underlying="SPY", expiry="2026-06-19", strike=450,
        option_type="call", side="sell_to_close", qty=1,
    )
    with pytest.raises(ValueError, match="opening sides"):
        closing_legs([leg])
