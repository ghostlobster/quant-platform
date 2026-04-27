"""
tests/test_e2e_bracket_lifecycle.py — bracket-order chain end to end.

Walks the full ``OrderIntent → PaperBrokerAdapter.place_bracket → parent
fill (buy/sell) → paper_bracket_orders pending row → check_brackets price
tick → child fill → portfolio flat`` pipeline on a real paper-trader DB.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.paper_trader as pt
import data.db as db_module
from providers.broker import OrderIntent


@pytest.fixture
def paper_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    monkeypatch.setattr(pt, "MAX_DRAWDOWN_PCT", 0.99)
    pt.init_paper_tables()
    return tmp_path


def test_bracket_take_profit_lifecycle(paper_env):
    """A long bracket with TP=110 fills the parent at 100, ignores
    interim ticks, and closes the child when price crosses TP."""
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    intent = OrderIntent(
        symbol="SPY", qty=10, side="buy",
        limit_price=100.0,           # paper adapter uses limit_price as the fill
        take_profit=110.0,
        stop_loss=95.0,
    )
    broker = PaperBrokerAdapter()
    parent = broker.place_bracket(intent)
    assert parent["status"] == "parent_filled"
    assert parent["ticker"] == "SPY"
    assert parent["children"]["take_profit"] == 110.0
    assert parent["children"]["stop_loss"] == 95.0

    # Position is open, one bracket pending.
    portfolio = pt.get_portfolio()
    assert set(portfolio["Ticker"]) == {"SPY"}
    assert portfolio.iloc[0]["Shares"] == pytest.approx(10.0)
    assert len(pt.get_pending_brackets()) == 1

    # Interim tick below TP: still pending.
    no_fire = pt.check_brackets({"SPY": 105.0})
    assert no_fire == []
    assert len(pt.get_pending_brackets()) == 1

    # Cross TP: bracket fires.
    fires = pt.check_brackets({"SPY": 112.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "take_profit"
    assert pt.get_portfolio().empty
    assert pt.get_pending_brackets() == []


def test_bracket_stop_loss_lifecycle(paper_env):
    """The stop-loss leg fires when the price crosses below the SL level."""
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    intent = OrderIntent(
        symbol="SPY", qty=10, side="buy",
        limit_price=100.0,
        stop_loss=95.0,
    )
    PaperBrokerAdapter().place_bracket(intent)

    fires = pt.check_brackets({"SPY": 94.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "stop_loss"
    assert pt.get_portfolio().empty


def test_bracket_trailing_stop_tracks_peak(paper_env):
    """Trailing stop slides up with the price and fires on a 5% pullback
    from the running peak."""
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    intent = OrderIntent(
        symbol="SPY", qty=10, side="buy",
        limit_price=100.0,
        trail_percent=0.05,
    )
    PaperBrokerAdapter().place_bracket(intent)

    # Tick up to $120 establishes a new peak; trigger is 120 × 0.95 = 114.
    pt.check_brackets({"SPY": 120.0})
    no_fire = pt.check_brackets({"SPY": 117.0})  # still above 114
    assert no_fire == []

    fires = pt.check_brackets({"SPY": 113.0})    # crosses trigger
    assert len(fires) == 1
    assert fires[0]["reason"] == "trail"
    assert pt.get_portfolio().empty


def test_bracket_cancel_prevents_future_fill(paper_env):
    """Cancelling a bracket before it triggers leaves the position open
    and ignores subsequent price ticks."""
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    intent = OrderIntent(
        symbol="SPY", qty=10, side="buy",
        limit_price=100.0,
        stop_loss=90.0,
    )
    parent = PaperBrokerAdapter().place_bracket(intent)
    assert pt.cancel_bracket(parent["bracket_id"]) is True

    # No pending brackets → check_brackets is a no-op even on a deep dip.
    fires = pt.check_brackets({"SPY": 80.0})
    assert fires == []
    assert pt.get_pending_brackets() == []
    # The parent position is still open — cancellation only affects the children.
    assert set(pt.get_portfolio()["Ticker"]) == {"SPY"}
