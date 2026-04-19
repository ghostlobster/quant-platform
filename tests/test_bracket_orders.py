"""
tests/test_bracket_orders.py — tests for OrderIntent + paper bracket sim.

Each case uses a temporary SQLite file via :func:`data.db._DB_PATH` and
``PAPER_STARTING_CASH`` so nothing hits a shared DB or live broker.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.paper_trader as pt
import data.db as db_module
from providers.broker import OrderIntent

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "paper.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    monkeypatch.setattr(pt, "MAX_DRAWDOWN_PCT", 0.99)  # disable circuit breaker for tests
    pt.init_paper_tables()
    return tmp_path


# ── OrderIntent validation ────────────────────────────────────────────────────

def test_order_intent_rejects_non_positive_qty():
    with pytest.raises(ValueError, match="qty"):
        OrderIntent(symbol="AAPL", qty=0, side="buy")


def test_order_intent_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        OrderIntent(symbol="AAPL", qty=1, side="long")


def test_order_intent_rejects_limit_without_price():
    with pytest.raises(ValueError, match="limit_price"):
        OrderIntent(symbol="AAPL", qty=1, side="buy", order_type="limit")


def test_order_intent_is_bracket_flag():
    plain = OrderIntent(symbol="AAPL", qty=1, side="buy")
    assert plain.is_bracket() is False

    tp_only = OrderIntent(symbol="AAPL", qty=1, side="buy", take_profit=100.0)
    assert tp_only.is_bracket() is True

    trail = OrderIntent(symbol="AAPL", qty=1, side="buy", trail_percent=0.02)
    assert trail.is_bracket() is True


def test_order_intent_rejects_non_positive_trail():
    with pytest.raises(ValueError, match="trail_percent"):
        OrderIntent(symbol="AAPL", qty=1, side="buy", trail_percent=0)


# ── place_bracket parent fill ─────────────────────────────────────────────────

def test_place_bracket_requires_at_least_one_child(paper_db):
    with pytest.raises(ValueError, match="take_profit"):
        pt.place_bracket("AAPL", shares=10, side="buy", entry_price=100.0)


def test_place_bracket_fills_parent_and_records_pending(paper_db):
    result = pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        take_profit=110.0, stop_loss=95.0,
    )
    assert result["status"] == "parent_filled"
    assert result["ticker"] == "AAPL"
    assert result["qty"] == 10.0
    # Parent leg actually debited cash and opened a position.
    portfolio = pt.get_portfolio()
    assert set(portfolio["Ticker"]) == {"AAPL"}
    assert portfolio.iloc[0]["Shares"] == 10.0
    # Child order recorded as pending.
    pending = pt.get_pending_brackets()
    assert len(pending) == 1
    assert pending[0]["take_profit"] == 110.0
    assert pending[0]["stop_loss"] == 95.0
    assert pending[0]["status"] == "pending"


# ── check_brackets: take-profit, stop-loss, trailing-stop ─────────────────────

def test_check_brackets_fires_take_profit(paper_db):
    pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        take_profit=110.0, stop_loss=95.0,
    )
    fires = pt.check_brackets({"AAPL": 112.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "take_profit"
    # Position closed.
    assert pt.get_portfolio().empty
    # No pending brackets remain.
    assert pt.get_pending_brackets() == []


def test_check_brackets_fires_stop_loss(paper_db):
    pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        take_profit=110.0, stop_loss=95.0,
    )
    fires = pt.check_brackets({"AAPL": 94.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "stop_loss"
    assert pt.get_portfolio().empty


def test_check_brackets_trailing_stop_tracks_peak(paper_db):
    pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        trail_percent=0.05,
    )
    # Tick up to $120 (peak). Trailing stop is now $120 * 0.95 = $114.
    pt.check_brackets({"AAPL": 120.0})
    fires = pt.check_brackets({"AAPL": 117.0})
    assert fires == []  # still above trigger
    fires = pt.check_brackets({"AAPL": 113.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "trail"


def test_check_brackets_ignores_missing_prices(paper_db):
    pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        stop_loss=90.0,
    )
    fires = pt.check_brackets({"TSLA": 10.0})
    assert fires == []
    # Still pending.
    assert len(pt.get_pending_brackets()) == 1


# ── Short-side bracket (sell entry) ───────────────────────────────────────────

def test_short_bracket_take_profit_triggers_on_downside(paper_db):
    # Open a 5-share long first, then a bracket short closes it on TP.
    pt.buy("AAPL", 5, 100.0)
    pt.place_bracket(
        "AAPL", shares=5, side="sell", entry_price=100.0,
        take_profit=90.0, stop_loss=105.0,
    )
    fires = pt.check_brackets({"AAPL": 88.0})
    assert len(fires) == 1
    assert fires[0]["reason"] == "take_profit"


# ── cancel_bracket ────────────────────────────────────────────────────────────

def test_cancel_bracket_marks_pending_cancelled(paper_db):
    result = pt.place_bracket(
        "AAPL", shares=10, side="buy", entry_price=100.0,
        stop_loss=90.0,
    )
    assert pt.cancel_bracket(result["bracket_id"]) is True
    # Subsequent price ticks must not re-fire a cancelled bracket.
    fires = pt.check_brackets({"AAPL": 85.0})
    assert fires == []
    # No pending rows left.
    assert pt.get_pending_brackets() == []
