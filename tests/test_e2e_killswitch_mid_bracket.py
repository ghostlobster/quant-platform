"""
tests/test_e2e_killswitch_mid_bracket.py — kill-switch tripped while a
bracket is open.

Closes part of #222.

Exercises the operator-cleanup chain: a bracket is open with pending
TP/SL children, the kill-switch trips, the operator's adapter must
refuse new orders but the existing ``cancel_bracket`` path still
works so the operator can clean up.

Chain under test:

  PaperBrokerAdapter.place_bracket → paper_bracket_orders 'pending'
    → trip_killswitch (Phase 1 fixture)
    → adapter.place_order rejected (PreTradeGuard sees the flag)
    → cancel_bracket marks bracket 'cancelled' (allowed — operator
       cleanup is not gated by the killswitch)
    → release flag → new orders succeed again

The test docstring locks the operator-cleanup invariant: a kill-switch
event must NOT make pending brackets uncancellable. That would leave
the operator unable to flatten exposure.
"""
from __future__ import annotations

import pytest

from adapters.broker.paper_adapter import PaperBrokerAdapter
from broker.paper_trader import (
    cancel_bracket,
    get_pending_brackets,
    place_bracket,
)
from providers.broker import OrderIntent

pytestmark = pytest.mark.e2e


# ── Happy path: trip + reject + cancel + release ────────────────────────────


def test_killswitch_blocks_new_orders_but_allows_bracket_cleanup(
    e2e_paper_env, trip_killswitch, monkeypatch
) -> None:
    """Open bracket → trip → adapter rejects new order → cancel still
    works → release → new order accepted."""
    # 1. Open a bracket while the kill-switch is OFF.
    monkeypatch.delenv("MAX_POSITION_PCT", raising=False)
    parent = place_bracket(
        ticker="SPY", shares=10, side="buy",
        entry_price=100.0, take_profit=110.0, stop_loss=95.0,
    )
    bracket_id = parent["bracket_id"]
    assert parent["status"] == "parent_filled"
    pending = get_pending_brackets()
    assert any(b["id"] == bracket_id for b in pending)

    # 2. Trip the kill-switch.
    flag_path = trip_killswitch()
    assert flag_path.exists()

    # 3. Adapter must reject any *new* order.
    adapter = PaperBrokerAdapter()
    intent = OrderIntent(symbol="AAPL", qty=1, side="buy")
    rejected = adapter.place_order(intent.symbol, intent.qty, intent.side)
    assert rejected["status"] == "rejected"
    assert "killswitch" in (rejected.get("reason") or "").lower()

    # 4. Operator cleanup path — cancelling the bracket is still allowed.
    ok = cancel_bracket(bracket_id)
    assert ok is True
    pending = get_pending_brackets()
    assert all(b["id"] != bracket_id for b in pending)

    # 5. Release the kill-switch — new orders resume.
    flag_path.unlink()
    assert not flag_path.exists()
    accepted = adapter.place_order("AAPL", 1, "buy")
    assert accepted["status"] == "filled"


# ── Mid-tick scenario ──────────────────────────────────────────────────────


def test_check_brackets_during_killswitch_still_fires_existing_children(
    e2e_paper_env, trip_killswitch
) -> None:
    """Documents the production behaviour: kill-switch only blocks
    *new* orders. Pending bracket children that hit their trigger
    price still execute — that's intentional (we want positions
    flattening, not stuck open). The cleanup-invariant fixture
    confirms the resulting fill is journaled.
    """
    from broker.paper_trader import check_brackets

    # Open a bracket then trip the kill-switch.
    parent = place_bracket(
        ticker="SPY", shares=10, side="buy",
        entry_price=100.0, take_profit=110.0,
    )
    bracket_id = parent["bracket_id"]
    trip_killswitch()

    # Tick to the take-profit price — the child should still fire.
    fired = check_brackets({"SPY": 111.0})
    assert any(f["bracket_id"] == bracket_id for f in fired)
    pending = get_pending_brackets()
    assert all(b["id"] != bracket_id for b in pending)


# ── Failure injection: cancel_bracket on a non-existent id ─────────────────


def test_cancel_bracket_returns_false_for_unknown_id(e2e_paper_env) -> None:
    """The cleanup path is fail-soft — cancelling a non-existent bracket
    returns False rather than raising, so an idempotent retry from the
    operator UI is safe."""
    assert cancel_bracket(99_999_999) is False
