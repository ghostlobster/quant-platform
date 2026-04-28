"""
tests/test_pretrade_guard.py — tests for risk/pretrade_guard.py.

Every case builds a ``GuardLimits`` directly (not ``from_env``) for
determinism and uses a tiny in-memory ``FakeBroker`` to avoid network /
SDK dependencies. Where a test exercises the daily-loss check, the trading
journal is redirected to a per-test temp SQLite file via JOURNAL_DB_PATH.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import journal.trading_journal as jt
from risk.pretrade_guard import GuardLimits, GuardViolation, PreTradeGuard

# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeBroker:
    """Minimal BrokerProvider stand-in for guard tests."""

    def __init__(self, equity: float = 100_000.0, positions: list[dict] | None = None):
        self.equity = equity
        self._positions = positions or []

    def get_account_info(self) -> dict:
        return {"equity": self.equity, "cash": self.equity}

    def get_positions(self) -> list[dict]:
        return list(self._positions)

    # The BrokerProvider Protocol has more methods, but the guard only
    # calls get_account_info + get_positions, so the rest are omitted.


@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    """Isolate trading_journal to a per-test SQLite file."""
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    jt.init_journal_table()
    return tmp_path


# ── 1. Blocklist ─────────────────────────────────────────────────────────────

def test_blocklist_rejects():
    broker = FakeBroker()
    guard = PreTradeGuard(
        GuardLimits(symbol_blocklist=frozenset({"XYZ"})),
        broker,
    )
    # Blocklisted → rejected.
    with pytest.raises(GuardViolation) as exc:
        guard.check("XYZ", 10, "buy")
    assert exc.value.reason == "symbol_blocklist"
    # Non-blocklisted → accepted.
    guard.check("AAPL", 10, "buy")


# ── 2. Max position % ────────────────────────────────────────────────────────

def test_max_position_pct_rejects_oversized():
    # 1% cap on $100k equity → $1000 notional max. A 20×$100 = $2000 order
    # violates the cap.
    broker = FakeBroker(equity=100_000.0)
    guard = PreTradeGuard(GuardLimits(max_position_pct=0.01), broker)
    with pytest.raises(GuardViolation) as exc:
        guard.check("AAPL", 20, "buy", limit_price=100.0)
    assert exc.value.reason == "max_position_pct"
    # 5×$100 = $500 is under the cap.
    guard.check("AAPL", 5, "buy", limit_price=100.0)


# ── 3. Max gross exposure ────────────────────────────────────────────────────

def test_max_gross_exposure_rejects():
    # 150% cap; existing positions at 140% of equity; a new 20% buy would
    # push projected gross to 160%.
    existing = [
        {"symbol": "SPY", "market_value": 140_000.0},
    ]
    broker = FakeBroker(equity=100_000.0, positions=existing)
    guard = PreTradeGuard(GuardLimits(max_gross_exposure=1.5), broker)
    with pytest.raises(GuardViolation) as exc:
        guard.check("QQQ", 100, "buy", limit_price=200.0)  # $20k
    assert exc.value.reason == "max_gross_exposure"


# ── 4. Max daily loss % (halt) ───────────────────────────────────────────────

def test_max_daily_loss_halts(journal_db):
    # Log a realised loss = -$6,000 today; cap 5% of $100k = $5k → any
    # new order must be rejected.
    trade_id = jt.log_entry(
        ticker="AAPL", side="BUY", qty=10, price=100.0,
        signal_source="test", regime="", notes="",
    )
    jt.log_exit(trade_id, price=40.0, pnl=-6000.0, exit_reason="stop", notes="")
    broker = FakeBroker(equity=100_000.0)
    guard = PreTradeGuard(GuardLimits(max_daily_loss_pct=0.05), broker)
    with pytest.raises(GuardViolation) as exc:
        guard.check("AAPL", 1, "buy", limit_price=100.0)
    assert exc.value.reason == "max_daily_loss_pct"


# ── 5. Max orders per day (accepted count only) ──────────────────────────────

def test_max_orders_per_day_counts_accepted_only():
    broker = FakeBroker()
    guard = PreTradeGuard(
        GuardLimits(
            max_orders_per_day=3,
            symbol_blocklist=frozenset({"BAD"}),
        ),
        broker,
    )
    # Three accepted orders succeed.
    for _ in range(3):
        guard.check("AAPL", 1, "buy")
    # Fourth accepted-order attempt rejects.
    with pytest.raises(GuardViolation) as exc:
        guard.check("AAPL", 1, "buy")
    assert exc.value.reason == "max_orders_per_day"

    # Rejections do NOT increment the counter — reset the day and confirm
    # blocklist rejections leave headroom.
    fresh = PreTradeGuard(
        GuardLimits(
            max_orders_per_day=2,
            symbol_blocklist=frozenset({"BAD"}),
        ),
        broker,
    )
    with pytest.raises(GuardViolation):
        fresh.check("BAD", 1, "buy")  # rejected — does not count
    with pytest.raises(GuardViolation):
        fresh.check("BAD", 1, "buy")  # rejected — does not count
    fresh.check("AAPL", 1, "buy")  # 1st accepted
    fresh.check("AAPL", 1, "buy")  # 2nd accepted
    with pytest.raises(GuardViolation) as exc2:
        fresh.check("AAPL", 1, "buy")  # 3rd would exceed cap
    assert exc2.value.reason == "max_orders_per_day"


# ── 6. Kill-switch file ──────────────────────────────────────────────────────

def test_killswitch_file_blocks(tmp_path: Path):
    killswitch = tmp_path / "killswitch.flag"
    killswitch.touch()
    broker = FakeBroker()
    guard = PreTradeGuard(GuardLimits(killswitch_path=killswitch), broker)
    with pytest.raises(GuardViolation) as exc:
        guard.check("AAPL", 1, "buy")
    assert exc.value.reason == "killswitch"

    # Remove the file → trading resumes.
    killswitch.unlink()
    guard.check("AAPL", 1, "buy")


# ── 7. Default (all unset) — no restrictions ─────────────────────────────────

def test_all_limits_unset_allows():
    broker = FakeBroker()
    guard = PreTradeGuard(GuardLimits(), broker)
    # Any order passes; no exceptions.
    for qty in (1, 10, 1_000_000):
        guard.check("AAPL", qty, "buy", limit_price=100.0)


# ── 8. from_env parses every documented variable ─────────────────────────────

def test_from_env_parses(monkeypatch, tmp_path: Path):
    ks_path = tmp_path / "ks"
    monkeypatch.setenv("MAX_POSITION_PCT", "0.1")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.03")
    monkeypatch.setenv("MAX_GROSS_EXPOSURE", "1.8")
    monkeypatch.setenv("MAX_ORDERS_PER_DAY", "25")
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "penny, meme ,spam")
    monkeypatch.setenv("KILLSWITCH_FILE", str(ks_path))

    limits = GuardLimits.from_env()
    assert limits.max_position_pct == 0.1
    assert limits.max_daily_loss_pct == 0.03
    assert limits.max_gross_exposure == 1.8
    assert limits.max_orders_per_day == 25
    assert limits.symbol_blocklist == frozenset({"PENNY", "MEME", "SPAM"})
    assert limits.killswitch_path == ks_path
    assert limits.any_active() is True


# ── 9. Day rollover resets the per-day counter ───────────────────────────────

def test_day_rollover_resets_order_count():
    broker = FakeBroker()
    # Mutable clock so the test can advance the UTC day.
    current = {"now": datetime(2026, 4, 19, 9, 0, tzinfo=timezone.utc)}

    def _clock():
        return current["now"]

    guard = PreTradeGuard(
        GuardLimits(max_orders_per_day=2),
        broker,
        clock=_clock,
    )
    guard.check("AAPL", 1, "buy")
    guard.check("AAPL", 1, "buy")
    with pytest.raises(GuardViolation):
        guard.check("AAPL", 1, "buy")

    # Advance to next UTC day — counter resets.
    current["now"] = datetime(2026, 4, 20, 0, 5, tzinfo=timezone.utc)
    guard.check("AAPL", 1, "buy")
    guard.check("AAPL", 1, "buy")
    with pytest.raises(GuardViolation) as exc:
        guard.check("AAPL", 1, "buy")
    assert exc.value.reason == "max_orders_per_day"


# ── 10. Failure-mode coverage (negative-test discipline #231) ───────────────


def test_account_snapshot_swallows_broker_exception():
    """``_account_snapshot`` is fail-safe: a broker that raises during
    ``get_account_info`` must not propagate. The guard sees ``equity=0``.

    Documented contract — fail-open on sizing: when equity is unknown
    (broker outage) the sizing checks (max_position_pct,
    max_gross_exposure, max_daily_loss_pct) are **skipped** rather
    than tripped. Order-flow gates (kill-switch, blocklist, rate
    limit) keep working. Rationale: with no equity info the guard
    can't compute meaningful position percentages, and rejecting
    every order during a broker hiccup would lock the operator out
    of all trading — including the kill-switch-cancellation flow.

    NOTE: this is a real fail-open and worth a follow-on ticket if
    we want to flip it to fail-closed (e.g. require a configurable
    ``DENY_ON_BROKER_OUTAGE=1`` env var). For now this test locks the
    current behaviour so a refactor doesn't change it accidentally.
    """

    class _BrokenBroker:
        def get_account_info(self):
            raise RuntimeError("broker offline")

        def get_positions(self):
            return []

    guard = PreTradeGuard(
        GuardLimits(max_position_pct=0.1),  # sizing limit
        _BrokenBroker(),
    )
    equity, positions = guard._account_snapshot()
    assert equity == 0.0
    assert positions == []

    # Order-flow gates still work — kill-switch, blocklist, rate limit.
    # Sizing-only limits are skipped at equity=0, so the call does NOT
    # raise. Lock that.
    guard.check("AAPL", 100, "buy", limit_price=150.0)  # no exception


def test_account_snapshot_handles_get_positions_exception():
    """A broker whose ``get_positions`` raises is treated the same as
    the get_account_info failure — the guard doesn't propagate. Both
    halves are in the same try block so either failure trips the
    fall-back together."""

    class _PartialBroker:
        def get_account_info(self):
            return {"equity": 100_000.0}

        def get_positions(self):
            raise RuntimeError("positions endpoint flaky")

    guard = PreTradeGuard(
        GuardLimits(max_gross_exposure=50_000.0),
        _PartialBroker(),
    )
    equity, positions = guard._account_snapshot()
    assert equity == 0.0  # both halves in one try → either failure → 0
    assert positions == []


def test_account_snapshot_handles_unparseable_equity():
    """Equity that's a string ('100k') or a dict-shaped blob falls back
    to 0 rather than blowing up the float() conversion."""

    class _GarbageBroker:
        def get_account_info(self):
            return {"equity": "one hundred thousand"}

        def get_positions(self):
            return []

    guard = PreTradeGuard(GuardLimits(max_position_pct=0.1), _GarbageBroker())
    equity, _ = guard._account_snapshot()
    assert equity == 0.0
