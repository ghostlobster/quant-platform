"""
tests/test_live_promotion_guard.py — paper→live promotion gate (#149).

Each case isolates the journal under JOURNAL_DB_PATH so the guard's
track-record probe runs against a deterministic fixture. The Alpaca /
IBKR / Schwab adapter classes are monkeypatched to a sentinel so we
exercise the gate without a real SDK.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import journal.trading_journal as jt
from providers.broker import (
    LivePromotionRefused,
    _live_promotion_check,
    _paper_track_record,
    get_broker,
    is_live_mode,
)

# ── Helpers ─────────────────────────────────────────────────────────────────

def _seed_paper_run(days: int, daily_pnl: float = 100.0) -> None:
    """Seed the journal with one closed trade per day for ``days`` days."""
    start = datetime.now(timezone.utc) - timedelta(days=days)
    for i in range(days):
        ts = (start + timedelta(days=i)).isoformat()
        # Cheat: drive entry_time + exit_time directly via a manual insert
        # so we don't have to wait for real-clock days to elapse.
        from journal.trading_journal import _get_connection

        conn = _get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO trades
                        (ticker, side, qty, entry_price, entry_time,
                         signal_source, regime, entry_notes,
                         exit_price, exit_time, pnl, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "TEST", "BUY", 1, 100.0, ts,
                        "test", "", "",
                        100.0 + (daily_pnl / 100), ts,
                        daily_pnl + (i * 0.1),  # tiny variance for non-zero std
                        "target",
                    ),
                )
        finally:
            conn.close()


@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    jt.init_journal_table()
    return tmp_path


@pytest.fixture
def fake_adapters(monkeypatch):
    """Replace each live adapter class with a constructor that records the call."""
    captured = {"alpaca": 0, "ibkr": 0, "schwab": 0}

    class _FakeAlpaca:
        def __init__(self):
            captured["alpaca"] += 1

    class _FakeIBKR:
        def __init__(self):
            captured["ibkr"] += 1

    class _FakeSchwab:
        def __init__(self):
            captured["schwab"] += 1

    monkeypatch.setattr(
        "adapters.broker.alpaca_adapter.AlpacaBrokerAdapter", _FakeAlpaca,
    )
    monkeypatch.setattr(
        "adapters.broker.ibkr_adapter.IBKRAdapter", _FakeIBKR,
    )
    monkeypatch.setattr(
        "adapters.broker.schwab_adapter.SchwabAdapter", _FakeSchwab,
    )
    return captured


# ── Track-record helper ─────────────────────────────────────────────────────

def test_track_record_empty_journal(journal_db):
    assert _paper_track_record() == (0, 0.0)


def test_track_record_counts_distinct_days_and_sharpe(journal_db):
    _seed_paper_run(days=20, daily_pnl=50.0)
    days, sharpe = _paper_track_record()
    assert days == 20
    assert sharpe > 0


def test_track_record_zero_std_returns_zero_sharpe(journal_db):
    _seed_paper_run(days=10, daily_pnl=100.0)
    # Patch out the variance so std == 0 → sharpe == 0 short-circuit.
    from journal.trading_journal import _get_connection

    conn = _get_connection()
    with conn:
        conn.execute("UPDATE trades SET pnl = 100.0")
    conn.close()
    days, sharpe = _paper_track_record()
    assert days == 10
    assert sharpe == 0.0


# ── Paper passes through unchanged ──────────────────────────────────────────

def test_paper_provider_never_gated(monkeypatch, journal_db):
    """The paper broker is exempt from the promotion gate by design."""
    monkeypatch.setenv("BROKER_PROVIDER", "paper")
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    # Should not raise.
    broker = get_broker()
    assert broker is not None


def test_is_live_mode_reflects_provider(monkeypatch):
    monkeypatch.setenv("BROKER_PROVIDER", "paper")
    assert is_live_mode() is False
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    assert is_live_mode() is True


# ── Live providers are gated ────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["alpaca", "ibkr", "schwab"])
def test_live_provider_refused_without_confirmation(
    monkeypatch, journal_db, fake_adapters, name,
):
    monkeypatch.setenv("BROKER_PROVIDER", name)
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    with pytest.raises(LivePromotionRefused, match="LIVE_TRADING_CONFIRMED"):
        get_broker()
    # No adapter was ever instantiated.
    assert sum(fake_adapters.values()) == 0


def test_live_provider_refused_with_short_track_record(
    monkeypatch, journal_db, fake_adapters,
):
    _seed_paper_run(days=5, daily_pnl=50.0)  # <30 day default
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    with pytest.raises(LivePromotionRefused, match="distinct trading days"):
        get_broker()


def test_live_provider_refused_with_low_sharpe(
    monkeypatch, journal_db, fake_adapters,
):
    """Negative-PnL paper run produces sharpe < 0.5 — gate refuses."""
    _seed_paper_run(days=30, daily_pnl=-100.0)
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.5")
    with pytest.raises(LivePromotionRefused, match="Sharpe"):
        get_broker()


def test_live_provider_admitted_with_track_record(
    monkeypatch, journal_db, fake_adapters,
):
    _seed_paper_run(days=30, daily_pnl=100.0)
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.0")
    broker = get_broker()
    assert broker is not None
    assert fake_adapters["alpaca"] == 1


def test_live_provider_can_relax_min_days(
    monkeypatch, journal_db, fake_adapters,
):
    _seed_paper_run(days=3, daily_pnl=100.0)
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_DAYS", "2")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.0")
    broker = get_broker()
    assert broker is not None


# ── Bypass ──────────────────────────────────────────────────────────────────

def test_live_provider_bypass_skips_both_checks(
    monkeypatch, journal_db, fake_adapters,
):
    monkeypatch.setenv("BROKER_PROVIDER", "alpaca")
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    monkeypatch.setenv("LIVE_PROMOTION_BYPASS", "1")
    broker = get_broker()
    assert broker is not None
    assert fake_adapters["alpaca"] == 1


# ── Unknown provider still raises ValueError ────────────────────────────────

def test_unknown_provider_raises_value_error(monkeypatch):
    monkeypatch.setenv("BROKER_PROVIDER", "fakebroker")
    with pytest.raises(ValueError, match="Unknown broker provider"):
        get_broker()


# ── Direct guard call ───────────────────────────────────────────────────────

def test_live_promotion_check_no_op_for_paper():
    # Should never raise, regardless of env state.
    _live_promotion_check("paper")
