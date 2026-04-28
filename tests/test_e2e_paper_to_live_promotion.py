"""
tests/test_e2e_paper_to_live_promotion.py — paper-to-live promotion gate.

Closes part of #222.

Exercises the full paper-to-live promotion chain:

  paper journal track record  → ``providers.broker._paper_track_record``
                              → ``_live_promotion_check`` (called from
                                 ``get_broker``) → either allow live
                                 broker instantiation or raise
                                 ``LivePromotionRefused`` with a clear
                                 message.

The two preconditions enforced together:

  1. ``LIVE_TRADING_CONFIRMED=true`` — explicit operator opt-in.
  2. Journal shows ≥ ``LIVE_PROMOTION_MIN_DAYS`` distinct trading days
     with a realised-PnL Sharpe ≥ ``LIVE_PROMOTION_MIN_SHARPE``.

Bypass (``LIVE_PROMOTION_BYPASS=1``) is for kill-switch recovery and
must skip both gates.

Cleanup-invariant fixture is opt-out: the journal seeded by these
tests is closed (entry+exit), so the "every fill has a journal row"
assertion would fire spuriously on the multiple seeded round trips.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from providers.broker import (
    LivePromotionRefused,
    _live_promotion_check,
    _paper_track_record,
)

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skip_invariant]


def _seed_paper_track_record(
    n_days: int,
    daily_pnls: list[float] | None = None,
) -> None:
    """Insert ``n_days`` closed paper trades with realised PnL.

    Uses the public journal API (``log_entry`` + ``log_exit``) and then
    backdates ``exit_time`` directly so each row registers as a distinct
    "day" in ``_paper_track_record`` (which groups by ``exit_time``,
    not ``entry_time``).
    """
    import sqlite3

    import journal.trading_journal as jt

    now = datetime.now(timezone.utc)
    pnls = daily_pnls or [0.5] * n_days
    assert len(pnls) == n_days

    trade_ids: list[tuple[int, str]] = []
    for i, pnl in enumerate(pnls):
        trade_id = jt.log_entry(
            ticker="AAPL",
            side="buy",
            qty=10,
            price=150.0,
            signal_source="momentum",
            regime="trending_bull",
        )
        jt.log_exit(
            trade_id=trade_id,
            price=150.0 + pnl,
            pnl=pnl,
            exit_reason="take_profit",
        )
        date = (now - timedelta(days=n_days - i)).isoformat()
        trade_ids.append((trade_id, date))

    # Backdate exit_time after log_exit (which always sets it to now).
    conn = sqlite3.connect(jt._get_db_path())
    try:
        for trade_id, date in trade_ids:
            conn.execute(
                "UPDATE trades SET entry_time=?, exit_time=? WHERE id=?",
                (date, date, trade_id),
            )
        conn.commit()
    finally:
        conn.close()


# ── Below threshold: refuse ────────────────────────────────────────────────


def test_refuses_when_confirmed_unset(e2e_journal_db, monkeypatch) -> None:
    """No ``LIVE_TRADING_CONFIRMED`` → refusal with a clear message
    naming the missing variable."""
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    monkeypatch.delenv("LIVE_PROMOTION_BYPASS", raising=False)
    with pytest.raises(LivePromotionRefused, match="LIVE_TRADING_CONFIRMED"):
        _live_promotion_check("alpaca")


def test_refuses_when_track_record_too_short(
    e2e_journal_db, monkeypatch
) -> None:
    """Confirmed but only 5 days in the journal → refusal naming the day count."""
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_DAYS", "30")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.0")
    monkeypatch.delenv("LIVE_PROMOTION_BYPASS", raising=False)
    _seed_paper_track_record(5)
    with pytest.raises(LivePromotionRefused, match="trading days"):
        _live_promotion_check("alpaca")


def test_refuses_when_sharpe_below_threshold(
    e2e_journal_db, monkeypatch
) -> None:
    """30 days but a Sharpe of zero (constant pnl) → refusal naming Sharpe."""
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_DAYS", "5")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.5")
    monkeypatch.delenv("LIVE_PROMOTION_BYPASS", raising=False)
    # Constant pnl ⇒ std dev 0 ⇒ Sharpe → 0 (or NaN). Either way < 0.5.
    _seed_paper_track_record(10, daily_pnls=[0.5] * 10)
    with pytest.raises(LivePromotionRefused, match="Sharpe"):
        _live_promotion_check("alpaca")


# ── Above threshold: allow ─────────────────────────────────────────────────


def test_allows_when_track_record_meets_thresholds(
    e2e_journal_db, monkeypatch
) -> None:
    """Confirmed + ≥ min_days + Sharpe ≥ min_sharpe → no exception."""
    monkeypatch.setenv("LIVE_TRADING_CONFIRMED", "true")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_DAYS", "5")
    monkeypatch.setenv("LIVE_PROMOTION_MIN_SHARPE", "0.1")
    monkeypatch.delenv("LIVE_PROMOTION_BYPASS", raising=False)
    # Mixed pnls so Sharpe is meaningful and positive.
    _seed_paper_track_record(
        10, daily_pnls=[1.0, 0.5, 0.8, 1.2, 0.9, 0.7, 1.1, 0.6, 1.3, 0.8]
    )
    # No exception means the gate allowed
    _live_promotion_check("alpaca")


def test_paper_provider_skips_check_entirely(monkeypatch) -> None:
    """Non-live providers (paper, mock) bypass the gate by design."""
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    monkeypatch.delenv("LIVE_PROMOTION_BYPASS", raising=False)
    # No exception → paper is allowed without confirmation
    _live_promotion_check("paper")
    _live_promotion_check("mock")


# ── Bypass override ─────────────────────────────────────────────────────────


def test_bypass_skips_both_gates(e2e_journal_db, monkeypatch) -> None:
    """LIVE_PROMOTION_BYPASS=1 → confirmation + track-record both skipped."""
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    monkeypatch.setenv("LIVE_PROMOTION_BYPASS", "1")
    # Empty journal — would normally refuse for short track record
    _live_promotion_check("alpaca")  # no exception


@pytest.mark.parametrize("flag", ["1", "true", "yes", "TRUE", "Yes"])
def test_bypass_flag_spellings(e2e_journal_db, monkeypatch, flag: str) -> None:
    monkeypatch.delenv("LIVE_TRADING_CONFIRMED", raising=False)
    monkeypatch.setenv("LIVE_PROMOTION_BYPASS", flag)
    _live_promotion_check("alpaca")


# ── _paper_track_record direct tests ────────────────────────────────────────


def test_track_record_zero_for_empty_journal(e2e_journal_db) -> None:
    """Fresh install ⇒ 0 days, 0.0 sharpe. The fail-safe so a zero-data
    journal never accidentally passes the gate."""
    days, sharpe = _paper_track_record()
    assert days == 0
    assert sharpe == 0.0


def test_track_record_counts_distinct_trading_days(e2e_journal_db) -> None:
    """3 trades on 3 distinct days → days = 3."""
    _seed_paper_track_record(3, daily_pnls=[0.5, 0.4, 0.6])
    days, _ = _paper_track_record()
    assert days == 3
