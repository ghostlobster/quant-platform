"""
tests/test_e2e_pretrade_guard_rejects_oversized.py — end-to-end pre-trade
guard chain.

Exercises the full ``OrderIntent → adapter.place_order → PreTradeGuard
→ broker.get_account_info → reject path → no journal entry`` chain on a
real ``PaperBrokerAdapter`` against a tmp SQLite quant.db. No mocks
beyond redirecting the on-disk DB paths.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.paper_trader as pt
import data.db as db_module
import journal.trading_journal as jt


@pytest.fixture
def paper_env(tmp_path, monkeypatch):-> None:
    """Isolate paper_trader + journal to per-test tmp SQLite files."""
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    # Disable the legacy paper-trader circuit breaker so the guard alone
    # decides accept / reject.
    monkeypatch.setattr(pt, "MAX_DRAWDOWN_PCT", 0.99)
    pt.init_paper_tables()
    jt.init_journal_table()
    return tmp_path


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/ghostlobster/quant-platform/issues/183 — "
           "PreTradeGuard reads `equity`/`portfolio_value` but the paper "
           "broker returns `total_value`, so dollar-sizing limits silently "
           "no-op. Fix lands separately; this test flips to PASS when the "
           "guard's _account_snapshot also accepts `total_value`.",
)
def test_guard_rejects_oversized_then_accepts_relaxed(paper_env, monkeypatch):-> None:
    """Tightening MAX_POSITION_PCT rejects a $10k order against $100k equity;
    relaxing the env var lets the same order fill."""
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    # 0.5% of equity → $500 cap. A 100×$100 = $10k buy must be rejected.
    monkeypatch.setenv("MAX_POSITION_PCT", "0.005")
    broker = PaperBrokerAdapter()

    rejected = broker.place_order("AAPL", qty=100, side="buy", limit_price=100.0)
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "max_position_pct"
    # No fill happened — portfolio is empty, no journal row written.
    assert pt.get_portfolio().empty
    assert jt.get_journal().empty

    # Relaxing the env to 50% leaves headroom; the same order now fills.
    monkeypatch.setenv("MAX_POSITION_PCT", "0.5")
    relaxed_broker = PaperBrokerAdapter()
    filled = relaxed_broker.place_order("AAPL", qty=100, side="buy", limit_price=100.0)
    assert filled["status"] == "filled"
    portfolio = pt.get_portfolio()
    assert set(portfolio["Ticker"]) == {"AAPL"}
    assert portfolio.iloc[0]["Shares"] == pytest.approx(100.0)


def test_blocklist_rejects_then_unset_allows(paper_env, monkeypatch):-> None:
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    monkeypatch.setenv("SYMBOL_BLOCKLIST", "MEME, SCAM")
    broker = PaperBrokerAdapter()
    rejected = broker.place_order("meme", qty=1, side="buy", limit_price=10.0)
    assert rejected["status"] == "rejected"
    assert rejected["reason"] == "symbol_blocklist"
    assert pt.get_portfolio().empty

    # Unrelated symbol still goes through.
    ok = broker.place_order("AAPL", qty=1, side="buy", limit_price=10.0)
    assert ok["status"] == "filled"


def test_killswitch_blocks_then_release_resumes(paper_env, monkeypatch, tmp_path):-> None:
    """Touching the kill-switch file blocks every adapter; removing the file
    resumes trading."""
    killswitch = tmp_path / "killswitch.flag"
    monkeypatch.setenv("KILLSWITCH_FILE", str(killswitch))

    from adapters.broker.paper_adapter import PaperBrokerAdapter

    killswitch.touch()
    broker = PaperBrokerAdapter()
    blocked = broker.place_order("AAPL", qty=1, side="buy", limit_price=100.0)
    assert blocked["status"] == "rejected"
    assert blocked["reason"] == "killswitch"
    assert pt.get_portfolio().empty

    # Remove the flag, fresh adapter, order proceeds.
    killswitch.unlink()
    broker2 = PaperBrokerAdapter()
    ok = broker2.place_order("AAPL", qty=1, side="buy", limit_price=100.0)
    assert ok["status"] == "filled"
