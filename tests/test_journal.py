"""
tests/test_journal.py — Tests for journal/trading_journal.py.

Uses tmp_path + monkeypatch to isolate every test to its own SQLite file.
The integration test also redirects the paper_trader DB so the two modules
operate independently within the same test process.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.db as db_module
import broker.paper_trader as pt
import journal.trading_journal as jt


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    """Redirect the journal to a per-test temporary SQLite file."""
    db_path = str(tmp_path / "test_journal.db")
    monkeypatch.setenv("JOURNAL_DB_PATH", db_path)
    jt.init_journal_table()
    return db_path


@pytest.fixture
def paper_db(tmp_path, monkeypatch):
    """Redirect paper_trader to a per-test temporary SQLite file."""
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "test_paper.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    pt.init_paper_tables()
    return tmp_path / "test_paper.db"


# ── log_entry ─────────────────────────────────────────────────────────────────

def test_log_entry_returns_int(journal_db):
    trade_id = jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="test_signal")
    assert isinstance(trade_id, int)
    assert trade_id > 0


def test_log_entry_increments(journal_db):
    id1 = jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig")
    id2 = jt.log_entry("MSFT", "BUY", 5, 300.0, signal_source="sig")
    assert id2 > id1


# ── log_exit ──────────────────────────────────────────────────────────────────

def test_log_exit_updates_record(journal_db):
    trade_id = jt.log_entry("MSFT", "BUY", 5, 300.0, signal_source="momentum")
    jt.log_exit(trade_id, price=320.0, pnl=100.0, exit_reason="target_hit")

    df = jt.get_journal()
    row = df[df["id"] == trade_id].iloc[0]
    assert row["exit_price"] == pytest.approx(320.0)
    assert row["pnl"] == pytest.approx(100.0)
    assert row["exit_reason"] == "target_hit"
    assert row["exit_time"] is not None


def test_log_exit_with_notes(journal_db):
    tid = jt.log_entry("TSLA", "BUY", 2, 200.0, signal_source="momentum")
    jt.log_exit(tid, price=180.0, pnl=-40.0, exit_reason="stop_loss", notes="hit trailing stop")

    df = jt.get_journal()
    row = df[df["id"] == tid].iloc[0]
    assert row["exit_notes"] == "hit trailing stop"


# ── get_journal filters ───────────────────────────────────────────────────────

def test_get_journal_returns_all_without_filters(journal_db):
    jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig1")
    jt.log_entry("TSLA", "BUY", 5, 200.0, signal_source="sig2")
    df = jt.get_journal()
    assert len(df) == 2


def test_get_journal_filter_by_ticker(journal_db):
    jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig1")
    jt.log_entry("TSLA", "BUY", 5, 200.0, signal_source="sig2")
    df = jt.get_journal(ticker="AAPL")
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"


def test_get_journal_filter_by_ticker_case_insensitive(journal_db):
    jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig1")
    df = jt.get_journal(ticker="aapl")
    assert len(df) == 1


def test_get_journal_filter_start_date_future_returns_empty(journal_db):
    jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig1")
    df = jt.get_journal(start_date="2099-01-01")
    assert len(df) == 0


def test_get_journal_filter_end_date_includes_today(journal_db):
    jt.log_entry("AAPL", "BUY", 10, 150.0, signal_source="sig1")
    df = jt.get_journal(end_date="2099-12-31")
    assert len(df) == 1


def test_get_journal_empty_returns_dataframe_with_columns(journal_db):
    df = jt.get_journal()
    assert hasattr(df, "columns")
    assert "ticker" in df.columns
    assert "pnl" in df.columns


# ── win_rate_by_signal_source ─────────────────────────────────────────────────

def test_win_rate_by_signal_source(journal_db):
    # 3 wins + 1 loss for "momentum"
    for _ in range(3):
        tid = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="momentum")
        jt.log_exit(tid, price=110.0, pnl=100.0, exit_reason="target")

    tid = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="momentum")
    jt.log_exit(tid, price=90.0, pnl=-100.0, exit_reason="stop")

    df = jt.win_rate_by_signal_source()
    assert len(df) == 1
    row = df[df["signal_source"] == "momentum"].iloc[0]
    assert int(row["total_trades"]) == 4
    assert int(row["wins"]) == 3
    assert row["win_rate"] == pytest.approx(0.75)
    assert row["avg_pnl"] == pytest.approx(50.0)  # (100+100+100-100)/4


def test_win_rate_by_signal_source_multiple_sources(journal_db):
    tid1 = jt.log_entry("AAPL", "BUY", 1, 100.0, signal_source="sig_a")
    jt.log_exit(tid1, price=110.0, pnl=10.0, exit_reason="target")

    tid2 = jt.log_entry("AAPL", "BUY", 1, 100.0, signal_source="sig_b")
    jt.log_exit(tid2, price=90.0, pnl=-10.0, exit_reason="stop")

    df = jt.win_rate_by_signal_source()
    sources = set(df["signal_source"].tolist())
    assert "sig_a" in sources
    assert "sig_b" in sources


def test_win_rate_excludes_open_trades(journal_db):
    # Open trade (no exit) should NOT count
    jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="momentum")

    tid = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="momentum")
    jt.log_exit(tid, price=110.0, pnl=100.0, exit_reason="target")

    df = jt.win_rate_by_signal_source()
    assert int(df.iloc[0]["total_trades"]) == 1  # only the closed trade


def test_win_rate_empty_returns_empty_dataframe(journal_db):
    df = jt.win_rate_by_signal_source()
    assert df.empty
    assert "signal_source" in df.columns
    assert "win_rate" in df.columns


# ── avg_pnl_by_regime ─────────────────────────────────────────────────────────

def test_avg_pnl_by_regime(journal_db):
    tid1 = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="sig", regime="bull")
    jt.log_exit(tid1, price=110.0, pnl=100.0, exit_reason="target")

    tid2 = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="sig", regime="bear")
    jt.log_exit(tid2, price=90.0, pnl=-50.0, exit_reason="stop")

    df = jt.avg_pnl_by_regime()
    assert len(df) == 2

    bull_row = df[df["regime"] == "bull"].iloc[0]
    assert bull_row["avg_pnl"] == pytest.approx(100.0)
    assert bull_row["win_rate"] == pytest.approx(1.0)

    bear_row = df[df["regime"] == "bear"].iloc[0]
    assert bear_row["avg_pnl"] == pytest.approx(-50.0)
    assert bear_row["win_rate"] == pytest.approx(0.0)


def test_avg_pnl_by_regime_excludes_open_trades(journal_db):
    # Open trade should be excluded
    jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="sig", regime="bull")

    tid = jt.log_entry("AAPL", "BUY", 10, 100.0, signal_source="sig", regime="bull")
    jt.log_exit(tid, price=120.0, pnl=200.0, exit_reason="target")

    df = jt.avg_pnl_by_regime()
    assert int(df.iloc[0]["total_trades"]) == 1


def test_avg_pnl_by_regime_empty_returns_empty_dataframe(journal_db):
    df = jt.avg_pnl_by_regime()
    assert df.empty
    assert "regime" in df.columns
    assert "avg_pnl" in df.columns


# ── paper_trader integration ──────────────────────────────────────────────────

def test_paper_buy_creates_journal_entry(tmp_path, monkeypatch, paper_db):
    """paper buy should auto-create a BUY journal entry via the hook."""
    journal_path = str(tmp_path / "test_journal.db")
    monkeypatch.setenv("JOURNAL_DB_PATH", journal_path)

    pt.buy("AAPL", shares=10, price=150.0)

    df = jt.get_journal()
    assert len(df) >= 1
    buy_rows = df[df["side"] == "BUY"]
    assert len(buy_rows) >= 1
    assert buy_rows.iloc[0]["ticker"] == "AAPL"


def test_paper_buy_then_sell_creates_two_journal_entries(tmp_path, monkeypatch, paper_db):
    """paper_buy followed by paper_sell creates two separate journal entries."""
    journal_path = str(tmp_path / "test_journal.db")
    monkeypatch.setenv("JOURNAL_DB_PATH", journal_path)

    pt.buy("AAPL", shares=10, price=150.0)
    pt.sell("AAPL", shares=10, price=160.0)

    df = jt.get_journal()
    assert len(df) == 2
    sides = set(df["side"].tolist())
    assert "BUY" in sides
    assert "SELL" in sides
