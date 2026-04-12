"""Tests for cron/monthly_wf.py"""
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on the path when running tests directly
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backtester.walk_forward import WalkForwardResult
from backtester.engine import BacktestResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Minimal OHLCV DataFrame that satisfies walk_forward requirements."""
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


def _make_wf_result() -> WalkForwardResult:
    """A fake WalkForwardResult with two positive windows."""
    br = BacktestResult(
        ticker="TST",
        strategy="sma_crossover",
        start_date="2022-01-01",
        end_date="2022-06-01",
        total_return_pct=0.05,
        buy_hold_return_pct=0.03,
        sharpe_ratio=1.2,
        sortino_ratio=1.5,
        max_drawdown_pct=-0.03,
        num_trades=4,
        win_rate_pct=60.0,
        avg_trade_pct=0.5,
    )
    return WalkForwardResult(
        windows=[br, br],
        avg_return=0.05,
        avg_sharpe=1.2,
        avg_sortino=1.5,
        avg_max_drawdown=-0.03,
        total_trades=8,
        consistency_score=1.0,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path, monkeypatch):
    """Redirect DB_PATH to a temp directory and return the path."""
    db_path = tmp_path / "data" / "wf_history.db"
    monkeypatch.setattr("cron.monthly_wf.DB_PATH", db_path)
    return db_path


@pytest.fixture()
def mock_yf_download(_make_df=None):
    """Patch yfinance.download to return a deterministic DataFrame."""
    df = _make_ohlcv()
    with patch("cron.monthly_wf.yf.download", return_value=df) as mock:
        yield mock


@pytest.fixture()
def mock_walk_forward():
    """Patch walk_forward to return a fake result without running the engine."""
    wf = _make_wf_result()
    with patch("cron.monthly_wf.walk_forward", return_value=wf) as mock:
        yield mock


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDbInit:
    def test_db_and_table_created_on_first_run(self, tmp_db, mock_yf_download, mock_walk_forward, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "SPY")

        from cron import monthly_wf
        monthly_wf.run()

        assert tmp_db.exists(), "DB file should be created"

        conn = sqlite3.connect(tmp_db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()

        assert "wf_results" in tables

    def test_schema_columns(self, tmp_db, mock_yf_download, mock_walk_forward, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "SPY")

        from cron import monthly_wf
        monthly_wf.run()

        conn = sqlite3.connect(tmp_db)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(wf_results)").fetchall()}
        conn.close()

        assert {"run_date", "ticker", "consistency_score", "total_return", "n_windows"}.issubset(cols)


class TestPersistence:
    def test_results_written_correctly(self, tmp_db, mock_yf_download, mock_walk_forward, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "SPY")

        from cron import monthly_wf
        rc = monthly_wf.run()

        assert rc == 0

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute("SELECT ticker, consistency_score, total_return, n_windows FROM wf_results").fetchall()
        conn.close()

        assert len(rows) == 1
        ticker, cs, tr, nw = rows[0]
        assert ticker == "SPY"
        assert cs == pytest.approx(1.0)
        assert tr == pytest.approx(0.05)
        assert nw == 2

    def test_multiple_tickers_all_persisted(self, tmp_db, mock_yf_download, mock_walk_forward, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "SPY,QQQ")

        from cron import monthly_wf
        monthly_wf.run()

        conn = sqlite3.connect(tmp_db)
        tickers = {r[0] for r in conn.execute("SELECT ticker FROM wf_results").fetchall()}
        conn.close()

        assert tickers == {"SPY", "QQQ"}


class TestUpsertBehavior:
    def test_rerun_same_date_overwrites_not_duplicates(self, tmp_db, mock_yf_download, mock_walk_forward, monkeypatch):
        """Second run on same date should update, not insert a duplicate row."""
        monkeypatch.setenv("WF_TICKERS", "SPY")

        from cron import monthly_wf

        monthly_wf.run()
        monthly_wf.run()  # second run on same date

        conn = sqlite3.connect(tmp_db)
        count = conn.execute("SELECT COUNT(*) FROM wf_results WHERE ticker='SPY'").fetchone()[0]
        conn.close()

        assert count == 1, "Re-run on same date must overwrite, not duplicate"

    def test_rerun_same_date_updates_values(self, tmp_db, mock_yf_download, monkeypatch):
        """Updated walk_forward result on re-run should be reflected in the DB."""
        monkeypatch.setenv("WF_TICKERS", "SPY")

        from cron import monthly_wf

        wf_first = _make_wf_result()
        wf_second = WalkForwardResult(
            windows=wf_first.windows,
            avg_return=0.10,
            avg_sharpe=2.0,
            avg_sortino=2.5,
            avg_max_drawdown=-0.02,
            total_trades=8,
            consistency_score=0.5,
        )

        with patch("cron.monthly_wf.walk_forward", return_value=wf_first):
            monthly_wf.run()

        with patch("cron.monthly_wf.walk_forward", return_value=wf_second):
            monthly_wf.run()

        conn = sqlite3.connect(tmp_db)
        row = conn.execute(
            "SELECT consistency_score, total_return FROM wf_results WHERE ticker='SPY'"
        ).fetchone()
        conn.close()

        assert row[0] == pytest.approx(0.5)
        assert row[1] == pytest.approx(0.10)


class TestFailureHandling:
    def test_failed_ticker_returns_exit_code_1(self, tmp_db, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "BADINPUT")

        with patch("cron.monthly_wf.yf.download", side_effect=RuntimeError("network error")):
            from cron import monthly_wf
            rc = monthly_wf.run()

        assert rc == 1

    def test_partial_failure_still_persists_successes(self, tmp_db, monkeypatch):
        monkeypatch.setenv("WF_TICKERS", "SPY,BADINPUT")

        df = _make_ohlcv()
        wf = _make_wf_result()

        def fake_download(ticker, **kwargs):
            if ticker == "SPY":
                return df
            raise RuntimeError("bad ticker")

        with patch("cron.monthly_wf.yf.download", side_effect=fake_download), \
             patch("cron.monthly_wf.walk_forward", return_value=wf):
            from cron import monthly_wf
            rc = monthly_wf.run()

        assert rc == 1

        conn = sqlite3.connect(tmp_db)
        rows = conn.execute("SELECT ticker FROM wf_results").fetchall()
        conn.close()

        assert [r[0] for r in rows] == ["SPY"]
