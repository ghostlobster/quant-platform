"""
tests/test_metrics_exporter.py — tests for risk/metrics_exporter.py.

Every case uses an in-memory ``FakeBroker`` and a per-test SQLite journal
so nothing hits the network or real portfolio_history table. The drawdown
alert test monkeypatches ``alerts.channels.broadcast`` to observe the
call instead of sending anything real.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import journal.trading_journal as jt
from risk import metrics_exporter as me

# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeBroker:
    def __init__(self, equity: float = 100_000.0, positions=None):
        self.equity = equity
        self._positions = positions or []

    def get_account_info(self) -> dict:
        return {"equity": self.equity, "cash": self.equity}

    def get_positions(self) -> list[dict]:
        return list(self._positions)


@pytest.fixture(autouse=True)
def _isolate_alert_cooldown():
    """Reset the module-level drawdown-alert cooldown between cases."""
    me.reset_alert_state()
    yield
    me.reset_alert_state()


@pytest.fixture
def journal_db(tmp_path, monkeypatch):
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    jt.init_journal_table()
    return tmp_path


@pytest.fixture
def isolated_portfolio_history(monkeypatch):
    """Replace the portfolio_history reader with a controllable stub."""
    state = {"series": []}
    monkeypatch.setattr(me, "_portfolio_value_series", lambda: list(state["series"]))
    return state


# ── 1. Snapshot from flat portfolio ──────────────────────────────────────────

def test_snapshot_flat_portfolio(isolated_portfolio_history):
    broker = FakeBroker(equity=50_000.0, positions=[])
    snap = me.compute_risk_snapshot(broker)
    assert snap.equity == 50_000.0
    assert snap.gross_exposure == 0.0
    assert snap.net_exposure == 0.0
    assert snap.daily_pnl == 0.0
    assert snap.var_95 == 0.0
    assert snap.drawdown == 0.0


# ── 2. Gross + net exposure from mixed positions ─────────────────────────────

def test_snapshot_exposure_math(isolated_portfolio_history):
    positions = [
        {"symbol": "AAPL", "market_value": 30_000.0, "unrealized_pl": 1_000.0},
        {"symbol": "TSLA", "market_value": -20_000.0, "unrealized_pl": -500.0},
    ]
    broker = FakeBroker(equity=100_000.0, positions=positions)
    snap = me.compute_risk_snapshot(broker)
    # gross = (|30k| + |20k|) / 100k = 0.5
    assert snap.gross_exposure == pytest.approx(0.5)
    # net = (30k + -20k) / 100k = 0.1
    assert snap.net_exposure == pytest.approx(0.1)
    # daily PnL = realised (0) + unrealised (1000 + -500) = 500
    assert snap.daily_pnl == pytest.approx(500.0)


# ── 3. Drawdown vs running peak ──────────────────────────────────────────────

def test_snapshot_drawdown_from_history(isolated_portfolio_history):
    isolated_portfolio_history["series"] = [100_000.0, 110_000.0, 120_000.0, 95_000.0]
    broker = FakeBroker(equity=90_000.0, positions=[])
    snap = me.compute_risk_snapshot(broker)
    # Peak across history + current = 120k; drawdown = (90 - 120) / 120 = -0.25.
    assert snap.drawdown == pytest.approx(-0.25)


def test_snapshot_drawdown_zero_at_new_high(isolated_portfolio_history):
    isolated_portfolio_history["series"] = [100_000.0, 105_000.0]
    broker = FakeBroker(equity=110_000.0, positions=[])
    snap = me.compute_risk_snapshot(broker)
    assert snap.drawdown == pytest.approx(0.0)


# ── 4. Daily P&L includes realised journal entries ───────────────────────────

def test_daily_pnl_includes_realised(journal_db, isolated_portfolio_history):
    trade_id = jt.log_entry(
        ticker="AAPL", side="BUY", qty=10, price=100.0,
        signal_source="test", regime="", notes="",
    )
    jt.log_exit(trade_id, price=120.0, pnl=200.0, exit_reason="target", notes="")
    broker = FakeBroker(
        equity=100_000.0,
        positions=[{"symbol": "TSLA", "market_value": 5_000.0, "unrealized_pl": 50.0}],
    )
    snap = me.compute_risk_snapshot(broker)
    assert snap.daily_pnl == pytest.approx(250.0)  # 200 realised + 50 unrealised


# ── 5. update_risk_gauges is a no-op when prom-client missing ────────────────

def test_update_gauges_is_safe_without_prometheus(monkeypatch):
    # Replace each gauge with a no-op; ensure update_risk_gauges does not raise.
    monkeypatch.setattr(me, "RISK_EQUITY", me._NoopMetric())
    monkeypatch.setattr(me, "RISK_GROSS_EXPOSURE", me._NoopMetric())
    monkeypatch.setattr(me, "RISK_NET_EXPOSURE", me._NoopMetric())
    monkeypatch.setattr(me, "RISK_DAILY_PNL", me._NoopMetric())
    monkeypatch.setattr(me, "RISK_VAR_95", me._NoopMetric())
    monkeypatch.setattr(me, "RISK_DRAWDOWN", me._NoopMetric())
    me.update_risk_gauges(me.RiskSnapshot(1.0, 0.5, 0.2, 42.0, 0.01, -0.05))


# ── 6. Drawdown alert breaches threshold once per cooldown ───────────────────

def test_maybe_alert_drawdown_fires_once_within_cooldown(monkeypatch):
    received: list[tuple[str, str]] = []

    def _fake_broadcast(subject: str, body: str, channels=None):
        received.append((subject, body))
        return {"ok": 1}

    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.10")
    monkeypatch.setenv("RISK_ALERT_COOLDOWN", "3600")
    monkeypatch.setattr("alerts.channels.broadcast", _fake_broadcast)

    breach = me.RiskSnapshot(100_000.0, 1.0, 0.5, -5_000.0, 0.02, -0.15)
    assert me.maybe_alert_drawdown(breach, now=1000.0) is True
    # Second call within the cooldown window is suppressed.
    assert me.maybe_alert_drawdown(breach, now=1500.0) is False
    assert len(received) == 1


def test_maybe_alert_drawdown_refires_after_cooldown(monkeypatch):
    received: list[tuple[str, str]] = []

    def _fake_broadcast(subject: str, body: str, channels=None):
        received.append((subject, body))
        return {"ok": 1}

    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.10")
    monkeypatch.setenv("RISK_ALERT_COOLDOWN", "1000")
    monkeypatch.setattr("alerts.channels.broadcast", _fake_broadcast)

    breach = me.RiskSnapshot(100_000.0, 1.0, 0.5, -5_000.0, 0.02, -0.15)
    assert me.maybe_alert_drawdown(breach, now=0.0) is True
    assert me.maybe_alert_drawdown(breach, now=1500.0) is True
    assert len(received) == 2


# ── 7. No alert when threshold not breached ──────────────────────────────────

def test_no_alert_when_under_threshold(monkeypatch):
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.10")
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "alerts.channels.broadcast",
        lambda s, b, channels=None: sent.append((s, b)) or {"ok": 1},
    )
    snap = me.RiskSnapshot(100_000.0, 1.0, 0.5, -500.0, 0.02, -0.05)  # -5% > -10%
    assert me.maybe_alert_drawdown(snap, now=0.0) is False
    assert sent == []


# ── 8. No alert when threshold unset ─────────────────────────────────────────

def test_no_alert_when_threshold_unset(monkeypatch):
    monkeypatch.delenv("MAX_DRAWDOWN_PCT", raising=False)
    monkeypatch.setattr(
        "alerts.channels.broadcast",
        lambda s, b, channels=None: (_ for _ in ()).throw(AssertionError("should not call")),
    )
    breach = me.RiskSnapshot(100_000.0, 1.0, 0.5, -50_000.0, 0.02, -0.5)
    assert me.maybe_alert_drawdown(breach, now=0.0) is False


# ── 9. risk_exporter_job returns a dict including the snapshot and flag ──────

def test_risk_exporter_job_returns_snapshot(monkeypatch, isolated_portfolio_history):
    broker = FakeBroker(equity=100_000.0, positions=[])
    monkeypatch.delenv("MAX_DRAWDOWN_PCT", raising=False)
    payload = me.risk_exporter_job(broker)
    assert set(payload) >= {
        "equity", "gross_exposure", "net_exposure", "daily_pnl",
        "var_95", "drawdown", "alerted",
    }
    assert payload["equity"] == 100_000.0
    assert payload["alerted"] is False


# ── 10. broker.get_account_info failure collapses to zero snapshot ───────────

def test_broker_failure_collapses_to_zero(isolated_portfolio_history):
    class ExplodingBroker:
        def get_account_info(self):
            raise RuntimeError("network error")

        def get_positions(self):
            return []

    snap = me.compute_risk_snapshot(ExplodingBroker())
    assert snap == me.RiskSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
