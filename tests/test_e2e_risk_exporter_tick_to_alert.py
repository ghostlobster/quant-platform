"""
tests/test_e2e_risk_exporter_tick_to_alert.py — risk-exporter scheduler tick.

End-to-end verification of the P1.2 risk-dashboard chain: a single call
to ``risk_exporter_job(broker)`` resolves equity + positions from the
broker, daily P&L from the journal, drawdown from
``portfolio_value_series``, updates the Prometheus gauges, and broadcasts
a drawdown alert through ``alerts.channels.broadcast`` once the breach
crosses the configured threshold.

Network is stubbed at the boundary: ``alerts.channels.broadcast`` and
``risk.metrics_exporter._portfolio_value_series`` are monkeypatched.
The scheduler tick itself is exercised via the public
``risk_exporter_job`` entry-point.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import journal.trading_journal as jt
from risk import metrics_exporter as me

# E2EFakeBroker comes from tests/conftest.py — same shape as the
# previous local FakeBroker but reused by every e2e file.
from tests.conftest import E2EFakeBroker as FakeBroker  # noqa: E402

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def _reset_alert_state():
    me.reset_alert_state()
    yield
    me.reset_alert_state()


def test_drawdown_breach_fires_alert_then_cooldown(monkeypatch, e2e_journal_db):
    """Tick 1 breaches the threshold and broadcasts; tick 2 (within
    cooldown) is silent; tick 3 after cooldown re-broadcasts."""
    # Seed journal with -$12k realised PnL today.
    trade_id = jt.log_entry(
        ticker="SPY", side="BUY", qty=10, price=100.0,
        signal_source="e2e_test", regime="", notes="",
    )
    jt.log_exit(trade_id, price=-100.0, pnl=-12_000.0, exit_reason="stop", notes="")

    # Portfolio history climbed to $120k peak before the dip.
    monkeypatch.setattr(
        me, "_portfolio_value_series",
        lambda: [100_000.0, 110_000.0, 120_000.0],
    )

    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.10")
    monkeypatch.setenv("RISK_ALERT_COOLDOWN", "100")  # 100s for clearer timing

    received: list[tuple[str, str]] = []

    def _fake_broadcast(subject, body, channels=None):
        received.append((subject, body))
        return {"ok": 1}

    monkeypatch.setattr("alerts.channels.broadcast", _fake_broadcast)

    broker = FakeBroker(equity=100_000.0, positions=[])

    # Tick 1: drawdown = (100 - 120) / 120 = -0.166..., below the -0.10 cap.
    snap = me.compute_risk_snapshot(broker)
    assert snap.drawdown == pytest.approx(-1 / 6)
    me.update_risk_gauges(snap)
    fired = me.maybe_alert_drawdown(snap, now=1000.0)
    assert fired is True
    assert len(received) == 1
    subject, body = received[0]
    assert subject == "Drawdown alert"
    assert "16.67" in body or "-0.17" in body or "-0.16" in body  # rendered drawdown

    # Tick 2 (within cooldown): no second broadcast.
    silenced = me.maybe_alert_drawdown(snap, now=1050.0)
    assert silenced is False
    assert len(received) == 1

    # Tick 3 (cooldown elapsed): broadcasts again.
    refired = me.maybe_alert_drawdown(snap, now=1200.0)
    assert refired is True
    assert len(received) == 2

    # Bus publish (P1.9) — every breach must also land on the risk stream.
    from bus.event_bus import get_event_bus, reset_event_bus
    from bus.events import EventType, Stream

    bus_rows = list(get_event_bus().replay(Stream.RISK))
    assert len(bus_rows) >= 2
    assert all(evt.event_type == EventType.RISK_BREACH for _, evt in bus_rows)
    assert bus_rows[0][1].payload["drawdown"] == pytest.approx(-1 / 6)
    reset_event_bus()


def test_no_alert_when_within_threshold(monkeypatch, e2e_journal_db):
    """Drawdown above the threshold (e.g. -3%) does not broadcast."""
    monkeypatch.setattr(
        me, "_portfolio_value_series",
        lambda: [100_000.0, 102_000.0],
    )
    monkeypatch.setenv("MAX_DRAWDOWN_PCT", "0.10")

    sent: list = []
    monkeypatch.setattr(
        "alerts.channels.broadcast",
        lambda s, b, channels=None: sent.append((s, b)),
    )

    broker = FakeBroker(equity=99_000.0)
    snap = me.compute_risk_snapshot(broker)
    # drawdown = (99 - 102) / 102 ≈ -0.029, above -0.10
    assert snap.drawdown > -0.05
    me.update_risk_gauges(snap)
    assert me.maybe_alert_drawdown(snap, now=0.0) is False
    assert sent == []


def test_risk_exporter_job_returns_payload(monkeypatch, e2e_journal_db):
    """The scheduler-callable entry-point returns a dict with every gauge
    field plus the `alerted` flag."""
    monkeypatch.setattr(me, "_portfolio_value_series", lambda: [])
    monkeypatch.delenv("MAX_DRAWDOWN_PCT", raising=False)

    broker = FakeBroker(
        equity=100_000.0,
        positions=[{"symbol": "SPY", "market_value": 30_000.0, "unrealized_pl": 250.0}],
    )
    payload = me.risk_exporter_job(broker)
    assert set(payload).issuperset(
        {"equity", "gross_exposure", "net_exposure", "daily_pnl",
         "var_95", "drawdown", "alerted"},
    )
    assert payload["equity"] == 100_000.0
    assert payload["gross_exposure"] == pytest.approx(0.3)
    assert payload["net_exposure"] == pytest.approx(0.3)
    assert payload["daily_pnl"] == pytest.approx(250.0)
    assert payload["alerted"] is False
