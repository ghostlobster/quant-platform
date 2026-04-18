"""Tests for scheduler.alerts:knowledge_health_job + scheduler bootstrap (#116)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton(monkeypatch):
    """Ensure each test gets a fresh agent singleton — the module-level cache
    would otherwise share alert timestamps across unrelated tests."""
    import scheduler.alerts as alerts

    monkeypatch.setattr(alerts, "_knowledge_agent_singleton", None)


def _agent_returning(recommendation: str, signal: str = "bullish"):
    sig = MagicMock()
    sig.signal = signal
    sig.confidence = 0.8
    sig.reasoning = f"mock verdict: {recommendation}"
    sig.metadata = {"recommendation": recommendation, "regime": "trending_bull"}
    agent = MagicMock()
    agent.run.return_value = sig
    return agent


def test_knowledge_health_job_logs_info_on_fresh(monkeypatch):
    import scheduler.alerts as alerts

    fake_agent = _agent_returning("fresh", signal="bullish")
    monkeypatch.setattr(alerts, "_knowledge_agent_singleton", fake_agent)

    with patch.object(alerts.logger, "info") as info_log, \
         patch.object(alerts.logger, "warning") as warn_log:
        result = alerts.knowledge_health_job()

    assert result["recommendation"] == "fresh"
    info_log.assert_called()
    warn_log.assert_not_called()
    fake_agent.run.assert_called_once_with({})


def test_knowledge_health_job_logs_warning_on_retrain(monkeypatch):
    import scheduler.alerts as alerts

    fake_agent = _agent_returning("retrain", signal="bearish")
    monkeypatch.setattr(alerts, "_knowledge_agent_singleton", fake_agent)

    with patch.object(alerts.logger, "warning") as warn_log:
        result = alerts.knowledge_health_job()

    assert result["recommendation"] == "retrain"
    warn_log.assert_called()


def test_knowledge_health_job_monitor_uses_info(monkeypatch):
    import scheduler.alerts as alerts

    fake_agent = _agent_returning("monitor", signal="neutral")
    monkeypatch.setattr(alerts, "_knowledge_agent_singleton", fake_agent)

    with patch.object(alerts.logger, "info") as info_log, \
         patch.object(alerts.logger, "warning") as warn_log:
        alerts.knowledge_health_job()

    info_log.assert_called()
    warn_log.assert_not_called()


def test_agent_singleton_reused_across_calls(monkeypatch):
    # The shared singleton is what makes the 24h alert stamp survive across
    # scheduled runs — assert the factory doesn't rebuild it every call.
    import scheduler.alerts as alerts

    calls = []

    class _FakeAgent:
        def run(self, context):
            calls.append(context)
            return _agent_returning("fresh").run({})

    monkeypatch.setattr(
        "agents.knowledge_agent.KnowledgeAdaptionAgent", _FakeAgent,
    )
    # First call constructs the singleton, subsequent calls reuse it.
    alerts.knowledge_health_job()
    first = alerts._knowledge_agent_singleton
    alerts.knowledge_health_job()
    second = alerts._knowledge_agent_singleton

    assert first is second
    assert len(calls) == 2


def test_scheduler_disabled_via_env(monkeypatch):
    import scheduler.alerts as alerts

    monkeypatch.setenv("KNOWLEDGE_HEALTH_ENABLED", "0")
    assert alerts.start_knowledge_health_scheduler() is None


def test_scheduler_registers_hourly_job(monkeypatch):
    import scheduler.alerts as alerts

    monkeypatch.delenv("KNOWLEDGE_HEALTH_ENABLED", raising=False)
    monkeypatch.delenv("KNOWLEDGE_HEALTH_CRON", raising=False)
    scheduler = alerts.start_knowledge_health_scheduler(paused=True)
    try:
        assert scheduler is not None
        jobs = {job.id: job for job in scheduler.get_jobs()}
        # Expected jobs: health check (#116) + live IC backfill (#115).
        assert set(jobs) == {"knowledge_health_job", "live_ic_backfill_job"}
        health_job = jobs["knowledge_health_job"]
        # Default cron is top-of-hour; confirm the trigger reflects that.
        assert (
            "hour='*'" in str(health_job.trigger)
            or "minute='0'" in str(health_job.trigger)
        )
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_honours_cron_expr(monkeypatch):
    import scheduler.alerts as alerts

    scheduler = alerts.start_knowledge_health_scheduler(
        cron_expr="*/15 * * * *", paused=True,
    )
    try:
        assert scheduler is not None
        jobs = {job.id: job for job in scheduler.get_jobs()}
        assert "knowledge_health_job" in jobs
        assert "*/15" in str(jobs["knowledge_health_job"].trigger)
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_reads_cron_from_env(monkeypatch):
    import scheduler.alerts as alerts

    monkeypatch.setenv("KNOWLEDGE_HEALTH_CRON", "0 */2 * * *")
    scheduler = alerts.start_knowledge_health_scheduler(paused=True)
    try:
        assert scheduler is not None
        jobs = {job.id: job for job in scheduler.get_jobs()}
        assert "*/2" in str(jobs["knowledge_health_job"].trigger)
    finally:
        scheduler.shutdown(wait=False)


def test_scheduler_registers_backfill_job(monkeypatch):
    """New #115 job: daily live_ic backfill registered on the same scheduler."""
    import scheduler.alerts as alerts

    monkeypatch.setenv("LIVE_IC_BACKFILL_CRON", "17 3 * * *")
    scheduler = alerts.start_knowledge_health_scheduler(paused=True)
    try:
        assert scheduler is not None
        jobs = {job.id: job for job in scheduler.get_jobs()}
        assert "live_ic_backfill_job" in jobs
        assert "hour='3'" in str(jobs["live_ic_backfill_job"].trigger)
        assert "minute='17'" in str(jobs["live_ic_backfill_job"].trigger)
    finally:
        scheduler.shutdown(wait=False)
