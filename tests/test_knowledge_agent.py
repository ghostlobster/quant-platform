"""Tests for agents/knowledge_agent.py (KnowledgeAdaptionAgent)."""
from __future__ import annotations

import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgentSignal
from agents.knowledge_agent import (
    KnowledgeAdaptionAgent,
    _safe_ratio,
    recommendation_multiplier,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_pickle(path: Path, payload, age_seconds: float = 0.0) -> Path:
    """Write a pickle file and backdate its mtime by ``age_seconds``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    if age_seconds:
        now = time.time()
        os.utime(path, (now - age_seconds, now - age_seconds))
    return path


@pytest.fixture
def model_paths(tmp_path, monkeypatch):
    """Return (baseline_path, regime_path) with env vars pointing at tmp_path."""
    baseline = tmp_path / "lgbm_alpha.pkl"
    regime = tmp_path / "lgbm_regime_models.pkl"
    monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", str(baseline))
    monkeypatch.setenv("LGBM_REGIME_MODELS_PATH", str(regime))
    return baseline, regime


@pytest.fixture
def isolate_metadata(monkeypatch):
    """Force _read_trained_ic to return None unless a test overrides it."""
    monkeypatch.setattr(
        "agents.knowledge_agent._read_trained_ic",
        lambda model_name: None,
    )
    monkeypatch.setattr(
        KnowledgeAdaptionAgent, "_metadata_reader",
        staticmethod(lambda model_name: None),
    )


# ── _safe_ratio ──────────────────────────────────────────────────────────────

def test_safe_ratio_handles_zero_and_sign_flip():
    assert _safe_ratio(None, 0.05) is None
    assert _safe_ratio(0.05, None) is None
    assert _safe_ratio(0.05, 0.0) is None
    # Sign flip → maximum decay (0.0)
    assert _safe_ratio(-0.02, 0.05) == 0.0
    assert _safe_ratio(0.02, -0.05) == 0.0
    # Normal ratio
    assert _safe_ratio(0.02, 0.04) == 0.5


def test_recommendation_multiplier_exposes_table():
    assert recommendation_multiplier("fresh") == 1.0
    assert recommendation_multiplier("monitor") == 0.7
    assert recommendation_multiplier("retrain") == 0.4
    # Unknown falls back to safe 1.0
    assert recommendation_multiplier("garbage") == 1.0


# ── Core verdict ladder ──────────────────────────────────────────────────────

def test_missing_pickles_bearish_retrain(model_paths, isolate_metadata):
    # No files exist yet
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert isinstance(sig, AgentSignal)
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["baseline_age_days"] is None
    assert "missing" in sig.reasoning.lower()


def test_fresh_and_covered_bullish(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)  # seconds old
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.signal == "bullish"
    assert sig.metadata["recommendation"] == "fresh"
    assert sig.metadata["regime_coverage"] == sorted(
        ["trending_bull", "trending_bear", "mean_reverting", "high_vol"]
    )


def test_stale_baseline_bearish_retrain(model_paths, isolate_metadata):
    baseline, regime = model_paths
    days = 60 * 86400
    _write_pickle(baseline, {"model": object()}, age_seconds=days)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}},
        age_seconds=days,
    )
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["baseline_age_days"] >= 45


def test_monitor_window_neutral(model_paths, isolate_metadata):
    baseline, regime = model_paths
    # 40 days is inside [35, 45] → monitor
    age = 40 * 86400
    _write_pickle(baseline, {"model": object()}, age_seconds=age)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}},
        age_seconds=age,
    )
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.signal == "neutral"
    assert sig.metadata["recommendation"] == "monitor"


def test_regime_not_covered_bearish(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1}},
        age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({"regime": "high_vol"})
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert "high_vol" in sig.reasoning


def test_ic_degradation_bearish(model_paths, monkeypatch):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    monkeypatch.setattr(
        KnowledgeAdaptionAgent, "_metadata_reader",
        staticmethod(lambda model_name: 0.05),  # trained IC = 0.05
    )
    # live IC = 0.01 → ratio 0.2 → bearish / retrain
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "live_ic": 0.01}
    )
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["ic_ratio"] == pytest.approx(0.2)


def test_ic_degradation_monitor(model_paths, monkeypatch):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    monkeypatch.setattr(
        KnowledgeAdaptionAgent, "_metadata_reader",
        staticmethod(lambda model_name: 0.05),
    )
    # live 0.033 → ratio 0.66 → monitor
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "live_ic": 0.033}
    )
    assert sig.signal == "neutral"
    assert sig.metadata["recommendation"] == "monitor"


def test_legacy_pickle_format_accepted(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    # Legacy: bare dict of {regime: model}
    _write_pickle(
        regime, {"trending_bull": 1, "trending_bear": 1},
        age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.metadata["regime_coverage"] == ["trending_bear", "trending_bull"]


def test_context_regime_overrides_live_lookup(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(regime, {"models": {"trending_bull": 1}}, age_seconds=60)
    # get_cached_live_regime must NOT be called when context provides regime.
    with patch("analysis.regime.get_cached_live_regime") as fake:
        KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    fake.assert_not_called()


def test_unexpected_exception_returns_neutral(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("kapow")
    monkeypatch.setattr("agents.knowledge_agent._safe_pickle_path", boom)
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.signal == "neutral"
    assert sig.confidence == 0.3
    assert sig.reasoning == "Knowledge state unavailable"
    assert sig.metadata["recommendation"] == "monitor"


# ── Caching of pickle reads ──────────────────────────────────────────────────

def test_regime_coverage_cached_within_ttl(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(regime, {"models": {"trending_bull": 1}}, age_seconds=60)

    counter = MagicMock(return_value=["trending_bull"])
    agent = KnowledgeAdaptionAgent()
    # Inject a counting loader via the cache.
    agent._cache._coverage = (None, 0.0, [])
    agent._cache.coverage(regime, counter)
    agent._cache.coverage(regime, counter)
    # Second call within TTL and with unchanged mtime should not re-invoke loader.
    assert counter.call_count == 1


# ── Alert throttling ─────────────────────────────────────────────────────────

def test_retrain_alert_fires_once_within_cooldown(model_paths, isolate_metadata, monkeypatch):
    # No pickles → retrain verdict on every call.
    channel = MagicMock()
    channel.send = MagicMock(return_value=True)
    monkeypatch.setattr(
        "providers.alert.get_alert_channel", lambda *a, **kw: channel,
    )
    monkeypatch.setenv("KNOWLEDGE_ALERT_COOLDOWN", "3600")

    agent = KnowledgeAdaptionAgent()
    agent.run({"regime": "trending_bull"})
    agent.run({"regime": "trending_bull"})
    assert channel.send.call_count == 1


def test_retrain_alert_refires_after_cooldown(model_paths, isolate_metadata, monkeypatch):
    channel = MagicMock()
    channel.send = MagicMock(return_value=True)
    monkeypatch.setattr(
        "providers.alert.get_alert_channel", lambda *a, **kw: channel,
    )
    monkeypatch.setenv("KNOWLEDGE_ALERT_COOLDOWN", "100")

    clock = [1_000_000.0]
    agent = KnowledgeAdaptionAgent()
    agent._clock = staticmethod(lambda: clock[0])  # type: ignore[method-assign]

    agent.run({"regime": "trending_bull"})
    clock[0] += 200.0  # past cooldown
    agent.run({"regime": "trending_bull"})
    assert channel.send.call_count == 2


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_exits_nonzero_on_retrain(tmp_path, monkeypatch):
    env = dict(os.environ)
    env["LGBM_ALPHA_MODEL_PATH"] = str(tmp_path / "missing.pkl")
    env["LGBM_REGIME_MODELS_PATH"] = str(tmp_path / "missing_regime.pkl")
    # Ensure the subprocess reaches this repo (not the user's cwd).
    result = subprocess.run(
        [sys.executable, "-m", "agents.knowledge_agent"],
        capture_output=True, text=True, env=env,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 1
    assert "retrain" in result.stdout


# ── Pickle path confinement (#124) ────────────────────────────────────────────

from agents.knowledge_agent import (  # noqa: E402  (after CLI tests is fine)
    _confine_pickle_path,
    _safe_pickle_path,
)


class TestConfinePicklePath:
    def test_accepts_repo_relative_path(self):
        # models/lgbm_alpha.pkl under the repo root
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / "models" / "lgbm_alpha.pkl"
        resolved = _confine_pickle_path(candidate)
        assert resolved == candidate.resolve()

    def test_accepts_tempdir_path(self, tmp_path):
        # tmp_path lives under the system temp dir, explicitly allowed for tests
        candidate = tmp_path / "fixture.pkl"
        candidate.write_bytes(b"")
        resolved = _confine_pickle_path(candidate)
        assert resolved == candidate.resolve()

    def test_rejects_etc_passwd(self):
        with pytest.raises(ValueError, match="outside allowed roots"):
            _confine_pickle_path("/etc/passwd")

    def test_rejects_traversal_out_of_repo(self):
        # A repo-rooted path that traverses up past the repo escapes to a
        # parent directory that is neither repo-root nor tmpdir.
        repo_root = Path(__file__).resolve().parent.parent
        traversal = repo_root / ".." / ".." / ".." / "etc" / "passwd"
        with pytest.raises(ValueError, match="outside allowed roots"):
            _confine_pickle_path(traversal)


class TestSafePicklePath:
    def test_env_override_under_tempdir_is_allowed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", str(tmp_path / "x.pkl"))
        resolved = _safe_pickle_path("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")
        assert resolved.parent == tmp_path.resolve()

    def test_env_override_outside_allowed_roots_raises(self, monkeypatch):
        monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", "/etc/passwd")
        with pytest.raises(ValueError, match="outside allowed roots"):
            _safe_pickle_path("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")

    def test_default_repo_relative_is_allowed(self, monkeypatch):
        monkeypatch.delenv("LGBM_ALPHA_MODEL_PATH", raising=False)
        resolved = _safe_pickle_path("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")
        repo_root = Path(__file__).resolve().parent.parent
        assert resolved == (repo_root / "models" / "lgbm_alpha.pkl").resolve()


def test_unsafe_env_path_fails_closed(monkeypatch, isolate_metadata):
    # When LGBM_ALPHA_MODEL_PATH points outside the allowed roots, the agent
    # must refuse to load and return a retrain verdict rather than silently
    # loading a potentially-hostile pickle.
    monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", "/etc/passwd")
    monkeypatch.setenv("LGBM_REGIME_MODELS_PATH", "/etc/shadow")
    sig = KnowledgeAdaptionAgent().run({"regime": "trending_bull"})
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert "untrusted path" in sig.reasoning


# ── at_risk demotion (#124) ───────────────────────────────────────────────────

def test_at_risk_demotes_fresh_to_monitor(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    # Would normally verdict "fresh"; at_risk=True demotes to monitor.
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "at_risk": True}
    )
    assert sig.signal == "neutral"
    assert sig.metadata["recommendation"] == "monitor"
    assert sig.metadata["at_risk"] is True
    assert "boundary" in sig.reasoning


def test_at_risk_false_keeps_fresh(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "at_risk": False}
    )
    assert sig.signal == "bullish"
    assert sig.metadata["recommendation"] == "fresh"
    assert sig.metadata["at_risk"] is False


def test_at_risk_does_not_override_retrain(model_paths, isolate_metadata):
    # With a stale baseline the verdict should stay retrain even if at_risk=True
    baseline, regime = model_paths
    days = 60 * 86400
    _write_pickle(baseline, {"model": object()}, age_seconds=days)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=days,
    )
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "at_risk": True}
    )
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"


# ── IC plateau demotion (#122) ────────────────────────────────────────────────

def test_plateau_demotes_fresh_to_monitor(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    # Would normally be fresh; plateau demotes to monitor.
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "plateau_detected": True}
    )
    assert sig.signal == "neutral"
    assert sig.metadata["recommendation"] == "monitor"
    assert sig.metadata["plateau_detected"] is True
    assert "plateau" in sig.reasoning.lower()


def test_plateau_false_keeps_fresh(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "plateau_detected": False}
    )
    assert sig.signal == "bullish"
    assert sig.metadata["recommendation"] == "fresh"
    assert sig.metadata["plateau_detected"] is False


def test_plateau_does_not_override_retrain(model_paths, isolate_metadata):
    # Stale baseline trumps plateau — still a hard retrain
    baseline, regime = model_paths
    days = 60 * 86400
    _write_pickle(baseline, {"model": object()}, age_seconds=days)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=days,
    )
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "plateau_detected": True}
    )
    assert sig.metadata["recommendation"] == "retrain"


# ── Opt-in auto-retrain (#119) ────────────────────────────────────────────────

def _make_agent_with_fake_stamp(popen):
    """Return (agent, stamp_dict) wired to an in-memory stamp store."""
    stamp: dict[str, float] = {}

    def _reader(name: str) -> float:
        return stamp.get(name, 0.0)

    def _writer(name: str, now: float) -> None:
        stamp[name] = now

    agent = KnowledgeAdaptionAgent()
    agent._popen = staticmethod(popen)  # type: ignore[method-assign]
    agent._retrain_reader = staticmethod(_reader)  # type: ignore[method-assign]
    agent._retrain_writer = staticmethod(_writer)  # type: ignore[method-assign]
    return agent, stamp


def test_auto_retrain_disabled_by_default(model_paths, isolate_metadata, monkeypatch):
    monkeypatch.delenv("KNOWLEDGE_AUTO_RETRAIN", raising=False)
    popen = MagicMock()
    agent, _ = _make_agent_with_fake_stamp(popen)
    # Missing pickles → retrain verdict.
    sig = agent.run({"regime": "trending_bull"})
    assert sig.metadata["recommendation"] == "retrain"
    popen.assert_not_called()
    assert "auto_retrain" not in sig.metadata


def test_auto_retrain_fires_once_within_cooldown(
    model_paths, isolate_metadata, monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    popen = MagicMock()
    popen.return_value = MagicMock(pid=12345, wait=MagicMock(return_value=0))
    agent, stamp = _make_agent_with_fake_stamp(popen)

    clock = [1_000_000.0]
    agent._clock = staticmethod(lambda: clock[0])  # type: ignore[method-assign]

    sig1 = agent.run({"regime": "trending_bull"})
    assert sig1.metadata["auto_retrain"]["auto_retrain"] == "launched"
    assert sig1.metadata["auto_retrain"]["pid"] == 12345

    # Second call within cooldown: popen must not fire again.
    clock[0] += 60.0
    sig2 = agent.run({"regime": "trending_bull"})
    assert popen.call_count == 1
    assert sig2.metadata["auto_retrain"]["auto_retrain"] == "throttled"
    assert sig2.metadata["auto_retrain"]["seconds_until_next"] > 0


def test_auto_retrain_refires_after_cooldown(
    model_paths, isolate_metadata, monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    monkeypatch.setenv("KNOWLEDGE_RETRAIN_COOLDOWN", "3600")
    popen = MagicMock()
    popen.return_value = MagicMock(pid=42, wait=MagicMock(return_value=0))
    agent, _ = _make_agent_with_fake_stamp(popen)

    clock = [1_000_000.0]
    agent._clock = staticmethod(lambda: clock[0])  # type: ignore[method-assign]

    agent.run({"regime": "trending_bull"})
    clock[0] += 7200.0  # 2h, past the 1h cooldown
    agent.run({"regime": "trending_bull"})
    assert popen.call_count == 2


def test_auto_retrain_not_fired_on_fresh_or_monitor(
    model_paths, isolate_metadata, monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    popen = MagicMock()
    agent, _ = _make_agent_with_fake_stamp(popen)

    # Fresh pickles → fresh verdict, no subprocess.
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = agent.run({"regime": "trending_bull"})
    assert sig.metadata["recommendation"] == "fresh"
    popen.assert_not_called()


def test_auto_retrain_subprocess_error_does_not_crash_agent(
    model_paths, isolate_metadata, monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    popen = MagicMock(side_effect=OSError("fork failed"))
    agent, stamp = _make_agent_with_fake_stamp(popen)

    sig = agent.run({"regime": "trending_bull"})
    # Agent still returns a valid AgentSignal.
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["auto_retrain"]["auto_retrain"] == "failed"
    assert "fork failed" in sig.metadata["auto_retrain"]["error"]
    # Stamp must not be written on failure — otherwise retries are throttled.
    assert stamp == {}


def test_auto_retrain_launch_alert_subject_differs(
    model_paths, isolate_metadata, monkeypatch,
):
    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    popen = MagicMock()
    popen.return_value = MagicMock(pid=99, wait=MagicMock(return_value=0))
    channel = MagicMock()
    channel.send = MagicMock(return_value=True)
    monkeypatch.setattr(
        "providers.alert.get_alert_channel", lambda *a, **kw: channel,
    )

    agent, _ = _make_agent_with_fake_stamp(popen)
    agent.run({"regime": "trending_bull"})

    # Two alerts fire: the stale-model alert + the auto-retrain launch alert.
    # They must use different subjects so operators can distinguish them.
    subjects = [call.args[0] for call in channel.send.call_args_list]
    assert any("retrain recommended" in s for s in subjects)
    assert any("auto-retrain launched" in s for s in subjects)


def _isolate_stamp_db(monkeypatch, tmp_path):
    """Point data.db at a fresh SQLite file in tmp_path and re-init schema."""
    db_file = tmp_path / "quant-stamp-test.db"
    import data.db as _db_mod
    monkeypatch.setattr(_db_mod, "_DB_PATH", db_file)
    _db_mod.init_db()


def test_auto_retrain_stamp_persists_across_instances(
    model_paths, isolate_metadata, monkeypatch, tmp_path,
):
    # Use a real SQLite DB so the stamp survives across
    # KnowledgeAdaptionAgent() instantiations — exactly the case the
    # SQLite-backed dedup stamp is designed to cover.
    _isolate_stamp_db(monkeypatch, tmp_path)

    monkeypatch.setenv("KNOWLEDGE_AUTO_RETRAIN", "1")
    popen = MagicMock()
    popen.return_value = MagicMock(pid=7, wait=MagicMock(return_value=0))

    clock = [2_000_000.0]

    agent1 = KnowledgeAdaptionAgent()
    agent1._popen = staticmethod(popen)  # type: ignore[method-assign]
    agent1._clock = staticmethod(lambda: clock[0])  # type: ignore[method-assign]
    agent1.run({"regime": "trending_bull"})
    assert popen.call_count == 1

    agent2 = KnowledgeAdaptionAgent()
    agent2._popen = staticmethod(popen)  # type: ignore[method-assign]
    agent2._clock = staticmethod(lambda: clock[0] + 60.0)  # inside cooldown
    sig = agent2.run({"regime": "trending_bull"})

    # Second agent must see the stamp from the first and NOT fire popen again.
    assert popen.call_count == 1
    assert sig.metadata["auto_retrain"]["auto_retrain"] == "throttled"


def test_stamp_read_write_roundtrip(tmp_path, monkeypatch):
    # Bypass the agent — exercise _read_stamp / _write_stamp directly.
    _isolate_stamp_db(monkeypatch, tmp_path)
    from agents.knowledge_agent import _read_stamp, _write_stamp

    assert _read_stamp("nonexistent") == 0.0
    _write_stamp("retrain_fired_at", 1_234_567.89)
    assert _read_stamp("retrain_fired_at") == 1_234_567.89
    # Upsert semantics: second write overwrites.
    _write_stamp("retrain_fired_at", 9_999_999.0)
    assert _read_stamp("retrain_fired_at") == 9_999_999.0


# ── Live-IC integration (#115) ────────────────────────────────────────────────

def test_ic_degradation_fires_via_live_ic_module(
    tmp_path, monkeypatch, model_paths,
):
    """End-to-end guard: collapsed IC computed by analysis.live_ic flips the
    agent's verdict to retrain via the existing IC-degradation branch."""
    # Isolate the DB so this test's rows don't leak across the suite.
    _isolate_stamp_db(monkeypatch, tmp_path)

    # Fresh pickles — without live IC the verdict would be `fresh`.
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )

    # Seed live_predictions with 20 perfectly anti-correlated rows so the
    # rolling IC comes back at -1.0 — well past the _IC_DROP_BEARISH threshold.
    from data.db import get_connection
    conn = get_connection()
    base_ts = 1_700_000_000.0
    with conn:
        for i in range(20):
            conn.execute(
                "INSERT INTO live_predictions (ts, ticker, model_name, "
                "score, horizon_d, realized) VALUES (?, ?, ?, ?, ?, ?)",
                (base_ts + i, f"T{i:02d}", "lgbm_alpha",
                 float(i), 5, -float(i)),
            )
    conn.close()

    # Arm the agent: trained IC 0.05 so live/trained ratio is negative →
    # clamped to 0.0 by _safe_ratio, well below _IC_DROP_BEARISH (0.5).
    monkeypatch.setattr(
        KnowledgeAdaptionAgent, "_metadata_reader",
        staticmethod(lambda model_name: 0.05),
    )

    # Pull the rolling IC. Use window=20 to match the seeded row count —
    # the default window=60 requires 30 warm-up rows, which is overkill
    # for this unit-level integration test.
    import analysis.live_ic as live_ic_mod
    live_ic_mod._ic_cache.clear()
    live_ic = live_ic_mod.rolling_live_ic("lgbm_alpha", window=20)
    assert live_ic is not None
    assert live_ic == pytest.approx(-1.0)

    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "live_ic": live_ic,
    })
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert "IC" in sig.reasoning


# ── Covariate-shift drift rung (#118) ─────────────────────────────────────────

def test_drift_retrain_forces_retrain_verdict(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "drift": {
            "level": "retrain", "max_psi": 0.30,
            "drifted_features": ["ret_5d", "realised_vol_21d"],
        },
    })
    assert sig.signal == "bearish"
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["drift_level"] == "retrain"
    assert sig.metadata["drift_max_psi"] == pytest.approx(0.30)
    assert sig.metadata["drifted_features"] == ["ret_5d", "realised_vol_21d"]
    assert "covariate shift" in sig.reasoning


def test_drift_monitor_demotes_fresh_to_monitor(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "drift": {
            "level": "monitor", "max_psi": 0.15,
            "drifted_features": ["ret_5d"],
        },
    })
    assert sig.signal == "neutral"
    assert sig.metadata["recommendation"] == "monitor"
    assert sig.metadata["drift_level"] == "monitor"
    assert "covariate shift" in sig.reasoning


def test_drift_none_keeps_fresh(model_paths, isolate_metadata):
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "drift": {
            "level": "none", "max_psi": 0.03, "drifted_features": [],
        },
    })
    assert sig.signal == "bullish"
    assert sig.metadata["recommendation"] == "fresh"
    assert sig.metadata["drift_level"] == "none"


def test_drift_does_not_override_stale_retrain(model_paths, isolate_metadata):
    """Stale baseline trumps a merely-monitor-level drift."""
    baseline, regime = model_paths
    days = 60 * 86400
    _write_pickle(baseline, {"model": object()}, age_seconds=days)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=days,
    )
    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "drift": {"level": "monitor", "max_psi": 0.15, "drifted_features": ["x"]},
    })
    assert sig.metadata["recommendation"] == "retrain"
    assert "baseline stale" in sig.reasoning


def test_drift_score_shortcut(model_paths, isolate_metadata):
    """Raw drift_score resolves to a tier via the default thresholds."""
    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )
    sig = KnowledgeAdaptionAgent().run(
        {"regime": "trending_bull", "drift_score": 0.30},
    )
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["drift_level"] == "retrain"


def test_drift_from_feature_frame_end_to_end(
    tmp_path, monkeypatch, model_paths, isolate_metadata,
):
    """Stored fingerprint + live shifted frame → agent sees retrain drift."""
    _isolate_stamp_db(monkeypatch, tmp_path)

    baseline, regime = model_paths
    _write_pickle(baseline, {"model": object()}, age_seconds=60)
    _write_pickle(
        regime, {"models": {"trending_bull": 1, "trending_bear": 1,
                             "mean_reverting": 1, "high_vol": 1}}, age_seconds=60,
    )

    # Seed a training fingerprint directly.
    import time as _time

    import numpy as _np
    import pandas as _pd

    from data.db import get_connection

    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO model_feature_stats "
            "(model_name, trained_at, feature_name, mean, std, "
            " q10, q50, q90, n_samples) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("lgbm_alpha", _time.time(), "ret_5d",
             0.0, 1.0, -1.28, 0.0, 1.28, 1000),
        )
    conn.close()

    # Live frame with a clear +2σ shift → expect retrain.
    rng = _np.random.default_rng(777)
    live_frame = _pd.DataFrame({"ret_5d": rng.standard_normal(2000) + 2.0})

    sig = KnowledgeAdaptionAgent().run({
        "regime": "trending_bull",
        "feature_frame": live_frame,
    })
    assert sig.metadata["recommendation"] == "retrain"
    assert sig.metadata["drift_level"] == "retrain"
