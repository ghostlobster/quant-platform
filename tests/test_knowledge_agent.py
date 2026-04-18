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
    monkeypatch.setattr("agents.knowledge_agent._resolve_path", boom)
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
