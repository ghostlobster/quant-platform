"""Tests for agents/knowledge_registry.py — ModelEntry zoo registry."""
from __future__ import annotations

import pytest

from agents.knowledge_registry import (
    ModelEntry,
    build_default_registry,
    worst_recommendation,
)

# ── ModelEntry dataclass ─────────────────────────────────────────────────────

def test_model_entry_fields():
    entry = ModelEntry(
        name="x",
        artefact_env="X_MODEL_PATH",
        artefact_default="models/x.pkl",
        metadata_name="x",
    )
    assert entry.name == "x"
    assert entry.artefact_env == "X_MODEL_PATH"
    assert entry.artefact_default == "models/x.pkl"
    assert entry.metadata_name == "x"
    assert entry.max_age_days == 45
    assert entry.is_baseline is False


def test_model_entry_is_frozen():
    entry = ModelEntry(
        name="x", artefact_env="X", artefact_default="y", metadata_name="x",
    )
    with pytest.raises((AttributeError, Exception)):
        entry.name = "mutated"  # type: ignore[misc]


def test_model_entry_custom_max_age():
    entry = ModelEntry(
        name="quarterly",
        artefact_env="Q", artefact_default="m/q.pkl", metadata_name="q",
        max_age_days=120,
    )
    assert entry.max_age_days == 120


# ── build_default_registry ───────────────────────────────────────────────────

def test_build_default_registry_includes_lgbm_alpha():
    registry = build_default_registry()
    names = [e.name for e in registry]
    assert "lgbm_alpha" in names
    baseline = next(e for e in registry if e.name == "lgbm_alpha")
    assert baseline.is_baseline is True


def test_build_default_registry_covers_zoo():
    registry = build_default_registry()
    names = {e.name for e in registry}
    # Every zoo member that currently defines MODEL_ENTRY should show up.
    # We assert on the ones that don't require torch (which may be absent
    # in CI). The torch-dependent entries are additive — if present they
    # don't break this test.
    expected = {"lgbm_alpha", "ridge_alpha", "bayesian_alpha",
                "mlp_alpha", "rf_long_short"}
    assert expected <= names


def test_build_default_registry_survives_missing_module(monkeypatch):
    """A strategy import that raises should drop that one entry, not
    the whole registry."""
    import importlib

    import agents.knowledge_registry as kr_mod

    real_import_module = importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "strategies.bayesian_signal":
            raise RuntimeError("missing optional dep")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)
    registry = kr_mod.build_default_registry()
    names = {e.name for e in registry}
    assert "bayesian_alpha" not in names
    assert "lgbm_alpha" in names  # other modules still register


def test_build_default_registry_env_override(monkeypatch):
    """Env-var override does not change the ModelEntry — path resolution
    happens later via ``_safe_pickle_path``. Confirm the registry still
    points callers at the overriding env var (the entry's contract)."""
    monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", "/tmp/x.pkl")
    registry = build_default_registry()
    baseline = next(e for e in registry if e.name == "lgbm_alpha")
    assert baseline.artefact_env == "LGBM_ALPHA_MODEL_PATH"


# ── worst_recommendation ─────────────────────────────────────────────────────

def test_worst_recommendation_retrain_wins():
    assert worst_recommendation(["fresh", "monitor", "retrain"]) == "retrain"
    assert worst_recommendation(["retrain", "fresh"]) == "retrain"


def test_worst_recommendation_monitor_over_fresh():
    assert worst_recommendation(["fresh", "monitor"]) == "monitor"
    assert worst_recommendation(["monitor"]) == "monitor"


def test_worst_recommendation_fresh_only():
    assert worst_recommendation(["fresh", "fresh"]) == "fresh"


def test_worst_recommendation_empty():
    assert worst_recommendation([]) == "fresh"


def test_worst_recommendation_ignores_unknown():
    # Unknown tags map to the fresh priority (0); retrain still wins.
    assert worst_recommendation(["retrain", "mystery"]) == "retrain"
    # All unknown → fresh (the default priority).
    assert worst_recommendation(["mystery"]) == "mystery"  # max() returns it
    # …but real call sites never produce unknown tags; this documents the
    # failure-open behaviour.
