"""
tests/test_walk_forward_executor.py — executor selection in walk_forward_parallel.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.walk_forward import _resolve_executor

# ── _resolve_executor — argument vs env var precedence ──────────────────────

def test_resolve_explicit_argument_wins(monkeypatch):
    monkeypatch.setenv("WF_EXECUTOR", "ray")
    assert _resolve_executor("serial") == "serial"


def test_resolve_env_var_used_when_no_argument(monkeypatch):
    monkeypatch.setenv("WF_EXECUTOR", "mp")
    assert _resolve_executor(None) == "mp"


def test_resolve_default_auto_picks_ray_when_importable(monkeypatch):
    monkeypatch.delenv("WF_EXECUTOR", raising=False)
    monkeypatch.delenv("RAY_ENABLED", raising=False)

    # Pretend ray is importable.
    import importlib

    real_import = importlib.import_module

    def _fake_import(name, *args, **kwargs):
        if name == "ray":
            return type("ray_stub", (), {})()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    # Module doesn't actually use importlib — it uses ``import ray``.
    # Provide it via sys.modules for the duration of the test.
    monkeypatch.setitem(sys.modules, "ray", type("_M", (), {})())
    assert _resolve_executor(None) == "ray"


def test_resolve_default_auto_falls_back_to_mp_without_ray(monkeypatch):
    monkeypatch.delenv("WF_EXECUTOR", raising=False)
    monkeypatch.delenv("RAY_ENABLED", raising=False)
    monkeypatch.setitem(sys.modules, "ray", None)
    # The bare ``import ray`` resolves to None which raises ImportError on
    # attribute access; force a clean ImportError by removing the entry.
    sys.modules.pop("ray", None)

    # Patch builtins.__import__ to raise on "ray" so the auto branch falls
    # back to multiprocessing.
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "ray":
            raise ImportError("test-fake")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert _resolve_executor(None) == "mp"


def test_resolve_legacy_ray_enabled_flag(monkeypatch):
    """Operators on the old RAY_ENABLED=1 flag still get Ray."""
    monkeypatch.delenv("WF_EXECUTOR", raising=False)
    monkeypatch.setenv("RAY_ENABLED", "1")
    assert _resolve_executor(None) == "ray"


def test_resolve_explicit_serial(monkeypatch):
    monkeypatch.setenv("RAY_ENABLED", "1")
    assert _resolve_executor("serial") == "serial"


def test_resolve_rejects_unknown_executor():
    with pytest.raises(ValueError, match="executor"):
        _resolve_executor("dask")


# ── walk_forward_parallel serial path round-trip ─────────────────────────────

def _toy_df(n=300):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = 100 + rng.normal(0, 1, n).cumsum()
    highs = closes + 0.5
    lows  = closes - 0.5
    opens = closes - 0.1
    vols  = rng.integers(1_000, 10_000, n)
    # Backtester engine expects capitalised OHLCV columns.
    return pd.DataFrame(
        {
            "Open": opens, "High": highs, "Low": lows, "Close": closes,
            "Volume": vols,
        },
        index=idx,
    )


def test_serial_executor_returns_results(monkeypatch):
    """The serial path runs every segment in-process — useful for debugging."""
    from backtester.walk_forward import walk_forward_parallel

    df = _toy_df(300)
    monkeypatch.delenv("WF_EXECUTOR", raising=False)
    monkeypatch.delenv("RAY_ENABLED", raising=False)
    wf = walk_forward_parallel(df, executor="serial", ticker="TST")
    assert wf.windows  # at least one segment ran
    assert wf.consistency_score >= 0


def test_parallel_executor_invalid_choice():
    from backtester.walk_forward import walk_forward_parallel

    df = _toy_df(300)
    with pytest.raises(ValueError, match="executor"):
        walk_forward_parallel(df, executor="dask")
