"""
tests/test_e2e_monthly_retrain_to_mlflow.py — retrain → registry log/promote.

Walks the full ``cron.monthly_ml_retrain.main → MLSignal.train →
checkpoint pickle → ModelRegistryProvider.log_model → promote("Staging")``
chain. The vendor MLflow client is replaced with a captured-call
``FakeRegistry``; LightGBM is bypassed via a ``FakeMLSignal`` so the
test runs without GPU or training data.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cron.monthly_ml_retrain as retrain


class _FakeRegistry:
    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.promoted: list[tuple[str, str, str]] = []

    def log_model(self, run_name, model_path, metrics, tags=None):
        self.logged.append(
            {
                "run_name": run_name,
                "model_path": model_path,
                "metrics": dict(metrics),
                "tags": dict(tags or {}),
            }
        )
        return "run-e2e-1"

    def promote(self, model_name, run_id, stage):
        self.promoted.append((model_name, run_id, stage))


class _FakeMLSignal:
    """Stand-in for strategies.ml_signal.MLSignal that records train calls
    and writes a 1-byte pickle so the registry log_model has a path that
    exists."""

    last_instance: "_FakeMLSignal | None" = None

    def __init__(self) -> None:
        self._model_path = os.environ.get("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")
        os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
        with open(self._model_path, "wb") as f:
            f.write(b"\x00")  # tiny placeholder so log_model can attach an artifact
        self.train_calls: list[tuple] = []
        _FakeMLSignal.last_instance = self

    def train(self, tickers, period="2y", lgbm_params=None):
        self.train_calls.append((tuple(tickers), period, lgbm_params))
        return {
            "train_ic":   0.052,
            "test_ic":    0.041,
            "train_icir": 0.92,
            "test_icir":  0.74,
            "n_train_samples": 1234,
            "n_test_samples":   456,
        }


@pytest.fixture
def fake_signal_and_registry(tmp_path, monkeypatch):
    """Replace MLSignal + load_best_params + get_model_registry."""
    monkeypatch.setenv("LGBM_ALPHA_MODEL_PATH", str(tmp_path / "models/lgbm.pkl"))
    monkeypatch.setattr(retrain, "_LGBM_AVAILABLE", True)
    monkeypatch.setattr(retrain, "MLSignal", _FakeMLSignal)
    monkeypatch.setattr(
        "strategies.ml_tuning.load_best_params", lambda name: None,
    )
    fake = _FakeRegistry()
    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: fake,
    )
    return fake


def test_retrain_logs_to_registry_and_promotes_to_staging(
    fake_signal_and_registry, monkeypatch,
):
    monkeypatch.setenv("WF_TICKERS", "SPY,QQQ")
    monkeypatch.setenv("ML_TRAIN_PERIOD", "1y")
    monkeypatch.setenv("MODEL_REGISTRY_ENABLED", "1")

    retrain.main([])  # no CLI flag → fall back to env

    fake = fake_signal_and_registry
    assert len(fake.logged) == 1
    entry = fake.logged[0]
    assert entry["run_name"] == "lgbm_alpha_retrain"
    assert entry["metrics"]["train_ic"] == pytest.approx(0.052)
    assert entry["metrics"]["test_ic"] == pytest.approx(0.041)
    assert entry["metrics"]["n_train"] == 1234
    assert entry["tags"]["source"] == "cron.monthly_ml_retrain"
    assert entry["tags"]["period"] == "1y"
    assert entry["tags"]["tickers"] == "SPY,QQQ"
    assert entry["tags"]["n_tickers"] == "2"

    assert fake.promoted == [("lgbm_alpha", "run-e2e-1", "Staging")]

    # The fake MLSignal recorded the train invocation.
    sig = _FakeMLSignal.last_instance
    assert sig is not None
    assert sig.train_calls == [(("SPY", "QQQ"), "1y", None)]


def test_retrain_skips_registry_when_no_log_to_mlflow(
    fake_signal_and_registry, monkeypatch,
):
    monkeypatch.setenv("WF_TICKERS", "SPY")
    retrain.main(["--no-log-to-mlflow"])
    assert fake_signal_and_registry.logged == []
    assert fake_signal_and_registry.promoted == []


def test_retrain_swallows_registry_exception_after_local_save(
    fake_signal_and_registry, monkeypatch,
):
    """A broken registry must not propagate — the local checkpoint has
    already been written and the cron should exit successfully."""

    class _BrokenRegistry(_FakeRegistry):
        def log_model(self, *args, **kwargs):
            raise RuntimeError("tracking server down")

    broken = _BrokenRegistry()
    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: broken,
    )
    monkeypatch.setenv("WF_TICKERS", "SPY")

    # Must not raise — main() exits zero even when the registry breaks.
    retrain.main([])

    # The exception was caught — no successful log/promote recorded.
    assert broken.logged == []
    assert broken.promoted == []
