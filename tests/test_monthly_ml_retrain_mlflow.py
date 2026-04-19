"""
tests/test_monthly_ml_retrain_mlflow.py — MLflow wiring in monthly_ml_retrain.

Mocks :func:`providers.model_registry.get_model_registry` to verify the
cron logs IC metrics and tags at the documented shape without requiring
mlflow to be installed.
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
        return "run-42"

    def promote(self, model_name, run_id, stage):
        self.promoted.append((model_name, run_id, stage))


@pytest.fixture
def fake_result() -> dict:
    return {
        "train_ic":   0.051,
        "test_ic":    0.043,
        "train_icir": 0.9,
        "test_icir":  0.72,
        "n_train_samples": 1234,
        "n_test_samples":   567,
    }


def _fake_registry_factory(fake: _FakeRegistry):
    import providers.model_registry as pmr

    return pmr, fake


# ── Flag resolution ─────────────────────────────────────────────────────────

def test_resolve_flag_cli_wins_over_env(monkeypatch):
    monkeypatch.setenv("MODEL_REGISTRY_ENABLED", "0")
    assert retrain._resolve_registry_flag(True) is True


def test_resolve_flag_env_default_on(monkeypatch):
    monkeypatch.delenv("MODEL_REGISTRY_ENABLED", raising=False)
    assert retrain._resolve_registry_flag(None) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off"])
def test_resolve_flag_env_disables(monkeypatch, raw):
    monkeypatch.setenv("MODEL_REGISTRY_ENABLED", raw)
    assert retrain._resolve_registry_flag(None) is False


# ── _log_to_registry happy path ─────────────────────────────────────────────

def test_log_to_registry_happy_path(monkeypatch, fake_result):
    fake = _FakeRegistry()
    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: fake,
    )
    monkeypatch.setenv("MLFLOW_REGISTERED_MODEL", "lgbm_alpha_test")

    run_id = retrain._log_to_registry(
        model_path="/tmp/model.pkl",
        train_result=fake_result,
        tickers=["SPY", "QQQ"],
        period="2y",
    )
    assert run_id == "run-42"
    assert len(fake.logged) == 1
    logged = fake.logged[0]
    assert logged["run_name"] == "lgbm_alpha_test_retrain"
    assert logged["model_path"] == "/tmp/model.pkl"
    assert logged["metrics"]["train_ic"] == pytest.approx(0.051)
    assert logged["metrics"]["test_ic"] == pytest.approx(0.043)
    assert logged["metrics"]["n_train"] == 1234
    assert logged["tags"]["model_name"] == "lgbm_alpha_test"
    assert logged["tags"]["period"] == "2y"
    assert logged["tags"]["n_tickers"] == "2"
    assert logged["tags"]["tickers"] == "SPY,QQQ"
    assert logged["tags"]["source"] == "cron.monthly_ml_retrain"
    assert fake.promoted == [("lgbm_alpha_test", "run-42", "Staging")]


def test_log_to_registry_empty_run_id_is_noop(monkeypatch, fake_result):
    class _Silent(_FakeRegistry):
        def log_model(self, *args, **kwargs):
            super().log_model(*args, **kwargs)
            return ""  # MLflow adapter returns empty when mlflow missing

    fake = _Silent()
    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: fake,
    )
    assert (
        retrain._log_to_registry("/tmp/m.pkl", fake_result, ["SPY"], "1y") is None
    )
    # log_model was still called; promote was not.
    assert len(fake.logged) == 1
    assert fake.promoted == []


def test_log_to_registry_swallows_log_model_exception(monkeypatch, fake_result):
    class _Broken(_FakeRegistry):
        def log_model(self, *args, **kwargs):
            raise RuntimeError("disk full")

    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: _Broken(),
    )
    # Must return None and not raise — a broken registry never halts the cron.
    assert (
        retrain._log_to_registry("/tmp/m.pkl", fake_result, ["SPY"], "1y") is None
    )


def test_log_to_registry_swallows_registry_unavailable(monkeypatch, fake_result):
    def _boom():
        raise ImportError("mlflow missing")

    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", _boom,
    )
    assert (
        retrain._log_to_registry("/tmp/m.pkl", fake_result, ["SPY"], "1y") is None
    )


def test_log_to_registry_swallows_promote_exception(monkeypatch, fake_result):
    class _PartBroken(_FakeRegistry):
        def promote(self, *args, **kwargs):
            raise RuntimeError("permission denied")

    fake = _PartBroken()
    monkeypatch.setattr(
        "providers.model_registry.get_model_registry", lambda: fake,
    )
    run_id = retrain._log_to_registry(
        "/tmp/m.pkl", fake_result, ["SPY"], "1y",
    )
    # Log still succeeded even though promotion raised.
    assert run_id == "run-42"
    assert len(fake.logged) == 1
