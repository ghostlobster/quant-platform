"""Tests for providers/model_registry.py and mock adapter (Issue #29)."""
from __future__ import annotations

import pytest


def test_mock_adapter_log_returns_run_id():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    run_id = reg.log_model(
        run_name="rl_sizer",
        model_path="/models/rl_sizer.zip",
        metrics={"sharpe": 1.2, "max_drawdown": 0.08},
    )
    assert run_id.startswith("mock-run-")


def test_mock_adapter_list_models_empty():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    assert reg.list_models() == []


def test_mock_adapter_list_models_after_log():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    reg.log_model("regime_clf", "/models/regime.pkl", {"accuracy": 0.75})
    reg.log_model("rl_sizer", "/models/rl.zip", {"sharpe": 1.5})

    models = reg.list_models()
    assert len(models) == 2
    names = {m["run_name"] for m in models}
    assert "regime_clf" in names
    assert "rl_sizer" in names


def test_mock_adapter_promote_and_load():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    run_id = reg.log_model("rl_sizer", "/models/rl.zip", {"sharpe": 1.3})
    reg.promote("rl_sizer", run_id, "Production")

    path = reg.load_model("rl_sizer", stage="Production")
    assert path == "/models/rl.zip"


def test_mock_adapter_load_not_found_raises():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    with pytest.raises(FileNotFoundError):
        reg.load_model("nonexistent_model", stage="Production")


def test_mock_adapter_promote_unknown_run_id_raises():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    with pytest.raises(KeyError):
        reg.promote("rl_sizer", "nonexistent-run-id", "Production")


def test_mock_adapter_metrics_stored():
    from adapters.model_registry.mock_adapter import MockModelRegistryAdapter

    reg = MockModelRegistryAdapter()
    metrics = {"sharpe": 1.8, "win_rate": 0.62}
    reg.log_model("my_model", "/path/to/model", metrics, tags={"env": "prod"})

    models = reg.list_models()
    assert len(models) == 1
    assert models[0]["metrics"] == metrics
    assert models[0]["tags"] == {"env": "prod"}


def test_get_model_registry_mock_provider():
    from providers.model_registry import get_model_registry

    reg = get_model_registry(provider="mock")
    # Should be usable without errors
    run_id = reg.log_model("test", "/tmp/model.pkl", {"loss": 0.01})
    assert run_id is not None


def test_get_model_registry_unknown_raises():
    from providers.model_registry import get_model_registry

    with pytest.raises(ValueError, match="Unknown model registry provider"):
        get_model_registry(provider="unknown_registry")
