"""
adapters/model_registry/mock_adapter.py — In-memory ModelRegistryProvider for tests.
"""
from __future__ import annotations

from typing import Any


class MockModelRegistryAdapter:
    """In-memory model registry — no external dependencies, suitable for unit tests."""

    def __init__(self) -> None:
        self._runs: dict[str, dict] = {}       # run_id → {name, path, metrics, tags}
        self._stages: dict[str, str] = {}      # "model_name:version" → stage
        self._counter = 0

    def log_model(
        self,
        run_name: str,
        model_path: str,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str:
        self._counter += 1
        run_id = f"mock-run-{self._counter:04d}"
        self._runs[run_id] = {
            "run_name": run_name,
            "model_path": model_path,
            "metrics": metrics,
            "tags": tags or {},
            "version": str(self._counter),
        }
        return run_id

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        for run_id, info in self._runs.items():
            key = f"{model_name}:{info['version']}"
            if self._stages.get(key) == stage and info["run_name"] == model_name:
                return info["model_path"]
        raise FileNotFoundError(f"No model '{model_name}' at stage '{stage}'")

    def list_models(self) -> list[dict]:
        return list(self._runs.values())

    def promote(self, model_name: str, run_id: str, stage: str) -> None:
        if run_id not in self._runs:
            raise KeyError(f"Unknown run_id: {run_id!r}")
        info = self._runs[run_id]
        key = f"{model_name}:{info['version']}"
        self._stages[key] = stage
