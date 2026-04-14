"""
providers/model_registry.py — ModelRegistryProvider protocol and factory.

Stores versioned model checkpoints with metrics and lifecycle stages.
Default: local MLflow file store (./mlruns).

ENV vars
--------
    MODEL_REGISTRY_PROVIDER   mlflow | mock  (default: mlflow)
    MLFLOW_TRACKING_URI       local path or s3:// URI (default: mlruns)
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable

__all__ = ["ModelRegistryProvider", "get_model_registry"]


@runtime_checkable
class ModelRegistryProvider(Protocol):
    """Duck-typed interface for model versioning and lifecycle management."""

    def log_model(
        self,
        run_name: str,
        model_path: str,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str:
        """
        Log a model artifact and its metrics.

        Returns the run_id string that can be used for promotion.
        """
        ...

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """
        Return the local path to the model artifact for *model_name* at *stage*.

        Raises FileNotFoundError if no model at that stage.
        """
        ...

    def list_models(self) -> list[dict]:
        """Return a list of dicts describing registered model versions."""
        ...

    def promote(self, model_name: str, run_id: str, stage: str) -> None:
        """
        Promote a model version identified by *run_id* to *stage*.

        Common stages: 'Staging', 'Production', 'Archived'
        """
        ...


def get_model_registry(provider: Optional[str] = None) -> ModelRegistryProvider:
    """
    Return a configured ModelRegistryProvider.

    Parameters
    ----------
    provider : str, optional
        Override MODEL_REGISTRY_PROVIDER env var.  One of: 'mlflow', 'mock'.
    """
    name = (provider or os.environ.get("MODEL_REGISTRY_PROVIDER", "mlflow")).lower().strip()
    if name == "mlflow":
        from adapters.model_registry.mlflow_adapter import MLflowAdapter
        return MLflowAdapter()
    if name == "mock":
        from adapters.model_registry.mock_adapter import MockModelRegistryAdapter
        return MockModelRegistryAdapter()
    raise ValueError(
        f"Unknown model registry provider: {name!r}. Valid options: mlflow, mock"
    )
