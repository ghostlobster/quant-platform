"""
adapters/model_registry/mlflow_adapter.py — ModelRegistryProvider backed by MLflow.

Uses local file store by default; switch to S3 or remote tracking server via
MLFLOW_TRACKING_URI environment variable.

Requires (optional): pip install mlflow>=2.0.0

ENV vars
--------
    MLFLOW_TRACKING_URI   local path or remote URI (default: mlruns)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


class MLflowAdapter:  # pragma: no cover
    """ModelRegistryProvider wrapping the MLflow tracking and registry APIs."""

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._uri = tracking_uri or _TRACKING_URI
        try:
            import mlflow  # type: ignore[import]
            mlflow.set_tracking_uri(self._uri)
            self._mlflow = mlflow
            logger.info("MLflowAdapter: tracking URI = %s", self._uri)
        except ImportError:
            self._mlflow = None
            logger.warning(
                "mlflow not installed; MLflowAdapter will no-op. "
                "Install with: pip install mlflow"
            )

    def log_model(
        self,
        run_name: str,
        model_path: str,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> str:
        if self._mlflow is None:
            logger.warning("MLflowAdapter.log_model: mlflow not available")
            return ""
        mlflow = self._mlflow
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_metrics(metrics)
            if tags:
                mlflow.set_tags(tags)
            # Log the model file as a generic artifact
            if Path(model_path).exists():
                mlflow.log_artifact(model_path, artifact_path="model")
            run_id = run.info.run_id
        logger.info("MLflowAdapter: logged run %s (model=%s)", run_id, model_path)
        return run_id

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        if self._mlflow is None:
            raise RuntimeError("mlflow not installed")
        mlflow = self._mlflow
        try:
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise FileNotFoundError(
                    f"No model '{model_name}' at stage '{stage}'"
                )
            version = versions[0]
            artifact_uri = client.get_model_version_download_uri(
                model_name, version.version
            )
            logger.info(
                "MLflowAdapter: loading %s v%s from %s",
                model_name, version.version, artifact_uri,
            )
            return artifact_uri  # caller reconstructs model from path
        except self._mlflow.exceptions.MlflowException as exc:
            raise FileNotFoundError(str(exc)) from exc

    def list_models(self) -> list[dict]:
        if self._mlflow is None:
            return []
        try:
            client = self._mlflow.tracking.MlflowClient()
            registered = client.search_registered_models()
            return [
                {
                    "name": m.name,
                    "latest_versions": [
                        {"version": v.version, "stage": v.current_stage}
                        for v in m.latest_versions
                    ],
                }
                for m in registered
            ]
        except Exception as exc:
            logger.warning("MLflowAdapter.list_models failed: %s", exc)
            return []

    def promote(self, model_name: str, run_id: str, stage: str) -> None:
        if self._mlflow is None:
            logger.warning("MLflowAdapter.promote: mlflow not available")
            return
        mlflow = self._mlflow
        client = mlflow.tracking.MlflowClient()
        # Register the model if not already registered
        try:
            result = mlflow.register_model(
                f"runs:/{run_id}/model", model_name
            )
            version = result.version
        except Exception as exc:
            logger.warning("MLflowAdapter: register_model failed: %s", exc)
            # Fallback: get existing latest version for this run
            versions = client.search_model_versions(f"run_id='{run_id}'")
            if not versions:
                raise
            version = versions[0].version

        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )
        logger.info(
            "MLflowAdapter: promoted %s v%s → %s", model_name, version, stage
        )
