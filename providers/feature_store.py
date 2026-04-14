"""
providers/feature_store.py — FeatureStoreProvider protocol and factory.

ENV vars
--------
    FEATURE_STORE_PROVIDER   memory | redis  (default: memory)
    REDIS_URL                (required for redis adapter)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class FeatureStoreProvider(Protocol):
    """Duck-typed interface for feature retrieval and storage."""

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict:
        """
        Retrieve named features for *entity_id*.

        Returns
        -------
        dict mapping feature name → value.  Missing features are omitted.
        """
        ...

    def set_features(self, entity_id: str, features: dict) -> None:
        """Store *features* for *entity_id*."""
        ...

    def list_features(self) -> list[str]:
        """Return all known feature names across all entities."""
        ...


def get_feature_store(provider: Optional[str] = None) -> FeatureStoreProvider:
    """
    Return a configured FeatureStoreProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the FEATURE_STORE_PROVIDER env var.  One of:
        ``memory``, ``redis``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (
        provider or os.environ.get("FEATURE_STORE_PROVIDER", "memory")
    ).lower().strip()
    if name == "memory":
        from adapters.feature_store.memory_adapter import InMemoryFeatureStoreAdapter
        return InMemoryFeatureStoreAdapter()
    if name == "redis":
        from adapters.feature_store.redis_adapter import RedisFeatureStoreAdapter
        return RedisFeatureStoreAdapter()
    raise ValueError(
        f"Unknown feature store provider: {name!r}. "
        "Valid options: memory, redis"
    )
