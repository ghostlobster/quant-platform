"""
adapters/feature_store/memory_adapter.py — In-memory feature store (default).

No external dependency.  Data is lost on process restart.
Thread-safe via a module-level lock and shared store.
"""
from __future__ import annotations

import threading

_store: dict[str, dict] = {}
_lock = threading.Lock()


class InMemoryFeatureStoreAdapter:
    """FeatureStoreProvider backed by a process-lifetime dict."""

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict:
        with _lock:
            entity = _store.get(entity_id, {})
        return {k: entity[k] for k in feature_names if k in entity}

    def set_features(self, entity_id: str, features: dict) -> None:
        with _lock:
            if entity_id not in _store:
                _store[entity_id] = {}
            _store[entity_id].update(features)

    def list_features(self) -> list[str]:
        with _lock:
            names: set[str] = set()
            for entity in _store.values():
                names.update(entity.keys())
        return sorted(names)
