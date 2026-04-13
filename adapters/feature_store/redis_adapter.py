"""
adapters/feature_store/redis_adapter.py — Redis-backed feature store.

Requires:  pip install redis
ENV vars:  REDIS_URL  (default: redis://localhost:6379/0)
           REDIS_FEATURE_TTL_SECONDS  (default: 86400 = 24 h; 0 = no expiry)
"""
from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

try:
    import redis as _redis
except ImportError:
    _redis = None  # type: ignore[assignment]

_lock = threading.Lock()
_client = None


def _get_client(url: str) -> object:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            if _redis is None:
                raise ImportError(
                    "redis package is required for RedisFeatureStoreAdapter. "
                    "Install it with: pip install redis"
                )
            _client = _redis.from_url(url, decode_responses=True)
    return _client


class RedisFeatureStoreAdapter:
    """FeatureStoreProvider backed by Redis hash keys."""

    _KEY_PREFIX = "fstore:"
    _INDEX_KEY  = "fstore:__feature_names__"

    def __init__(
        self,
        url: str | None = None,
        ttl: int | None = None,
    ) -> None:
        if _redis is None:
            raise ImportError(
                "redis package is required for RedisFeatureStoreAdapter. "
                "Install it with: pip install redis"
            )
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._ttl = ttl if ttl is not None else int(
            os.environ.get("REDIS_FEATURE_TTL_SECONDS", "86400")
        )
        self._r = _get_client(self._url)

    def _key(self, entity_id: str) -> str:
        return f"{self._KEY_PREFIX}{entity_id}"

    def get_features(self, entity_id: str, feature_names: list[str]) -> dict:
        raw = self._r.hmget(self._key(entity_id), feature_names)
        result = {}
        for name, val in zip(feature_names, raw):
            if val is not None:
                try:
                    result[name] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    result[name] = val
        return result

    def set_features(self, entity_id: str, features: dict) -> None:
        key = self._key(entity_id)
        mapping = {k: json.dumps(v) for k, v in features.items()}
        self._r.hset(key, mapping=mapping)
        if self._ttl > 0:
            self._r.expire(key, self._ttl)
        # Track feature names in a global set
        if features:
            self._r.sadd(self._INDEX_KEY, *features.keys())

    def list_features(self) -> list[str]:
        return sorted(self._r.smembers(self._INDEX_KEY))
