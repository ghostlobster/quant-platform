"""
bus/event_bus.py — Redis Streams (with in-memory fallback).

Activation is opt-in: ``EVENT_BUS_ENABLED=1`` flips publishers from no-op
into a real ``redis.Redis().xadd()`` call. When the env var is unset,
``publish()`` is silent so existing crons / scheduler ticks pay zero
cost.

Two backends share the same surface:

* :class:`RedisEventBus` — production. Uses ``redis-py`` Streams
  (``XADD`` for publish, ``XRANGE`` / ``XREAD`` for replay).
* :class:`InMemoryEventBus` — fallback when the ``redis`` package is
  missing or ``EVENT_BUS_ENABLED!=1``. Backed by a per-stream list so
  unit tests can exercise the publish/replay surface without spinning
  up Redis.

Both implement the same protocol:

    bus.publish(event: Event, stream: str | None = None) -> str
        Returns a monotonic message-id (``"<timestamp>-<seq>"`` form).

    bus.replay(stream: str, since: str = "0", limit: int = 1000)
        Yields ``(message_id, Event)`` tuples in publish order.

    bus.subscribe(stream: str, group: str, consumer: str, block_ms: int)
        Generator over ``(message_id, Event)`` for blocking consumption
        (Redis only — the in-memory fallback raises NotImplementedError).

Public factory: :func:`get_event_bus()`. Threadsafe singleton; honours
``EVENT_BUS_ENABLED`` and ``REDIS_URL`` env vars.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from typing import Iterator

import structlog

from bus.events import Event, EventType, Stream  # noqa: F401 — re-export

logger = structlog.get_logger(__name__)

try:
    import redis as _redis
except ImportError:
    _redis = None


def _is_enabled() -> bool:
    raw = os.environ.get("EVENT_BUS_ENABLED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


# ── Redis backend ───────────────────────────────────────────────────────────

class RedisEventBus:
    """Redis Streams implementation."""

    def __init__(
        self,
        url: str | None = None,
        *,
        max_len: int | None = None,
    ) -> None:
        if _redis is None:
            raise ImportError(
                "redis package is required for RedisEventBus. "
                "Install it with: pip install redis",
            )
        self._url = url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self._max_len = max_len if max_len is not None else int(
            os.environ.get("EVENT_BUS_MAX_LEN", "100000"),
        )
        self._client = _redis.from_url(self._url, decode_responses=True)

    def publish(self, event: Event, stream: str | None = None) -> str:
        target = stream or Stream.for_event(event.event_type)
        fields = {
            "event_type": event.event_type,
            "ts": event.ts,
            "payload": json.dumps(event.payload),
        }
        kwargs: dict = {}
        if self._max_len:
            kwargs["maxlen"] = self._max_len
            kwargs["approximate"] = True
        return self._client.xadd(target, fields, **kwargs)

    def replay(
        self,
        stream: str,
        since: str = "0",
        limit: int = 1000,
    ) -> Iterator[tuple[str, Event]]:
        rows = self._client.xrange(stream, min=since, count=limit)
        for msg_id, fields in rows:
            yield msg_id, _decode_event(fields)

    def subscribe(
        self,
        stream: str,
        group: str,
        consumer: str,
        block_ms: int = 5000,
    ) -> Iterator[tuple[str, Event]]:
        # Best-effort group create — ignore BUSYGROUP if it already exists.
        try:
            self._client.xgroup_create(stream, group, id="0", mkstream=True)
        except Exception as exc:  # pragma: no cover — group already exists
            if "BUSYGROUP" not in str(exc):
                raise
        while True:
            response = self._client.xreadgroup(
                group, consumer, {stream: ">"}, count=10, block=block_ms,
            )
            if not response:
                continue
            for _stream, messages in response:
                for msg_id, fields in messages:
                    yield msg_id, _decode_event(fields)


# ── In-memory backend ───────────────────────────────────────────────────────

class InMemoryEventBus:
    """List-backed fallback. Not thread-safe across processes — single-process
    only. Useful for unit tests and offline dev runs."""

    def __init__(self) -> None:
        self._streams: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        self._seq = 0
        self._lock = threading.Lock()

    def _next_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"{int(time.time() * 1000)}-{self._seq}"

    def publish(self, event: Event, stream: str | None = None) -> str:
        target = stream or Stream.for_event(event.event_type)
        fields = {
            "event_type": event.event_type,
            "ts": event.ts,
            "payload": json.dumps(event.payload),
        }
        msg_id = self._next_id()
        with self._lock:
            self._streams[target].append((msg_id, fields))
        return msg_id

    def replay(
        self,
        stream: str,
        since: str = "0",
        limit: int = 1000,
    ) -> Iterator[tuple[str, Event]]:
        with self._lock:
            rows = list(self._streams.get(stream, []))
        out: list[tuple[str, Event]] = []
        for msg_id, fields in rows:
            if since != "0" and msg_id < since:
                continue
            out.append((msg_id, _decode_event(fields)))
            if len(out) >= limit:
                break
        yield from out

    def subscribe(self, *args, **kwargs):
        raise NotImplementedError(
            "InMemoryEventBus does not support blocking subscribe; "
            "use a real Redis instance via EVENT_BUS_ENABLED=1.",
        )


def _decode_event(fields: dict) -> Event:
    payload_raw = fields.get("payload", "{}")
    try:
        payload = json.loads(payload_raw)
    except (TypeError, json.JSONDecodeError):
        payload = {"raw": payload_raw}
    return Event(
        event_type=fields.get("event_type", ""),
        payload=payload,
        ts=fields.get("ts", ""),
    )


# ── Singleton factory ───────────────────────────────────────────────────────

_bus_lock = threading.Lock()
_bus_instance: RedisEventBus | InMemoryEventBus | None = None


def get_event_bus() -> RedisEventBus | InMemoryEventBus:
    """Return the configured singleton bus.

    Picks Redis when ``EVENT_BUS_ENABLED=1`` and the ``redis`` package
    is importable; otherwise falls back to the in-memory backend so
    callers never crash on a missing dependency.
    """
    global _bus_instance
    if _bus_instance is not None:
        return _bus_instance
    with _bus_lock:
        if _bus_instance is not None:
            return _bus_instance
        if _is_enabled() and _redis is not None:
            try:
                _bus_instance = RedisEventBus()
                logger.info("event_bus: Redis backend active")
            except Exception as exc:
                logger.warning(
                    "event_bus: Redis init failed, falling back to memory",
                    error=str(exc),
                )
                _bus_instance = InMemoryEventBus()
        else:
            _bus_instance = InMemoryEventBus()
    return _bus_instance


def reset_event_bus() -> None:
    """Reset the singleton — intended for tests."""
    global _bus_instance
    with _bus_lock:
        _bus_instance = None


# ── Publish helper (best-effort, never raises) ──────────────────────────────

def publish(event_type: str, payload: dict, stream: str | None = None) -> str | None:
    """Convenience wrapper used by cron / scheduler / agent code.

    Failures are swallowed with a structured-log warning — the bus must
    never bring down the calling cron when Redis is unreachable.
    """
    try:
        evt = Event(event_type=event_type, payload=payload)
        return get_event_bus().publish(evt, stream=stream)
    except Exception as exc:
        logger.warning("event_bus: publish failed",
                       event_type=event_type, error=str(exc))
        return None
