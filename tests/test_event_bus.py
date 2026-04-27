"""
tests/test_event_bus.py — bus.event_bus + bus.events round-trip.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bus import event_bus
from bus.event_bus import (
    InMemoryEventBus,
    Stream,
    get_event_bus,
    publish,
    reset_event_bus,
)
from bus.events import Event, EventType


@pytest.fixture(autouse=True)
def _reset_bus():
    reset_event_bus()
    yield
    reset_event_bus()


# ── Event dataclass ──────────────────────────────────────────────────────────

def test_event_validates_event_type():
    with pytest.raises(ValueError, match="event_type"):
        Event(event_type="", payload={})


def test_event_validates_payload_type():
    with pytest.raises(ValueError, match="payload"):
        Event(event_type=EventType.SIGNAL_GENERATED, payload="not-a-dict")


def test_event_default_ts_is_set():
    e = Event(EventType.SIGNAL_GENERATED, {"ticker": "AAPL"})
    assert e.ts  # ISO 8601 string


def test_event_type_constants():
    expected = {
        "signal.generated", "order.placed", "order.rejected",
        "order.filled", "risk.breach", "kill.switch",
    }
    assert set(EventType.all()) == expected


def test_stream_routing_by_event_type():
    assert Stream.for_event(EventType.SIGNAL_GENERATED) == Stream.SIGNALS
    assert Stream.for_event(EventType.ORDER_PLACED)     == Stream.ORDERS
    assert Stream.for_event(EventType.ORDER_REJECTED)   == Stream.ORDERS
    assert Stream.for_event(EventType.RISK_BREACH)      == Stream.RISK
    assert Stream.for_event(EventType.KILLSWITCH)       == Stream.ORDERS
    assert Stream.for_event("unknown.event")            == "events"


# ── In-memory backend ───────────────────────────────────────────────────────

def test_inmemory_publish_and_replay_round_trip():
    bus = InMemoryEventBus()
    bus.publish(Event(EventType.SIGNAL_GENERATED, {"ticker": "AAPL", "score": 0.4}))
    bus.publish(Event(EventType.ORDER_PLACED,    {"ticker": "AAPL", "qty": 10}))

    sigs = list(bus.replay(Stream.SIGNALS))
    orders = list(bus.replay(Stream.ORDERS))

    assert len(sigs)   == 1
    assert len(orders) == 1
    assert sigs[0][1].event_type   == EventType.SIGNAL_GENERATED
    assert sigs[0][1].payload      == {"ticker": "AAPL", "score": 0.4}
    assert orders[0][1].event_type == EventType.ORDER_PLACED


def test_inmemory_replay_with_limit():
    bus = InMemoryEventBus()
    for i in range(5):
        bus.publish(Event(EventType.SIGNAL_GENERATED, {"i": i}))
    rows = list(bus.replay(Stream.SIGNALS, limit=3))
    assert len(rows) == 3


def test_inmemory_replay_since_filter():
    bus = InMemoryEventBus()
    ids: list[str] = []
    for i in range(3):
        ids.append(bus.publish(Event(EventType.SIGNAL_GENERATED, {"i": i})))
    rows = list(bus.replay(Stream.SIGNALS, since=ids[1]))
    assert len(rows) == 2  # rows >= ids[1]


def test_inmemory_subscribe_raises():
    bus = InMemoryEventBus()
    with pytest.raises(NotImplementedError):
        next(bus.subscribe("signals", "g", "c"))


# ── publish() helper + factory selection ────────────────────────────────────

def test_publish_helper_uses_inmemory_when_disabled(monkeypatch):
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    bus = get_event_bus()
    assert isinstance(bus, InMemoryEventBus)

    msg_id = publish(EventType.RISK_BREACH, {"drawdown": -0.15})
    assert msg_id is not None
    rows = list(bus.replay(Stream.RISK))
    assert len(rows) == 1
    assert rows[0][1].payload == {"drawdown": -0.15}


def test_factory_falls_back_to_memory_when_redis_init_fails(monkeypatch):
    monkeypatch.setenv("EVENT_BUS_ENABLED", "1")

    def _broken_init(*args, **kwargs):
        raise RuntimeError("redis unreachable")

    monkeypatch.setattr(event_bus, "RedisEventBus", _broken_init)

    bus = get_event_bus()
    assert isinstance(bus, InMemoryEventBus)


def test_publish_helper_swallows_event_validation_errors(monkeypatch):
    """An invalid event payload shape must not crash the cron — the
    helper logs and returns None."""
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    msg_id = publish("", {"bad": "type"})
    assert msg_id is None


def test_factory_singleton_returns_same_instance(monkeypatch):
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2


def test_reset_event_bus_clears_singleton(monkeypatch):
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    bus1 = get_event_bus()
    reset_event_bus()
    bus2 = get_event_bus()
    assert bus1 is not bus2


# ── Decode round-trip ───────────────────────────────────────────────────────

def test_decode_event_handles_dict_payload():
    bus = InMemoryEventBus()
    bus.publish(Event(EventType.ORDER_FILLED, {"id": "abc", "qty": 5}))
    rows = list(bus.replay(Stream.ORDERS))
    assert rows[0][1].payload == {"id": "abc", "qty": 5}


def test_decode_event_falls_back_when_payload_unparseable():
    """The decoder defends against non-JSON strings landing on a stream."""
    bus = InMemoryEventBus()
    bus._streams[Stream.SIGNALS].append(
        ("123-1", {"event_type": "signal.generated", "ts": "now", "payload": "not-json"}),
    )
    rows = list(bus.replay(Stream.SIGNALS))
    assert rows[0][1].payload == {"raw": "not-json"}


# ── Stream override ─────────────────────────────────────────────────────────

def test_publish_respects_explicit_stream(monkeypatch):
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    publish(EventType.SIGNAL_GENERATED, {"ticker": "AAPL"}, stream="custom")
    bus = get_event_bus()
    rows = list(bus.replay("custom"))
    assert len(rows) == 1


# ── JSON serialisation of the bus message ────────────────────────────────────

def test_publish_serialises_payload_as_json():
    bus = InMemoryEventBus()
    bus.publish(Event(EventType.ORDER_FILLED, {"id": 1, "filled": True}))
    fields = bus._streams[Stream.ORDERS][0][1]
    # Stored as JSON string under the "payload" field.
    assert isinstance(fields["payload"], str)
    assert json.loads(fields["payload"]) == {"id": 1, "filled": True}
