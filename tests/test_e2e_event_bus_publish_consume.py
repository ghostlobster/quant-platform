"""
tests/test_e2e_event_bus_publish_consume.py — bus.publish round-trip.

Closes part of #222.

Exercises the full event-bus chain end-to-end:

  publish(event_type, payload) → ``get_event_bus()`` → backend.publish
  → backend.replay → consumer iterates and asserts on the decoded
  Event.

Two backends are exercised back-to-back:

  * In-memory (``EVENT_BUS_ENABLED!=1``) — the default, also the
    fallback when the ``redis`` package is missing.
  * Redis-mocked (``EVENT_BUS_ENABLED=1`` + a ``MagicMock`` redis
    module via ``_fake_redis_module``) — the production path.

The cleanup-invariant fixture is opt-out for these tests: they
publish events that don't correspond to paper_trades, so the
"every fill has a journal row" assertion is not relevant.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from bus import event_bus
from bus.event_bus import (
    InMemoryEventBus,
    Stream,
    get_event_bus,
    publish,
    reset_event_bus,
)
from bus.events import Event, EventType

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skip_invariant]


def _fake_redis_module(client: MagicMock) -> MagicMock:
    """Stand-in for the ``redis`` package — same helper as #217 unit tests."""
    fake = MagicMock()
    fake.from_url.return_value = client
    return fake


@pytest.fixture(autouse=True)
def _reset_bus_singleton():
    reset_event_bus()
    yield
    reset_event_bus()


# ── Happy path: in-memory bus end-to-end ───────────────────────────────────


def test_publish_then_replay_yields_decoded_event(monkeypatch) -> None:
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    msg_id = publish(
        EventType.ORDER_FILLED, {"ticker": "AAPL", "qty": 10, "price": 150.0}
    )
    assert msg_id is not None  # publish returned an id

    bus = get_event_bus()
    assert isinstance(bus, InMemoryEventBus)

    rows = list(bus.replay(Stream.ORDERS))
    assert len(rows) == 1
    received_id, evt = rows[0]
    assert received_id == msg_id
    assert evt.event_type == EventType.ORDER_FILLED
    assert evt.payload == {"ticker": "AAPL", "qty": 10, "price": 150.0}


def test_replay_since_offset_filters_earlier_events(monkeypatch) -> None:
    """Consumer that resumes from a known offset must only see newer events."""
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    bus = get_event_bus()
    ids = []
    for i in range(3):
        ids.append(
            bus.publish(Event(EventType.SIGNAL_GENERATED, {"i": i}))
        )
    rows = list(bus.replay(Stream.SIGNALS, since=ids[1]))
    received = [evt.payload["i"] for _, evt in rows]
    assert received == [1, 2]


def test_multi_stream_isolation(monkeypatch) -> None:
    """Events on one stream don't bleed into another."""
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    publish(EventType.SIGNAL_GENERATED, {"ticker": "AAPL"})
    publish(EventType.RISK_BREACH, {"drawdown": -0.15})
    publish(EventType.ORDER_FILLED, {"id": 1})

    bus = get_event_bus()
    sig_rows = list(bus.replay(Stream.SIGNALS))
    risk_rows = list(bus.replay(Stream.RISK))
    order_rows = list(bus.replay(Stream.ORDERS))
    assert len(sig_rows) == 1 and sig_rows[0][1].event_type == EventType.SIGNAL_GENERATED
    assert len(risk_rows) == 1 and risk_rows[0][1].event_type == EventType.RISK_BREACH
    assert len(order_rows) == 1 and order_rows[0][1].event_type == EventType.ORDER_FILLED


# ── Redis-enabled path with mocked redis ───────────────────────────────────


def test_redis_enabled_publish_round_trips_via_xadd(monkeypatch) -> None:
    """``EVENT_BUS_ENABLED=1`` + redis available → RedisEventBus is used.
    publish hits xadd; replay parses xrange."""
    monkeypatch.setenv("EVENT_BUS_ENABLED", "1")
    client = MagicMock()
    client.xadd.return_value = "1700000000000-0"
    client.xrange.return_value = [
        ("1700000000000-0", {
            "event_type": EventType.ORDER_FILLED,
            "ts": "2024-01-01T00:00:00Z",
            "payload": json.dumps({"ticker": "AAPL", "qty": 10}),
        }),
    ]
    monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))

    msg_id = publish(EventType.ORDER_FILLED, {"ticker": "AAPL", "qty": 10})
    assert msg_id == "1700000000000-0"
    client.xadd.assert_called_once()

    bus = get_event_bus()
    rows = list(bus.replay(Stream.ORDERS))
    assert len(rows) == 1
    assert rows[0][1].payload == {"ticker": "AAPL", "qty": 10}


# ── Failure injection: publish swallows backend exceptions ─────────────────


def test_publish_swallows_backend_exception(monkeypatch) -> None:
    """When the backend's publish raises, the helper logs and returns
    None — the cron must not crash on a flaky bus."""
    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    bus = get_event_bus()  # InMemoryEventBus

    def _raise(*args, **kwargs):
        raise RuntimeError("bus down")

    monkeypatch.setattr(bus, "publish", _raise)
    out = publish(EventType.RISK_BREACH, {"drawdown": -0.20})
    assert out is None


def test_redis_init_failure_falls_back_to_memory(monkeypatch) -> None:
    """``EVENT_BUS_ENABLED=1`` but RedisEventBus init raises → fall
    back to InMemoryEventBus with a warning log."""
    monkeypatch.setenv("EVENT_BUS_ENABLED", "1")
    monkeypatch.setattr(event_bus, "_redis", MagicMock())
    monkeypatch.setattr(
        event_bus,
        "RedisEventBus",
        lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("redis unreachable"),
        ),
    )
    bus = get_event_bus()
    assert isinstance(bus, InMemoryEventBus)
    # Bus is still usable after fallback
    msg_id = publish(EventType.RISK_BREACH, {"drawdown": -0.15})
    assert msg_id is not None
