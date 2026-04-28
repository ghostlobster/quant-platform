"""
tests/test_event_bus.py — bus.event_bus + bus.events round-trip.

Coverage target ≥ 85 % combined line+branch (closes #217). Lifted from
the 66 % baseline by adding mock-Redis tests that exercise the Redis
backend's publish / replay / subscribe paths plus the factory's
"Redis-enabled" success path. The few remaining unreachable lines are
the defensive double-checked-lock branch in ``get_event_bus`` and the
``BUSYGROUP`` re-raise in ``RedisEventBus.subscribe`` (already marked
``# pragma: no cover``).
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bus import event_bus
from bus.event_bus import (
    InMemoryEventBus,
    RedisEventBus,
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
    """``EVENT_BUS_ENABLED=1`` + ``_redis`` available + RedisEventBus
    raises on init → fall back to InMemoryEventBus with a warning log
    (covers lines 216-221)."""
    monkeypatch.setenv("EVENT_BUS_ENABLED", "1")
    # Ensure _redis is non-None so the try/except branch is reached.
    monkeypatch.setattr(event_bus, "_redis", MagicMock())

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


# ── Redis backend (mocked redis-py) ─────────────────────────────────────────

def _fake_redis_module(client: MagicMock) -> MagicMock:
    """Return a stand-in for the ``redis`` package whose ``from_url``
    yields the supplied client mock. Keeps every test that needs the
    Redis path hermetic — no socket, no subprocess."""
    fake = MagicMock()
    fake.from_url.return_value = client
    return fake


class TestRedisEventBus:
    def test_init_raises_when_redis_package_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(event_bus, "_redis", None)
        with pytest.raises(ImportError, match="redis package is required"):
            RedisEventBus()

    def test_init_uses_env_url_and_default_maxlen(self, monkeypatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://example:6379/2")
        monkeypatch.delenv("EVENT_BUS_MAX_LEN", raising=False)
        client = MagicMock()
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus()
        assert bus._url == "redis://example:6379/2"
        assert bus._max_len == 100_000  # documented default
        assert bus._client is client

    def test_init_explicit_url_overrides_env(self, monkeypatch) -> None:
        monkeypatch.setenv("REDIS_URL", "redis://shouldnt-win:6379/0")
        client = MagicMock()
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus(url="redis://wins:1234/9", max_len=42)
        assert bus._url == "redis://wins:1234/9"
        assert bus._max_len == 42

    def test_init_zero_maxlen_disables_capping(self, monkeypatch) -> None:
        client = MagicMock()
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus(max_len=0)
        assert bus._max_len == 0

    def test_publish_calls_xadd_with_maxlen(self, monkeypatch) -> None:
        client = MagicMock()
        client.xadd.return_value = "1700000000000-0"
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus(max_len=500)
        out = bus.publish(
            Event(EventType.SIGNAL_GENERATED, {"ticker": "AAPL"}),
            stream="signals",
        )
        assert out == "1700000000000-0"
        client.xadd.assert_called_once()
        args, kwargs = client.xadd.call_args
        target, fields = args
        assert target == "signals"
        assert fields["event_type"] == EventType.SIGNAL_GENERATED
        assert json.loads(fields["payload"]) == {"ticker": "AAPL"}
        assert kwargs["maxlen"] == 500
        assert kwargs["approximate"] is True

    def test_publish_skips_maxlen_when_zero(self, monkeypatch) -> None:
        """``max_len=0`` → no maxlen kwarg passed to xadd (the truthy
        check on line 89 is the branch we're exercising)."""
        client = MagicMock()
        client.xadd.return_value = "1-0"
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus(max_len=0)
        bus.publish(Event(EventType.RISK_BREACH, {"dd": -0.1}))
        kwargs = client.xadd.call_args.kwargs
        assert "maxlen" not in kwargs
        assert "approximate" not in kwargs

    def test_publish_routes_to_default_stream_when_unspecified(
        self, monkeypatch
    ) -> None:
        client = MagicMock()
        client.xadd.return_value = "1-0"
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus()
        bus.publish(Event(EventType.RISK_BREACH, {"dd": -0.05}))
        target = client.xadd.call_args.args[0]
        assert target == Stream.RISK

    def test_replay_yields_decoded_events(self, monkeypatch) -> None:
        client = MagicMock()
        client.xrange.return_value = [
            ("100-0", {
                "event_type": EventType.SIGNAL_GENERATED,
                "ts": "2024-01-01T00:00:00Z",
                "payload": json.dumps({"ticker": "AAPL"}),
            }),
            ("101-0", {
                "event_type": EventType.ORDER_PLACED,
                "ts": "2024-01-01T00:00:01Z",
                "payload": json.dumps({"qty": 5}),
            }),
        ]
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus()
        rows = list(bus.replay("signals", since="50", limit=99))
        client.xrange.assert_called_once_with("signals", min="50", count=99)
        assert len(rows) == 2
        assert rows[0][1].event_type == EventType.SIGNAL_GENERATED
        assert rows[0][1].payload == {"ticker": "AAPL"}
        assert rows[1][1].payload == {"qty": 5}

    def test_subscribe_creates_group_and_yields(self, monkeypatch) -> None:
        client = MagicMock()
        # First xreadgroup returns empty (block timeout) → loop continues;
        # second returns one message; third raises StopIteration via
        # side_effect=BreakLoop sentinel.

        class _Stop(Exception):
            pass

        client.xreadgroup.side_effect = [
            [],  # block timeout — loop continues
            [(
                "signals",
                [(
                    "1-0",
                    {
                        "event_type": EventType.SIGNAL_GENERATED,
                        "ts": "now",
                        "payload": json.dumps({"ticker": "AAPL"}),
                    },
                )],
            )],
            _Stop(),  # bail the second iteration so the test terminates
        ]
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = RedisEventBus()

        # Drain at most 1 yielded message before _Stop fires.
        gen = bus.subscribe("signals", "g", "c", block_ms=10)
        msg_id, evt = next(gen)
        with pytest.raises(_Stop):
            next(gen)

        client.xgroup_create.assert_called_once_with(
            "signals", "g", id="0", mkstream=True
        )
        assert msg_id == "1-0"
        assert evt.event_type == EventType.SIGNAL_GENERATED
        assert evt.payload == {"ticker": "AAPL"}


# ── Factory: Redis-enabled path ─────────────────────────────────────────────

class TestFactoryRedisPath:
    def test_factory_picks_redis_when_enabled_and_available(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("EVENT_BUS_ENABLED", "1")
        client = MagicMock()
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = get_event_bus()
        assert isinstance(bus, RedisEventBus)

    def test_factory_falls_back_when_redis_pkg_missing(
        self, monkeypatch
    ) -> None:
        """``EVENT_BUS_ENABLED=1`` but ``_redis is None`` → in-memory."""
        monkeypatch.setenv("EVENT_BUS_ENABLED", "1")
        monkeypatch.setattr(event_bus, "_redis", None)
        bus = get_event_bus()
        assert isinstance(bus, InMemoryEventBus)

    @pytest.mark.parametrize("flag", ["0", "false", "no", "off", ""])
    def test_factory_disabled_flags_use_memory(
        self, monkeypatch, flag: str
    ) -> None:
        monkeypatch.setenv("EVENT_BUS_ENABLED", flag)
        bus = get_event_bus()
        assert isinstance(bus, InMemoryEventBus)

    @pytest.mark.parametrize("flag", ["1", "true", "yes", "on", "TRUE", "Yes"])
    def test_factory_enabled_flags_route_to_redis(
        self, monkeypatch, flag: str
    ) -> None:
        monkeypatch.setenv("EVENT_BUS_ENABLED", flag)
        client = MagicMock()
        monkeypatch.setattr(event_bus, "_redis", _fake_redis_module(client))
        bus = get_event_bus()
        assert isinstance(bus, RedisEventBus)


# ── Concurrency: singleton + in-memory bus are thread-safe ──────────────────

def test_inmemory_publish_under_concurrent_writers() -> None:
    """The internal ``threading.Lock`` should serialise publishes; with
    N threads writing K events each we expect exactly N*K rows on the
    stream and unique message ids."""
    import threading

    bus = InMemoryEventBus()
    n_threads = 8
    per_thread = 25

    def _worker() -> None:
        for i in range(per_thread):
            bus.publish(Event(EventType.SIGNAL_GENERATED, {"i": i}))

    threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    rows = list(bus.replay(Stream.SIGNALS, limit=10_000))
    assert len(rows) == n_threads * per_thread
    ids = [r[0] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate message ids — lock missing"


def test_factory_singleton_thread_safe(monkeypatch) -> None:
    """Two threads calling ``get_event_bus()`` see the same instance."""
    import threading

    monkeypatch.delenv("EVENT_BUS_ENABLED", raising=False)
    seen: list = []

    def _worker() -> None:
        seen.append(get_event_bus())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(b) for b in seen}) == 1


# ── _decode_event: defensive paths ──────────────────────────────────────────

def test_decode_event_falls_back_when_payload_is_none() -> None:
    """A row with no payload field decodes to an empty payload dict."""
    bus = InMemoryEventBus()
    bus._streams["custom"].append(
        ("999-0", {"event_type": "x.y", "ts": "now"}),
    )
    rows = list(bus.replay("custom"))
    assert rows[0][1].payload == {}
