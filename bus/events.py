"""
bus/events.py — typed event constants + Event dataclass for the bus.

Event types are namespaced ``<domain>.<verb>`` strings so consumers can
filter with simple prefix matching. The set is intentionally small;
publishers stick to this enumeration so dashboards and the replay tool
can rely on stable identifiers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ── Event-type constants ────────────────────────────────────────────────────

class EventType:
    SIGNAL_GENERATED = "signal.generated"
    ORDER_PLACED     = "order.placed"
    ORDER_REJECTED   = "order.rejected"
    ORDER_FILLED     = "order.filled"
    RISK_BREACH      = "risk.breach"
    KILLSWITCH       = "kill.switch"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (
            cls.SIGNAL_GENERATED,
            cls.ORDER_PLACED,
            cls.ORDER_REJECTED,
            cls.ORDER_FILLED,
            cls.RISK_BREACH,
            cls.KILLSWITCH,
        )


# ── Stream constants ────────────────────────────────────────────────────────

class Stream:
    """Default stream names. Operators may override per-call."""

    SIGNALS  = "signals"
    ORDERS   = "orders"
    RISK     = "risk"

    @classmethod
    def for_event(cls, event_type: str) -> str:
        if event_type.startswith("signal."):
            return cls.SIGNALS
        if event_type.startswith("order.") or event_type.startswith("kill."):
            return cls.ORDERS
        if event_type.startswith("risk."):
            return cls.RISK
        return "events"  # generic fallback


# ── Event dataclass ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Event:
    """Single bus payload."""

    event_type: str
    payload: dict[str, Any]
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )

    def __post_init__(self) -> None:
        if not self.event_type or not isinstance(self.event_type, str):
            raise ValueError("event_type must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise ValueError(
                f"payload must be a dict, got {type(self.payload).__name__}",
            )
