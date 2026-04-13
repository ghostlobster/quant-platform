from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AlertProvider(Protocol):
    def send(self, message: str, *, level: str = "info", channel: Optional[str] = None) -> bool: ...


def get_alert_channel(provider: Optional[str] = None) -> AlertProvider:
    name = (provider or os.environ.get("ALERT_PROVIDER", "noop")).lower()
    if name == "slack":
        from adapters.alert.slack_adapter import SlackAlertAdapter
        return SlackAlertAdapter()
    elif name == "email":
        from adapters.alert.email_adapter import EmailAlertAdapter
        return EmailAlertAdapter()
    elif name == "noop":
        from adapters.alert.noop_adapter import NoopAlertAdapter
        return NoopAlertAdapter()
    raise ValueError(f"Unknown alert provider: {name!r}. Valid: slack, email, noop")
