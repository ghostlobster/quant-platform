"""
providers/alert.py — AlertProvider protocol and factory.

ENV vars
--------
    ALERT_PROVIDER   noop | slack | email  (default: noop)
    SLACK_WEBHOOK_URL            (required for slack adapter)
    EMAIL_USERNAME, EMAIL_PASSWORD, EMAIL_TO  (required for email adapter)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class AlertProvider(Protocol):
    """Duck-typed interface for sending operational alerts."""

    def send(
        self,
        message: str,
        *,
        level: str = "info",
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send an alert message.

        Parameters
        ----------
        message : alert body text
        level   : severity — ``"info"``, ``"warning"``, ``"error"``
        channel : optional destination override (channel name, email address, etc.)

        Returns
        -------
        True if the alert was delivered successfully, False otherwise.
        """
        ...


def get_alert_channel(provider: Optional[str] = None) -> AlertProvider:
    """
    Return a configured AlertProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the ALERT_PROVIDER env var.  One of:
        ``noop``, ``slack``, ``email``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (provider or os.environ.get("ALERT_PROVIDER", "noop")).lower().strip()
    if name == "noop":
        from adapters.alert.noop_adapter import NoopAlertAdapter
        return NoopAlertAdapter()
    if name == "slack":
        from adapters.alert.slack_adapter import SlackAlertAdapter
        return SlackAlertAdapter()
    if name == "email":
        from adapters.alert.email_adapter import EmailAlertAdapter
        return EmailAlertAdapter()
    raise ValueError(
        f"Unknown alert provider: {name!r}. "
        "Valid options: noop, slack, email"
    )
