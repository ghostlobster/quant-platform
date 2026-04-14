"""
adapters/alert/email_adapter.py — AlertProvider using SMTP (reuses alerts/channels.py).

ENV vars
--------
    EMAIL_SMTP_HOST   (default: smtp.gmail.com)
    EMAIL_SMTP_PORT   (default: 587)
    EMAIL_USERNAME
    EMAIL_PASSWORD
    EMAIL_TO
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class EmailAlertAdapter:
    """AlertProvider that sends alerts via SMTP, delegating to EmailChannel."""

    def __init__(self) -> None:
        from alerts.channels import EmailChannel
        username = os.environ.get("EMAIL_USERNAME", "")
        password = os.environ.get("EMAIL_PASSWORD", "")
        to_addr  = os.environ.get("EMAIL_TO", "")
        if not (username and password and to_addr):
            raise ValueError(
                "EMAIL_USERNAME, EMAIL_PASSWORD, and EMAIL_TO must all be set "
                "to use EmailAlertAdapter."
            )
        self._channel = EmailChannel(
            smtp_host  = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com"),
            smtp_port  = int(os.environ.get("EMAIL_SMTP_PORT", "587")),
            username   = username,
            password   = password,
            to_address = to_addr,
        )

    def send(
        self,
        message: str,
        *,
        level: str = "info",
        channel: Optional[str] = None,
    ) -> bool:
        subject = f"[{level.upper()}] Quant Platform Alert"
        return self._channel.send(subject=subject, body=message)
