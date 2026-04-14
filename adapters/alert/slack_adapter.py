"""
adapters/alert/slack_adapter.py — AlertProvider wrapping Slack incoming webhook.

ENV vars
--------
    SLACK_WEBHOOK_URL   Slack incoming webhook URL
    SLACK_CHANNEL       override channel (e.g. #alerts) — optional
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    "info": ":information_source:",
    "warning": ":warning:",
    "error": ":red_circle:",
}


class SlackAlertAdapter:
    """AlertProvider that sends messages to a Slack incoming webhook."""

    def __init__(
        self,
        webhook_url: str | None = None,
        default_channel: str | None = None,
    ) -> None:
        if _requests is None:
            raise ImportError(
                "requests package is required for SlackAlertAdapter. "
                "Install it with: pip install requests"
            )
        self._webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
        if not self._webhook_url:
            raise ValueError(
                "SLACK_WEBHOOK_URL must be set to use SlackAlertAdapter."
            )
        self._default_channel = (
            default_channel or os.environ.get("SLACK_CHANNEL", "")
        )

    def send(
        self,
        message: str,
        *,
        level: str = "info",
        channel: Optional[str] = None,
    ) -> bool:
        emoji = _LEVEL_EMOJI.get(level.lower(), ":bell:")
        text = f"{emoji} *[{level.upper()}]* {message}"
        payload: dict = {"text": text}
        dest = channel or self._default_channel
        if dest:
            payload["channel"] = dest
        try:
            resp = _requests.post(
                self._webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=10,
            )
            if resp.status_code == 200:
                logger.debug("Slack alert sent [%s]", level)
                return True
            logger.error("Slack webhook returned HTTP %d", resp.status_code)
            return False
        except Exception as exc:
            logger.error("Slack send failed: %s", exc)
            return False
