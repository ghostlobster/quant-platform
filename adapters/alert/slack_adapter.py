"""Slack alert adapter — delegates to alerts.channels broadcast."""
from __future__ import annotations

import os
from typing import Optional


class SlackAlertAdapter:
    def __init__(self) -> None:
        self._webhook_url = os.environ.get("SLACK_WEBHOOK_URL", "")
        if not self._webhook_url:
            raise ValueError(
                "SLACK_WEBHOOK_URL environment variable is required for SlackAlertAdapter"
            )
        try:
            import requests

            self._requests = requests
        except ImportError as e:
            raise ImportError("requests not installed. Run: pip install requests") from e

    def send(self, message: str, *, level: str = "info", channel: Optional[str] = None) -> bool:
        payload: dict = {"text": f"[{level.upper()}] {message}"}
        if channel:
            payload["channel"] = channel
        try:
            r = self._requests.post(self._webhook_url, json=payload, timeout=10)
            r.raise_for_status()
            return True
        except Exception:
            return False
