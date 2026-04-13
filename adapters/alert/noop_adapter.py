from __future__ import annotations

from typing import Optional


class NoopAlertAdapter:
    def send(self, message: str, *, level: str = "info", channel: Optional[str] = None) -> bool:
        return True
