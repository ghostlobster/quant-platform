"""
adapters/alert/noop_adapter.py — Silent no-op alert (default when no creds configured).

No external dependency.  All sends succeed silently.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class NoopAlertAdapter:
    """AlertProvider that discards all messages. Safe default for dev/test."""

    def send(
        self,
        message: str,
        *,
        level: str = "info",
        channel: Optional[str] = None,
    ) -> bool:
        logger.debug("NoopAlertAdapter.send [%s]: %s", level.upper(), message[:80])
        return True
