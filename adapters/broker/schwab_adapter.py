"""Schwab broker adapter — stub for Charles Schwab API integration."""
from __future__ import annotations

from typing import Optional


class SchwabAdapter:
    """Charles Schwab adapter via schwab-py or direct OAuth2 API."""

    def __init__(self) -> None:
        try:
            import schwab  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "schwab-py not installed. Run: pip install schwab-py"
            ) from e

    def get_account_info(self) -> dict:
        raise NotImplementedError("Schwab adapter not fully implemented")

    def get_positions(self) -> list[dict]:
        raise NotImplementedError("Schwab adapter not fully implemented")

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        raise NotImplementedError("Schwab adapter not fully implemented")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Schwab adapter not fully implemented")

    def get_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("Schwab adapter not fully implemented")
