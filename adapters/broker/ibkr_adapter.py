"""IBKR broker adapter — stub for Interactive Brokers integration."""
from __future__ import annotations

from typing import Optional


class IBKRAdapter:
    """Interactive Brokers adapter via ib_insync or TWS API."""

    def __init__(self) -> None:
        try:
            import ib_insync  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "ib_insync not installed. Run: pip install ib_insync"
            ) from e

    def get_account_info(self) -> dict:
        raise NotImplementedError("IBKR adapter not fully implemented")

    def get_positions(self) -> list[dict]:
        raise NotImplementedError("IBKR adapter not fully implemented")

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        raise NotImplementedError("IBKR adapter not fully implemented")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("IBKR adapter not fully implemented")

    def get_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("IBKR adapter not fully implemented")
