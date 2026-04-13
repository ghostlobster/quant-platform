"""Broker adapter delegating to broker.alpaca_bridge."""
from __future__ import annotations

from typing import Optional


class AlpacaBrokerAdapter:
    def get_account_info(self) -> dict:
        from broker.alpaca_bridge import get_account

        return get_account() or {}

    def get_positions(self) -> list[dict]:
        from broker.alpaca_bridge import get_positions

        return get_positions()

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        from broker.alpaca_bridge import place_order

        order = place_order(symbol, qty, side, order_type=order_type, limit_price=limit_price)
        if order is None:
            return {}
        return {
            "order_id": order.order_id,
            "symbol": order.ticker,
            "qty": order.qty,
            "side": order.side,
            "order_type": order.order_type,
            "status": order.status,
            "filled_avg_price": order.filled_avg_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        from broker.alpaca_bridge import cancel_order

        return cancel_order(order_id)

    def get_orders(self, status: str = "open") -> list[dict]:
        from broker.alpaca_bridge import get_orders

        return get_orders(status=status)
