"""In-memory paper trading broker — no external dependencies."""
from __future__ import annotations

import uuid
from typing import Optional


class PaperBrokerAdapter:
    """Simple in-memory paper broker for testing and development."""

    def __init__(self) -> None:
        self._cash: float = 100_000.0
        self._positions: dict[str, dict] = {}
        self._orders: dict[str, dict] = {}

    def get_account_info(self) -> dict:
        market_value = sum(p["qty"] * p["avg_price"] for p in self._positions.values())
        return {
            "cash": self._cash,
            "buying_power": self._cash,
            "equity": self._cash + market_value,
            "market_value": market_value,
        }

    def get_positions(self) -> list[dict]:
        return list(self._positions.values())

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        fill_price = limit_price or 100.0  # mock fill at limit or $100
        order_id = str(uuid.uuid4())
        cost = qty * fill_price

        if side == "buy":
            self._cash -= cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos["qty"] + qty
                total_cost = pos["qty"] * pos["avg_price"] + cost
                pos["qty"] = total_qty
                pos["avg_price"] = total_cost / total_qty
            else:
                self._positions[symbol] = {"symbol": symbol, "qty": qty, "avg_price": fill_price}
        elif side == "sell":
            self._cash += cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                pos["qty"] -= qty
                if pos["qty"] <= 0:
                    del self._positions[symbol]

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": "filled",
            "filled_avg_price": fill_price,
        }
        self._orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id]["status"] = "cancelled"
            return True
        return False

    def get_orders(self, status: str = "open") -> list[dict]:
        if status == "all":
            return list(self._orders.values())
        return [o for o in self._orders.values() if o["status"] == status]
