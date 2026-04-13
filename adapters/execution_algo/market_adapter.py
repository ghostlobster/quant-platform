from __future__ import annotations

from typing import Any


class MarketOrderAdapter:
    """Immediate market order — no slicing."""

    def execute(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        broker: Any,
        *,
        duration_minutes: int = 30,
        **kwargs: Any,
    ) -> list[dict]:
        result = broker.place_order(symbol, total_qty, side, order_type="market")
        return [result]
