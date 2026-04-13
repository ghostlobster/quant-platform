"""Time-Weighted Average Price: splits order into N equal slices over duration."""
from __future__ import annotations

import time
from typing import Any


class TWAPAdapter:
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
        slices = kwargs.get("slices", max(2, duration_minutes))
        slice_qty = total_qty / slices
        interval = (duration_minutes * 60) / slices
        results = []
        for i in range(slices):
            result = broker.place_order(symbol, slice_qty, side, order_type="market")
            results.append(result)
            if i < slices - 1:
                time.sleep(interval)  # in real use; tests mock this
        return results
