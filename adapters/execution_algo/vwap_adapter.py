"""Volume-Weighted Average Price: slices order proportionally to expected volume."""
from __future__ import annotations

import time
from typing import Any


class VWAPAdapter:
    """
    Simplified VWAP: distributes the order into equal slices over duration.
    A production implementation would weight slices by historical intraday volume profiles.
    """

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
                time.sleep(interval)
        return results
