"""
adapters/execution_algo/twap_adapter.py — Time-Weighted Average Price execution.

Slices *total_qty* into equal child orders placed every TWAP_SLICE_SECONDS over
the *duration_minutes* window.

ENV vars
--------
    TWAP_SLICE_SECONDS   seconds between child orders (default: 60)
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class TWAPAdapter:
    """ExecutionAlgoProvider that slices orders evenly over a time window."""

    def __init__(self, slice_seconds: int | None = None) -> None:
        self._slice_seconds = slice_seconds or int(
            os.environ.get("TWAP_SLICE_SECONDS", "60")
        )

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
        duration_seconds = duration_minutes * 60
        n_slices = max(1, math.ceil(duration_seconds / self._slice_seconds))
        slice_qty = total_qty / n_slices
        # Round to 4 decimal places; last slice absorbs rounding error
        slice_qty = round(slice_qty, 4)

        logger.info(
            "TWAP: %s %s x%.4f in %d slices over %d min",
            side.upper(), symbol, total_qty, n_slices, duration_minutes,
        )

        fills: list[dict] = []
        placed = 0.0
        for i in range(n_slices):
            is_last = i == n_slices - 1
            qty = round(total_qty - placed, 4) if is_last else slice_qty
            if qty <= 0:
                break
            result = broker.place_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market",
            )
            fills.append(result)
            placed = round(placed + qty, 4)
            if not is_last:
                time.sleep(self._slice_seconds)

        return fills
