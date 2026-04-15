"""
adapters/execution_algo/market_adapter.py — Simple market order pass-through.

Sends the entire *total_qty* as a single market order immediately.
No external dependency.
"""
from __future__ import annotations

import logging
from typing import Any

from adapters.execution_algo.result import ExecutionResult

logger = logging.getLogger(__name__)


class MarketAlgoAdapter:
    """ExecutionAlgoProvider that places one market order for the full quantity."""

    def execute(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        broker: Any,
        *,
        duration_minutes: int = 30,
        decision_price: float = 0.0,
        **kwargs: Any,
    ) -> ExecutionResult:
        logger.info(
            "MarketAlgo: placing single market order %s %s x%.4f",
            side.upper(), symbol, total_qty,
        )
        fill = broker.place_order(
            symbol=symbol,
            qty=total_qty,
            side=side,
            order_type="market",
        )
        return ExecutionResult.from_fills(
            fills=[fill],
            symbol=symbol,
            side=side,
            algo="market",
            decision_price=decision_price,
        )
