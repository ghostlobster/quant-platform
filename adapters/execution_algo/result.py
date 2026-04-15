"""
adapters/execution_algo/result.py — Typed execution result dataclass.

ExecutionResult captures the outcome of any execution algorithm:
fill details, slippage metrics, and the algorithm used.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_SLIPPAGE_WARN_BPS = 100.0  # warn when slippage exceeds 1%


@dataclass
class ExecutionResult:
    """Immutable record of a completed execution algo run."""

    symbol: str
    side: str                   # 'buy' | 'sell'
    total_qty: float
    fills: list[dict] = field(default_factory=list)
    algo: str = "market"        # 'market' | 'twap' | 'vwap'
    decision_price: float = 0.0  # price at time of trading decision (pre-execution)
    avg_fill_price: float = 0.0  # weighted-average actual fill price
    slippage_bps: float = 0.0   # (avg_fill - decision) / decision * 1e4
    executed_at: float = field(default_factory=time.time)

    @classmethod
    def from_fills(
        cls,
        fills: list[dict],
        symbol: str,
        side: str,
        algo: str,
        decision_price: float = 0.0,
    ) -> "ExecutionResult":
        """
        Build an ExecutionResult from a list of raw broker fill dicts.

        Each fill dict is expected to contain at least a 'price' and 'qty'
        (or 'filled_qty') key.  Missing keys are handled gracefully.
        """
        total_qty = 0.0
        total_notional = 0.0

        for fill in fills:
            qty = float(
                fill.get("qty")
                or fill.get("filled_qty")
                or fill.get("shares")
                or 0
            )
            price = float(fill.get("price") or fill.get("fill_price") or 0)
            total_qty += qty
            total_notional += qty * price

        avg_fill = total_notional / total_qty if total_qty > 0 else 0.0

        if decision_price > 0 and avg_fill > 0:
            slippage_bps = (avg_fill - decision_price) / decision_price * 1e4
        else:
            slippage_bps = 0.0

        if abs(slippage_bps) > _SLIPPAGE_WARN_BPS:
            logger.warning(
                "High slippage detected: %s %s %.4f bps (decision=%.4f avg_fill=%.4f)",
                side.upper(), symbol, slippage_bps, decision_price, avg_fill,
            )

        return cls(
            symbol=symbol.upper(),
            side=side.lower(),
            total_qty=total_qty,
            fills=fills,
            algo=algo,
            decision_price=decision_price,
            avg_fill_price=round(avg_fill, 6),
            slippage_bps=round(slippage_bps, 4),
        )
