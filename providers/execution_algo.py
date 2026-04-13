from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class ExecutionAlgoProvider(Protocol):
    def execute(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        broker: Any,
        *,
        duration_minutes: int = 30,
        **kwargs: Any,
    ) -> list[dict]: ...


def get_execution_algo(algo: Optional[str] = None) -> ExecutionAlgoProvider:
    name = (algo or os.environ.get("EXECUTION_ALGO", "market")).lower()
    if name == "market":
        from adapters.execution_algo.market_adapter import MarketOrderAdapter
        return MarketOrderAdapter()
    elif name == "twap":
        from adapters.execution_algo.twap_adapter import TWAPAdapter
        return TWAPAdapter()
    elif name == "vwap":
        from adapters.execution_algo.vwap_adapter import VWAPAdapter
        return VWAPAdapter()
    raise ValueError(f"Unknown execution algo: {name!r}. Valid: market, twap, vwap")
