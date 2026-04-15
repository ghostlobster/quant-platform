"""
providers/execution_algo.py — ExecutionAlgoProvider protocol and factory.

ENV vars
--------
    EXECUTION_ALGO   market | twap | vwap  (default: market)
    TWAP_SLICE_SECONDS   (default: 60)
    VWAP_LOOKBACK_DAYS   (default: 5)
"""
from __future__ import annotations

import os
from typing import Any, Optional, Protocol, runtime_checkable

from adapters.execution_algo.result import ExecutionResult

__all__ = ["ExecutionAlgoProvider", "ExecutionResult", "get_execution_algo"]


@runtime_checkable
class ExecutionAlgoProvider(Protocol):
    """Duck-typed interface for order execution algorithms."""

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
        """
        Execute *total_qty* shares of *symbol* via *broker* over *duration_minutes*.

        Parameters
        ----------
        symbol           : ticker, e.g. ``"AAPL"``
        total_qty        : total shares to trade (positive)
        side             : ``"buy"`` or ``"sell"``
        broker           : any object implementing BrokerProvider
        duration_minutes : total execution window
        **kwargs         : algorithm-specific tuning parameters

        Returns
        -------
        list of dicts, one per child order placed.
        """
        ...


def get_execution_algo(algo: Optional[str] = None) -> ExecutionAlgoProvider:
    """
    Return a configured ExecutionAlgoProvider adapter.

    Parameters
    ----------
    algo : str, optional
        Override the EXECUTION_ALGO env var.  One of:
        ``market``, ``twap``, ``vwap``.

    Raises
    ------
    ValueError
        If the algo name is not recognised.
    """
    name = (algo or os.environ.get("EXECUTION_ALGO", "market")).lower().strip()
    if name == "market":
        from adapters.execution_algo.market_adapter import MarketAlgoAdapter
        return MarketAlgoAdapter()
    if name == "twap":
        from adapters.execution_algo.twap_adapter import TWAPAdapter
        return TWAPAdapter()
    if name == "vwap":
        from adapters.execution_algo.vwap_adapter import VWAPAdapter
        return VWAPAdapter()
    raise ValueError(
        f"Unknown execution algo: {name!r}. "
        "Valid options: market, twap, vwap"
    )
