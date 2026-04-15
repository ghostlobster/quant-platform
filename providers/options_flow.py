"""
providers/options_flow.py — OptionsFlowProvider protocol and factory.

Provides unusual options activity signals for use in the screener and Greeks tab.

ENV vars
--------
    OPTIONS_FLOW_PROVIDER   thetadata | unusual_whales | mock  (default: mock)
    THETADATA_API_KEY       ThetaData API key
    UNUSUAL_WHALES_TOKEN    Unusual Whales API token
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

__all__ = ["OptionsFlowProvider", "OptionsFlowResult", "get_options_flow"]


@dataclass
class OptionsFlowResult:
    """Summary of recent options flow for a symbol."""

    symbol: str
    call_volume: float
    put_volume: float
    call_put_ratio: float           # > 1.0 = bullish flow
    unusual_score: float            # [-1.0, 1.0]: +1 = extremely bullish flow
    avg_20d_call_put_ratio: float   # baseline for comparison
    is_unusual: bool                # True if current ratio > 1.5x or < 0.67x baseline


@runtime_checkable
class OptionsFlowProvider(Protocol):
    """Duck-typed interface for options flow data providers."""

    def get_flow(self, symbol: str, lookback_days: int = 1) -> list[dict]:
        """Return raw options flow records for *symbol*."""
        ...

    def unusual_activity_score(self, symbol: str) -> OptionsFlowResult:
        """Return a summarised unusual activity score for *symbol*."""
        ...


def get_options_flow(provider: Optional[str] = None) -> OptionsFlowProvider:
    """
    Return a configured OptionsFlowProvider.

    Parameters
    ----------
    provider : str, optional
        Override OPTIONS_FLOW_PROVIDER env var.
        One of: 'thetadata', 'unusual_whales', 'mock'.
    """
    name = (
        provider or os.environ.get("OPTIONS_FLOW_PROVIDER", "mock")
    ).lower().strip()

    if name == "thetadata":
        from adapters.options_flow.thetadata_adapter import ThetaDataAdapter
        return ThetaDataAdapter()
    if name in ("unusual_whales", "unusual-whales"):
        from adapters.options_flow.unusual_whales_adapter import UnusualWhalesAdapter
        return UnusualWhalesAdapter()
    if name == "mock":
        from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter
        return MockOptionsFlowAdapter()
    raise ValueError(
        f"Unknown options flow provider: {name!r}. "
        "Valid options: thetadata, unusual_whales, mock"
    )
