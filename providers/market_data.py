"""
providers/market_data.py — MarketDataProvider protocol and factory.

ENV vars
--------
    MARKET_DATA_PROVIDER   alpaca | yfinance | mock  (default: yfinance)
    ALPACA_API_KEY, ALPACA_SECRET_KEY  (required for alpaca adapter)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class MarketDataProvider(Protocol):
    """Duck-typed interface for OHLCV bars and real-time quotes."""

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """
        Return OHLCV bars for *symbol* between *start* and *end*.

        Parameters
        ----------
        symbol    : ticker, e.g. ``"AAPL"``
        timeframe : resolution string, e.g. ``"1Day"``, ``"1Hour"``
        start     : ISO-8601 date string, e.g. ``"2024-01-01"``
        end       : ISO-8601 date string, e.g. ``"2024-12-31"``

        Returns
        -------
        list of dicts with keys: t, o, h, l, c, v
        """
        ...

    def get_quote(self, symbol: str) -> dict:
        """Return the latest quote for *symbol*."""
        ...

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Return latest quotes for multiple symbols keyed by symbol."""
        ...


def get_market_data(provider: Optional[str] = None) -> MarketDataProvider:
    """
    Return a configured MarketDataProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the MARKET_DATA_PROVIDER env var.  One of:
        ``alpaca``, ``yfinance``, ``mock``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (
        provider or os.environ.get("MARKET_DATA_PROVIDER", "yfinance")
    ).lower().strip()
    if name == "alpaca":
        from adapters.market_data.alpaca_adapter import AlpacaMarketDataAdapter
        return AlpacaMarketDataAdapter()
    if name == "yfinance":
        from adapters.market_data.yfinance_adapter import YFinanceAdapter
        return YFinanceAdapter()
    if name == "mock":
        from adapters.market_data.mock_adapter import MockMarketDataAdapter
        return MockMarketDataAdapter()
    raise ValueError(
        f"Unknown market data provider: {name!r}. "
        "Valid options: alpaca, yfinance, mock"
    )
