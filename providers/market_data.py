from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]: ...
    def get_quote(self, symbol: str) -> dict: ...
    def get_quotes(self, symbols: list[str]) -> dict[str, dict]: ...


def get_market_data(provider: Optional[str] = None) -> MarketDataProvider:
    name = (provider or os.environ.get("MARKET_DATA_PROVIDER", "mock")).lower()
    if name == "alpaca":
        from adapters.market_data.alpaca_adapter import AlpacaMarketDataAdapter
        return AlpacaMarketDataAdapter()
    elif name == "yfinance":
        from adapters.market_data.yfinance_adapter import YFinanceAdapter
        return YFinanceAdapter()
    elif name == "mock":
        from adapters.market_data.mock_adapter import MockMarketDataAdapter
        return MockMarketDataAdapter()
    raise ValueError(f"Unknown market data provider: {name!r}. Valid: alpaca, yfinance, mock")
