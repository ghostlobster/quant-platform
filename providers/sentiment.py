from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SentimentProvider(Protocol):
    def score(self, text: str) -> float: ...
    def batch_score(self, texts: list[str]) -> list[float]: ...
    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float: ...


def get_sentiment(provider: Optional[str] = None) -> SentimentProvider:
    name = (provider or os.environ.get("SENTIMENT_PROVIDER", "mock")).lower()
    if name == "vader":
        from adapters.sentiment.vader_adapter import VaderSentimentAdapter
        return VaderSentimentAdapter()
    elif name == "stocktwits":
        from adapters.sentiment.stocktwits_adapter import StocktwitsAdapter
        return StocktwitsAdapter()
    elif name == "mock":
        from adapters.sentiment.mock_adapter import MockSentimentAdapter
        return MockSentimentAdapter()
    raise ValueError(f"Unknown sentiment provider: {name!r}. Valid: vader, stocktwits, mock")
