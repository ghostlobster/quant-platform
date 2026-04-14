"""
providers/sentiment.py — SentimentProvider protocol and factory.

ENV vars
--------
    SENTIMENT_PROVIDER   vader | stocktwits | mock  (default: vader)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class SentimentProvider(Protocol):
    """Duck-typed interface for text sentiment scoring."""

    def score(self, text: str) -> float:
        """
        Score a single text snippet.

        Returns
        -------
        float in [-1.0, 1.0] where -1.0 is very negative, 1.0 is very positive.
        """
        ...

    def batch_score(self, texts: list[str]) -> list[float]:
        """Score multiple texts. Returns a list of floats in [-1.0, 1.0]."""
        ...

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        """
        Aggregate sentiment for *symbol* over the past *lookback_hours*.

        Returns a float in [-1.0, 1.0].
        """
        ...


def get_sentiment(provider: Optional[str] = None) -> SentimentProvider:
    """
    Return a configured SentimentProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the SENTIMENT_PROVIDER env var.  One of:
        ``vader``, ``stocktwits``, ``mock``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (
        provider or os.environ.get("SENTIMENT_PROVIDER", "vader")
    ).lower().strip()
    if name == "vader":
        from adapters.sentiment.vader_adapter import VADERAdapter
        return VADERAdapter()
    if name == "stocktwits":
        from adapters.sentiment.stocktwits_adapter import StocktwitsAdapter
        return StocktwitsAdapter()
    if name == "mock":
        from adapters.sentiment.mock_adapter import MockSentimentAdapter
        return MockSentimentAdapter()
    raise ValueError(
        f"Unknown sentiment provider: {name!r}. "
        "Valid options: vader, stocktwits, mock"
    )
