"""
adapters/sentiment/mock_adapter.py — Deterministic mock sentiment for tests.

No external SDK required.  Always returns 0.5 (mildly positive).
"""
from __future__ import annotations


class MockSentimentAdapter:
    """SentimentProvider that returns fixed deterministic scores."""

    _SCORE = 0.5

    def score(self, text: str) -> float:
        return self._SCORE

    def batch_score(self, texts: list[str]) -> list[float]:
        return [self._SCORE] * len(texts)

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        return self._SCORE
