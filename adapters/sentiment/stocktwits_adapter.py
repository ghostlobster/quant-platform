"""
adapters/sentiment/stocktwits_adapter.py — SentimentProvider using Stocktwits public API.

No API key required for the public endpoints (rate-limited).

REST endpoint: https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

_BASE = "https://api.stocktwits.com/api/2"
_BULL_SCORE =  0.7
_BEAR_SCORE = -0.7


class StocktwitsAdapter:
    """SentimentProvider that pulls bullish/bearish signals from Stocktwits."""

    def _fetch_messages(self, symbol: str, limit: int = 30) -> list[dict]:
        url = f"{_BASE}/streams/symbol/{symbol.upper()}.json?limit={limit}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quant-platform/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return data.get("messages", [])
        except Exception as exc:
            logger.warning("Stocktwits fetch failed for %s: %s", symbol, exc)
            return []

    def _message_score(self, msg: dict) -> Optional[float]:
        entities = msg.get("entities", {})
        sentiment = entities.get("sentiment")
        if sentiment is None:
            return None
        basic = sentiment.get("basic", "").upper()
        if basic == "Bullish".upper():
            return _BULL_SCORE
        if basic == "Bearish".upper():
            return _BEAR_SCORE
        return None

    def score(self, text: str) -> float:
        """
        Score arbitrary text by checking for bull/bear keywords.
        Stocktwits is primarily a ticker stream; use vader for free text.
        """
        lower = text.lower()
        bull = sum(w in lower for w in ("bullish", "buy", "long", "moon", "calls"))
        bear = sum(w in lower for w in ("bearish", "sell", "short", "puts", "crash"))
        total = bull + bear
        if total == 0:
            return 0.0
        return (bull - bear) / total * 0.7

    def batch_score(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        """Return sentiment score for symbol (30-min SQLite cache)."""
        from adapters.sentiment.cache import cache_read, cache_write

        cached = cache_read(symbol, "stocktwits")
        if cached is not None:
            return cached

        messages = self._fetch_messages(symbol)
        scored = [self._message_score(m) for m in messages]
        valid = [s for s in scored if s is not None]
        score = sum(valid) / len(valid) if valid else 0.0
        cache_write(symbol, "stocktwits", score)
        return score
