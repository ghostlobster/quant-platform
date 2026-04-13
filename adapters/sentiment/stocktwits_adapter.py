"""StockTwits sentiment adapter."""
from __future__ import annotations


class StocktwitsAdapter:
    def __init__(self) -> None:
        try:
            import requests

            self._requests = requests
        except ImportError as e:
            raise ImportError("requests not installed. Run: pip install requests") from e
        self._base_url = "https://api.stocktwits.com/api/2"

    def score(self, text: str) -> float:
        # StockTwits doesn't score free text; return neutral
        return 0.0

    def batch_score(self, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        """Fetch recent StockTwits messages and compute bullish/bearish ratio."""
        try:
            r = self._requests.get(
                f"{self._base_url}/streams/symbol/{symbol}.json",
                params={"limit": 30},
                timeout=10,
            )
            r.raise_for_status()
            messages = r.json().get("messages", [])
            sentiments = [
                m.get("entities", {}).get("sentiment", {}).get("basic")
                for m in messages
                if m.get("entities", {}).get("sentiment")
            ]
            if not sentiments:
                return 0.0
            bullish = sentiments.count("Bullish")
            bearish = sentiments.count("Bearish")
            total = bullish + bearish
            return (bullish - bearish) / total if total > 0 else 0.0
        except Exception:
            return 0.0
