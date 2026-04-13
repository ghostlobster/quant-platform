class MockSentimentAdapter:
    def score(self, text: str) -> float:
        return 0.0

    def batch_score(self, texts: list[str]) -> list[float]:
        return [0.0] * len(texts)

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        return 0.0
