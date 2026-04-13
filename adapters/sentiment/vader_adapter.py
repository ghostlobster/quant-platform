class VaderSentimentAdapter:
    def __init__(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            self._analyzer = SentimentIntensityAnalyzer()
        except ImportError as e:
            raise ImportError(
                "vaderSentiment not installed. Run: pip install vaderSentiment"
            ) from e

    def score(self, text: str) -> float:
        return self._analyzer.polarity_scores(text)["compound"]

    def batch_score(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        # VADER can't fetch news; return neutral
        return 0.0
