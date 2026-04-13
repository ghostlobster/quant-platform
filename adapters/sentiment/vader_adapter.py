"""
adapters/sentiment/vader_adapter.py — SentimentProvider using VADER.

Falls back to the project's existing lexicon scorer (data/sentiment.py) if
vaderSentiment is not installed — so this adapter always works.

Requires (optional):  pip install vaderSentiment
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


class VADERAdapter:
    """SentimentProvider backed by VADER (or project lexicon fallback)."""

    def __init__(self) -> None:
        if _VADER_AVAILABLE:
            self._analyzer = _VADER()
            self._method = "vader"
        else:
            logger.info(
                "vaderSentiment not installed; using built-in lexicon scorer. "
                "For better results: pip install vaderSentiment"
            )
            self._analyzer = None
            self._method = "lexicon"

    def score(self, text: str) -> float:
        """Return compound sentiment in [-1.0, 1.0]."""
        if not text or not text.strip():
            return 0.0
        if self._analyzer is not None:
            scores = self._analyzer.polarity_scores(text)
            return float(scores["compound"])
        # Fallback to project lexicon
        from data.sentiment import score_text
        result = score_text(text)
        return float(result.score)

    def batch_score(self, texts: list[str]) -> list[float]:
        return [self.score(t) for t in texts]

    def ticker_sentiment(self, symbol: str, lookback_hours: int = 24) -> float:
        """Fetch recent news via yfinance and score headlines."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol.upper())
            news = ticker.news or []
            headlines = []
            for item in news[:20]:
                title = (
                    item.get("title")
                    or item.get("content", {}).get("title", "")
                )
                if title:
                    headlines.append(title)
            if not headlines:
                return 0.0
            scores = self.batch_score(headlines)
            return sum(scores) / len(scores)
        except Exception as exc:
            logger.warning("ticker_sentiment failed for %s: %s", symbol, exc)
            return 0.0
