"""agents/sentiment_agent.py — Specialist agent wrapping the sentiment provider."""
from __future__ import annotations

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentAgent:
    """Returns a signal based on ticker-level sentiment score."""

    name = "sentiment_agent"

    def run(self, context: dict) -> AgentSignal:
        ticker = context.get("ticker", "SPY")
        try:
            from providers.sentiment import get_sentiment
            provider = get_sentiment()
            score = provider.ticker_sentiment(ticker)
        except Exception as exc:
            logger.debug("SentimentAgent: sentiment fetch failed: %s", exc)
            return AgentSignal(
                agent_name=self.name,
                signal="neutral",
                confidence=0.3,
                reasoning="Sentiment data unavailable",
            )

        if score > 0.2:
            signal, confidence = "bullish", min(0.9, 0.5 + score)
        elif score < -0.2:
            signal, confidence = "bearish", min(0.9, 0.5 + abs(score))
        else:
            signal, confidence = "neutral", 0.5

        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning=f"Sentiment score for {ticker}: {score:.3f}",
            metadata={"score": score},
        )
