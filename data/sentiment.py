"""
News sentiment scoring using keyword-based VADER-style approach.
Falls back gracefully if transformers/VADER not installed.
"""
import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Simple financial sentiment lexicon (positive/negative word lists)
_POSITIVE = {
    "beat", "beats", "exceeded", "record", "growth", "profit", "surge",
    "rally", "upgrade", "buy", "bullish", "strong", "outperform", "raised",
    "dividend", "acquisition", "partnership", "innovation", "expansion",
    "revenue", "earnings", "positive", "optimistic", "momentum", "recovery",
}
_NEGATIVE = {
    "miss", "misses", "disappointed", "loss", "decline", "drop", "fall",
    "downgrade", "sell", "bearish", "weak", "underperform", "cut", "layoff",
    "lawsuit", "investigation", "fraud", "default", "debt", "risk", "warning",
    "recall", "scandal", "concern", "uncertainty", "volatility", "recession",
}


@dataclass
class SentimentScore:
    text: str
    score: float          # -1.0 (very negative) to +1.0 (very positive)
    label: str            # 'positive', 'negative', 'neutral'
    confidence: float     # 0.0 - 1.0
    method: str           # 'lexicon' or 'transformer'
    positive_words: List[str] = field(default_factory=list)
    negative_words: List[str] = field(default_factory=list)


def _lexicon_score(text: str) -> SentimentScore:
    """Score text using keyword matching (always available, no dependencies)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    pos_hits = list(words & _POSITIVE)
    neg_hits = list(words & _NEGATIVE)
    pos_count = len(pos_hits)
    neg_count = len(neg_hits)
    total = pos_count + neg_count

    if total == 0:
        return SentimentScore(text=text[:100], score=0.0, label='neutral',
                              confidence=0.5, method='lexicon')

    score = (pos_count - neg_count) / total
    confidence = min(total / 10, 1.0)  # more hits = more confidence

    if score > 0.1:
        label = 'positive'
    elif score < -0.1:
        label = 'negative'
    else:
        label = 'neutral'

    return SentimentScore(
        text=text[:100], score=score, label=label,
        confidence=confidence, method='lexicon',
        positive_words=pos_hits, negative_words=neg_hits,
    )


def score_text(text: str, use_transformer: bool = False) -> SentimentScore:
    """
    Score a single text snippet.
    Tries transformer (FinBERT) if requested and available, else falls back to lexicon.
    """
    if not text or not text.strip():
        return SentimentScore(text='', score=0.0, label='neutral',
                              confidence=0.0, method='lexicon')

    if use_transformer:
        try:
            from transformers import pipeline
            pipe = pipeline("sentiment-analysis",
                            model="ProsusAI/finbert",
                            max_length=512, truncation=True)
            result = pipe(text[:512])[0]
            label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            score = label_map.get(result["label"].lower(), 0.0)
            return SentimentScore(
                text=text[:100], score=score,
                label=result["label"].lower(),
                confidence=float(result["score"]),
                method='transformer',
            )
        except Exception as e:
            logger.warning(f"Transformer unavailable, using lexicon: {e}")

    return _lexicon_score(text)


def score_headlines(headlines: List[str],
                    use_transformer: bool = False) -> dict:
    """
    Score a list of headlines and return aggregate sentiment.
    Returns: {avg_score, label, positive_count, negative_count, neutral_count, scores}
    """
    if not headlines:
        return {"avg_score": 0.0, "label": "neutral",
                "positive_count": 0, "negative_count": 0,
                "neutral_count": 0, "scores": []}

    scores = [score_text(h, use_transformer) for h in headlines]
    avg = sum(s.score for s in scores) / len(scores)
    counts = {
        "positive_count": sum(1 for s in scores if s.label == "positive"),
        "negative_count": sum(1 for s in scores if s.label == "negative"),
        "neutral_count":  sum(1 for s in scores if s.label == "neutral"),
    }
    label = "positive" if avg > 0.1 else "negative" if avg < -0.1 else "neutral"
    return {"avg_score": float(avg), "label": label, **counts, "scores": scores}


def get_ticker_sentiment(ticker: str) -> dict:
    """
    Fetch recent news headlines for a ticker via yfinance and score them.
    Returns aggregate sentiment dict.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        news = t.news or []
        headlines = []
        for item in news[:20]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                headlines.append(title)
        if not headlines:
            return {"avg_score": 0.0, "label": "neutral", "headline_count": 0}
        result = score_headlines(headlines)
        result["headline_count"] = len(headlines)
        result["ticker"] = ticker
        return result
    except Exception as e:
        logger.warning(f"Could not fetch news for {ticker}: {e}")
        return {"avg_score": 0.0, "label": "neutral", "headline_count": 0}
