import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
from data.sentiment import score_text, score_headlines, get_ticker_sentiment, SentimentScore, _lexicon_score


def test_positive_headline():
    s = score_text("Company beats earnings expectations with record profit growth")
    assert s.label == "positive"
    assert s.score > 0

def test_negative_headline():
    s = score_text("Company misses earnings, announces layoffs and investigation")
    assert s.label == "negative"
    assert s.score < 0

def test_neutral_headline():
    s = score_text("Company announces quarterly results")
    assert s.label == "neutral"

def test_empty_text():
    s = score_text("")
    assert s.score == 0.0
    assert s.label == "neutral"

def test_score_headlines_aggregate():
    headlines = [
        "Record profits beat expectations",
        "Strong growth momentum continues",
        "Company misses revenue targets",
    ]
    result = score_headlines(headlines)
    assert "avg_score" in result
    assert "label" in result
    assert result["positive_count"] + result["negative_count"] + result["neutral_count"] == 3

def test_empty_headlines():
    result = score_headlines([])
    assert result["avg_score"] == 0.0

def test_confidence_range():
    s = score_text("Strong record growth beats all expectations")
    assert 0.0 <= s.confidence <= 1.0


# --- transformer path ---

def test_score_text_transformer_unavailable_falls_back_to_lexicon():
    """When transformers is not installed, score_text falls back to lexicon."""
    with patch.dict("sys.modules", {"transformers": None}):
        s = score_text("Company beats profit record", use_transformer=True)
    assert s.method == "lexicon"


def test_score_text_transformer_exception_falls_back():
    """Any error inside the transformer path falls back to lexicon."""
    mock_pipeline = MagicMock(side_effect=RuntimeError("GPU OOM"))
    mock_transformers = MagicMock()
    mock_transformers.pipeline = mock_pipeline
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        s = score_text("Company beats earnings", use_transformer=True)
    assert s.method == "lexicon"


def test_score_text_transformer_success():
    """Mocked transformer returns a transformer-labelled result."""
    mock_result = [{"label": "positive", "score": 0.97}]
    mock_pipe = MagicMock(return_value=mock_result)
    mock_transformers = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=mock_pipe)
    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        s = score_text("Record earnings beat", use_transformer=True)
    assert s.method == "transformer"
    assert s.score == 1.0
    assert s.confidence == pytest.approx(0.97)


# --- get_ticker_sentiment ---

def test_get_ticker_sentiment_with_news():
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {"title": "Company beats earnings record"},
        {"title": "Strong growth momentum"},
    ]
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_ticker_sentiment("AAPL")
    assert result["headline_count"] == 2
    assert "avg_score" in result
    assert result["ticker"] == "AAPL"


def test_get_ticker_sentiment_no_news():
    mock_ticker = MagicMock()
    mock_ticker.news = []
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_ticker_sentiment("AAPL")
    assert result["avg_score"] == 0.0
    assert result["label"] == "neutral"
    assert result["headline_count"] == 0


def test_get_ticker_sentiment_news_with_content_field():
    """Some yfinance versions nest the title inside 'content'."""
    mock_ticker = MagicMock()
    mock_ticker.news = [
        {"content": {"title": "Company misses earnings targets"}},
    ]
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = get_ticker_sentiment("TSLA")
    assert result["headline_count"] == 1


def test_get_ticker_sentiment_yfinance_exception():
    with patch("yfinance.Ticker", side_effect=Exception("network error")):
        result = get_ticker_sentiment("AAPL")
    assert result["avg_score"] == 0.0
    assert result["label"] == "neutral"
