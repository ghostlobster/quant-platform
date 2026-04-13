import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.sentiment import score_headlines, score_text


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
