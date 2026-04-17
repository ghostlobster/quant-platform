"""Tests for strategies/sentiment_signal.py — sentiment alpha blend."""
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.sentiment_signal import (
    fetch_sentiment_scores,
    sentiment_alpha_scores,
)

# ── fetch_sentiment_scores ───────────────────────────────────────────────────

def test_fetch_empty_tickers_returns_empty_dict():
    assert fetch_sentiment_scores([]) == {}


def test_fetch_returns_neutral_when_provider_unavailable(monkeypatch):
    """When providers.sentiment.get_sentiment() raises, return zeros."""
    def _broken(*a, **k): raise RuntimeError("no provider configured")
    monkeypatch.setattr("providers.sentiment.get_sentiment", _broken)
    scores = fetch_sentiment_scores(["AAPL", "MSFT"])
    assert scores == {"AAPL": 0.0, "MSFT": 0.0}


def test_fetch_delegates_to_provider_ticker_sentiment(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [0.4, -0.2, 0.9]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = fetch_sentiment_scores(["AAPL", "MSFT", "GOOG"])
    assert scores == {"AAPL": 0.4, "MSFT": -0.2, "GOOG": 0.9}
    assert provider.ticker_sentiment.call_count == 3


def test_fetch_clips_to_unit_range(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [1.5, -3.0, 0.0]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = fetch_sentiment_scores(["A", "B", "C"])
    assert scores == {"A": 1.0, "B": -1.0, "C": 0.0}


def test_fetch_handles_per_ticker_errors(monkeypatch):
    """An exception on one ticker should not abort the whole batch."""
    provider = MagicMock()

    def _ticker_sentiment(ticker, lookback_hours):
        if ticker == "BAD":
            raise RuntimeError("lookup failed")
        return 0.3

    provider.ticker_sentiment.side_effect = _ticker_sentiment
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = fetch_sentiment_scores(["AAPL", "BAD", "MSFT"])
    assert scores["AAPL"] == pytest.approx(0.3)
    assert scores["BAD"] == 0.0
    assert scores["MSFT"] == pytest.approx(0.3)


def test_fetch_passes_lookback_hours(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.return_value = 0.0
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    fetch_sentiment_scores(["AAPL"], lookback_hours=72)
    _, kwargs = provider.ticker_sentiment.call_args
    assert kwargs.get("lookback_hours") == 72


def test_fetch_passes_provider_name(monkeypatch):
    captured: dict = {}

    def _get(provider_name=None, *a, **k):
        captured["name"] = provider_name
        provider = MagicMock()
        provider.ticker_sentiment.return_value = 0.0
        return provider

    monkeypatch.setattr("providers.sentiment.get_sentiment", _get)
    fetch_sentiment_scores(["AAPL"], provider_name="stocktwits")
    assert captured["name"] == "stocktwits"


# ── sentiment_alpha_scores ──────────────────────────────────────────────────

def test_alpha_scores_empty_universe_returns_empty():
    assert sentiment_alpha_scores([]) == {}


def test_alpha_scores_z_scores_across_universe(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [0.0, 0.5, -0.5]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = sentiment_alpha_scores(["A", "B", "C"])
    # Z-scoring of (0, 0.5, -0.5) → mean 0, std = 0.408; z values =
    # (0, 1.225, -1.225), clipped to [-1, 1] = (0.0, 1.0, -1.0).
    assert scores["A"] == pytest.approx(0.0, abs=1e-6)
    assert scores["B"] == pytest.approx(1.0, abs=1e-6)
    assert scores["C"] == pytest.approx(-1.0, abs=1e-6)


def test_alpha_scores_all_identical_returns_zero(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.return_value = 0.42   # every ticker same
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = sentiment_alpha_scores(["A", "B", "C"])
    for v in scores.values():
        assert v == 0.0


def test_alpha_scores_all_zero_returns_zero(monkeypatch):
    """Universe with no sentiment data collapses to neutral."""
    provider = MagicMock()
    provider.ticker_sentiment.return_value = 0.0
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = sentiment_alpha_scores(["A", "B", "C"])
    assert all(v == 0.0 for v in scores.values())


def test_alpha_scores_preserve_ticker_keys(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [0.1, -0.3, 0.4]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = sentiment_alpha_scores(["AAPL", "MSFT", "GOOG"])
    assert set(scores.keys()) == {"AAPL", "MSFT", "GOOG"}


def test_alpha_scores_bounded_in_unit_interval(monkeypatch):
    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [1.0, -1.0, 0.2, 0.8]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    scores = sentiment_alpha_scores(["A", "B", "C", "D"])
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_alpha_scores_blend_compatibility_with_ensemble(monkeypatch):
    """The output dict must be a drop-in source for blend_signals."""
    from strategies.ensemble_signal import blend_signals

    provider = MagicMock()
    provider.ticker_sentiment.side_effect = [0.6, -0.4]
    monkeypatch.setattr(
        "providers.sentiment.get_sentiment", lambda *a, **k: provider,
    )
    sentiment = sentiment_alpha_scores(["AAPL", "MSFT"])
    lgbm = {"AAPL": 0.3, "MSFT": 0.1}
    ridge = {"AAPL": 0.5, "MSFT": -0.2}
    blended = blend_signals(lgbm, ridge, sentiment)
    assert set(blended.keys()) == {"AAPL", "MSFT"}
    for v in blended.values():
        assert -1.0 <= v <= 1.0
