"""Tests for strategies/gnn_signal.py (Issue #36)."""
from __future__ import annotations

import numpy as np
import pytest


def test_build_sector_adjacency_same_sector():
    from strategies.gnn_signal import build_sector_adjacency

    tickers = ["AAPL", "MSFT", "NVDA"]  # All Technology
    adj = build_sector_adjacency(tickers)
    assert adj.shape == (3, 3)
    # Same sector → all off-diagonal entries should be 1
    assert adj[0, 1] == pytest.approx(1.0)
    assert adj[1, 2] == pytest.approx(1.0)
    # Diagonal should be 0
    assert adj[0, 0] == pytest.approx(0.0)


def test_build_sector_adjacency_different_sectors():
    from strategies.gnn_signal import build_sector_adjacency

    tickers = ["AAPL", "JPM"]  # Technology vs Financials
    adj = build_sector_adjacency(tickers)
    assert adj.shape == (2, 2)
    # Different sectors → all off-diagonal entries should be 0
    assert adj[0, 1] == pytest.approx(0.0)
    assert adj[1, 0] == pytest.approx(0.0)


def test_build_sector_adjacency_unknown_ticker():
    from strategies.gnn_signal import build_sector_adjacency

    tickers = ["AAPL", "UNKNOWN_XYZ"]
    adj = build_sector_adjacency(tickers)
    assert adj.shape == (2, 2)
    # Unknown sector should not connect to any other
    assert adj[0, 1] == pytest.approx(0.0)


def test_build_node_features_shape():
    from strategies.gnn_signal import build_node_features

    tickers = ["AAPL", "MSFT", "JPM"]
    features = build_node_features(tickers)
    assert features.shape == (3, 8)
    assert features.dtype == np.float32


def test_build_node_features_regime_encoding():
    from strategies.gnn_signal import build_node_features

    tickers = ["AAPL"]
    features_bull = build_node_features(tickers, regime="trending_bull")
    features_bear = build_node_features(tickers, regime="trending_bear")

    # Regime one-hot features are at positions [3:7]
    # trending_bull is index 0 in REGIME_STATES
    assert features_bull[0, 3] == pytest.approx(1.0)   # trending_bull = 1
    assert features_bear[0, 3] == pytest.approx(0.0)   # trending_bull = 0 for bear


def test_build_node_features_rsi_normalized():
    from strategies.gnn_signal import build_node_features

    tickers = ["AAPL"]
    indicators = {"AAPL": {"rsi": 70.0, "momentum": 0.05, "sma_signal": 1.0}}
    features = build_node_features(tickers, indicators=indicators)
    # RSI should be divided by 100
    assert features[0, 0] == pytest.approx(0.70)


def test_build_node_features_sentiment():
    from strategies.gnn_signal import build_node_features

    tickers = ["AAPL"]
    sentiments = {"AAPL": 0.42}
    features = build_node_features(tickers, sentiments=sentiments)
    # Sentiment is the last feature (index 7)
    assert features[0, 7] == pytest.approx(0.42)


def test_gnn_signal_fallback_score_without_model():
    from strategies.gnn_signal import GNNSignal

    # No checkpoint exists → should use fallback scorer
    gnn = GNNSignal(model_path="/nonexistent/model.pt")
    assert gnn._model is None

    tickers = ["AAPL", "MSFT"]
    scores = gnn.score(tickers)
    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"AAPL", "MSFT"}
    for ticker, score in scores.items():
        assert -1.0 <= score <= 1.0, f"Score out of range for {ticker}: {score}"


def test_gnn_signal_fallback_score_empty_tickers():
    from strategies.gnn_signal import GNNSignal

    gnn = GNNSignal(model_path="/no/model.pt")
    scores = gnn.score([])
    assert scores == {}


def test_gnn_signal_fallback_handles_all_known_tickers():
    from strategies.gnn_signal import TICKER_SECTOR, GNNSignal

    gnn = GNNSignal(model_path="/no/model.pt")
    tickers = list(TICKER_SECTOR.keys())
    scores = gnn.score(tickers)
    assert len(scores) == len(tickers)
    for score in scores.values():
        assert -1.0 <= score <= 1.0
