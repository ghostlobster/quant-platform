"""Tests for specialist agents: regime, risk, screener, sentiment, execution (Issue #38)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agents.base import AgentSignal

# Ensure yfinance is available as a mock so data.fetcher can be imported locally
sys.modules.setdefault("yfinance", MagicMock())


# ── RegimeAgent ────────────────────────────────────────────────────────────────

def test_regime_agent_bull_signal():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    result = agent.run({"regime": "trending_bull"})
    assert isinstance(result, AgentSignal)
    assert result.signal == "bullish"
    assert result.confidence == pytest.approx(0.8)
    assert result.agent_name == "regime_agent"


def test_regime_agent_bear_signal():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    result = agent.run({"regime": "trending_bear"})
    assert result.signal == "bearish"


def test_regime_agent_high_vol_bearish():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    result = agent.run({"regime": "high_vol"})
    assert result.signal == "bearish"


def test_regime_agent_mean_reverting_neutral():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    result = agent.run({"regime": "mean_reverting"})
    assert result.signal == "neutral"


def test_regime_agent_unknown_regime_neutral():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    result = agent.run({"regime": "unknown_regime_xyz"})
    assert result.signal == "neutral"


def test_regime_agent_fetches_regime_when_absent():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    with patch("analysis.regime.get_live_regime_with_llm",
               return_value={"regime": "trending_bull"}):
        result = agent.run({})
    assert result.signal == "bullish"


def test_regime_agent_fallback_on_fetch_error():
    from agents.regime_agent import RegimeAgent
    agent = RegimeAgent()
    with patch("analysis.regime.get_live_regime_with_llm",
               side_effect=RuntimeError("no data")):
        result = agent.run({})
    assert result.signal == "neutral"
    assert result.confidence <= 0.5


# ── RiskAgent ──────────────────────────────────────────────────────────────────

def test_risk_agent_neutral_on_empty_portfolio():
    from agents.risk_agent import RiskAgent
    agent = RiskAgent()
    result = agent.run({"portfolio": {"positions": {}}})
    assert result.signal == "neutral"
    assert result.agent_name == "risk_agent"


def test_risk_agent_neutral_when_no_portfolio_key():
    from agents.risk_agent import RiskAgent
    agent = RiskAgent()
    result = agent.run({})
    assert result.signal == "neutral"


def test_risk_agent_bearish_when_correlation_alert():
    from agents.risk_agent import RiskAgent
    from risk.correlation import CorrelationAlert

    mock_alert = CorrelationAlert(
        alert_type="position_concentration",
        value=0.85,
        threshold=0.25,
        message="AAPL is 85% of portfolio",
        ticker="AAPL",
    )
    agent = RiskAgent()
    positions = {"AAPL": 85_000.0, "MSFT": 15_000.0}
    mock_df = pd.DataFrame({"Close": [100.0] * 30})

    with patch("risk.correlation.check_correlation_alerts", return_value=[mock_alert]), \
         patch("data.fetcher.fetch_ohlcv", return_value=mock_df):
        result = agent.run({"portfolio": {"positions": positions}})
    assert result.signal == "bearish"


# ── ScreenerAgent ──────────────────────────────────────────────────────────────

def test_screener_agent_no_ticker_returns_neutral():
    from agents.screener_agent import ScreenerAgent
    agent = ScreenerAgent()
    result = agent.run({})
    assert result.signal == "neutral"
    assert result.agent_name == "screener_agent"


def test_screener_agent_bullish_on_trending_up():
    from agents.screener_agent import ScreenerAgent

    mock_run_screener = MagicMock(
        return_value=pd.DataFrame([{"Signal": "Trending Up", "Ticker": "AAPL"}])
    )
    mock_screener_mod = MagicMock()
    mock_screener_mod.run_screener = mock_run_screener
    with patch.dict(sys.modules, {"screener.screener": mock_screener_mod}):
        agent = ScreenerAgent()
        result = agent.run({"ticker": "AAPL"})
    assert result.signal == "bullish"


def test_screener_agent_bearish_on_trending_down():
    from agents.screener_agent import ScreenerAgent

    mock_screener_mod = MagicMock()
    mock_screener_mod.run_screener.return_value = pd.DataFrame(
        [{"Signal": "Trending Down", "Ticker": "SPY"}]
    )
    with patch.dict(sys.modules, {"screener.screener": mock_screener_mod}):
        agent = ScreenerAgent()
        result = agent.run({"ticker": "SPY"})
    assert result.signal == "bearish"


def test_screener_agent_fallback_on_error():
    from agents.screener_agent import ScreenerAgent

    mock_screener_mod = MagicMock()
    mock_screener_mod.run_screener.side_effect = RuntimeError("db error")
    with patch.dict(sys.modules, {"screener.screener": mock_screener_mod}):
        agent = ScreenerAgent()
        result = agent.run({"ticker": "AAPL"})
    assert result.signal == "neutral"


# ── SentimentAgent ─────────────────────────────────────────────────────────────

def test_sentiment_agent_bullish_positive_score():
    from agents.sentiment_agent import SentimentAgent

    mock_provider = MagicMock()
    mock_provider.ticker_sentiment.return_value = 0.6

    with patch("providers.sentiment.get_sentiment", return_value=mock_provider):
        agent = SentimentAgent()
        result = agent.run({"ticker": "AAPL"})
    assert result.signal == "bullish"
    assert result.agent_name == "sentiment_agent"
    assert result.confidence > 0.5


def test_sentiment_agent_bearish_negative_score():
    from agents.sentiment_agent import SentimentAgent

    mock_provider = MagicMock()
    mock_provider.ticker_sentiment.return_value = -0.5

    with patch("providers.sentiment.get_sentiment", return_value=mock_provider):
        agent = SentimentAgent()
        result = agent.run({"ticker": "MSFT"})
    assert result.signal == "bearish"


def test_sentiment_agent_neutral_near_zero():
    from agents.sentiment_agent import SentimentAgent

    mock_provider = MagicMock()
    mock_provider.ticker_sentiment.return_value = 0.05

    with patch("providers.sentiment.get_sentiment", return_value=mock_provider):
        agent = SentimentAgent()
        result = agent.run({"ticker": "SPY"})
    assert result.signal == "neutral"


def test_sentiment_agent_fallback_on_error():
    from agents.sentiment_agent import SentimentAgent
    with patch("providers.sentiment.get_sentiment", side_effect=RuntimeError("api down")):
        agent = SentimentAgent()
        result = agent.run({"ticker": "TSLA"})
    assert result.signal == "neutral"
    assert result.confidence <= 0.5


# ── ExecutionAgent ─────────────────────────────────────────────────────────────

def test_execution_agent_market_for_small_order():
    from agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    result = agent.run({"regime": "trending_bull", "order_size": 1000})
    assert result.signal == "neutral"
    assert result.metadata["recommended_algo"] == "market"
    assert result.agent_name == "execution_agent"


def test_execution_agent_twap_for_high_vol():
    from agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    result = agent.run({"regime": "high_vol", "order_size": 1000})
    assert result.metadata["recommended_algo"] == "twap"


def test_execution_agent_twap_for_large_order():
    from agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    result = agent.run({"regime": "trending_bull", "order_size": 15_000})
    assert result.metadata["recommended_algo"] == "twap"


def test_execution_agent_vwap_for_moderate_order():
    from agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    result = agent.run({"regime": "mean_reverting", "order_size": 7_000})
    assert result.metadata["recommended_algo"] == "vwap"


def test_execution_agent_high_confidence():
    from agents.execution_agent import ExecutionAgent
    agent = ExecutionAgent()
    result = agent.run({})
    assert result.confidence >= 0.8
