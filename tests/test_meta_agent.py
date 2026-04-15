"""Tests for agents/meta_agent.py (Issue #38)."""
from __future__ import annotations

from agents.base import AgentSignal


def _make_mock_agent(name: str, signal: str, confidence: float):
    """Return a simple mock agent object."""
    from unittest.mock import MagicMock

    agent = MagicMock()
    agent.name = name
    agent.run.return_value = AgentSignal(
        agent_name=name,
        signal=signal,
        confidence=confidence,
        reasoning=f"mock {name}",
    )
    return agent


def test_meta_agent_returns_bullish_when_all_bullish():
    from agents.meta_agent import MetaAgent

    agents = [
        _make_mock_agent("regime_agent", "bullish", 0.8),
        _make_mock_agent("risk_agent", "bullish", 0.9),
        _make_mock_agent("sentiment_agent", "bullish", 0.7),
    ]
    meta = MetaAgent(agents=agents)
    result = meta.run({"ticker": "AAPL"})
    assert result["signal"] == "bullish"
    assert result["weighted_score"] > 0


def test_meta_agent_returns_bearish_when_all_bearish():
    from agents.meta_agent import MetaAgent

    agents = [
        _make_mock_agent("regime_agent", "bearish", 0.8),
        _make_mock_agent("risk_agent", "bearish", 0.85),
        _make_mock_agent("sentiment_agent", "bearish", 0.75),
    ]
    meta = MetaAgent(agents=agents)
    result = meta.run({"ticker": "SPY"})
    assert result["signal"] == "bearish"
    assert result["weighted_score"] < 0


def test_meta_agent_returns_neutral_for_mixed_signals():
    from agents.meta_agent import MetaAgent

    agents = [
        _make_mock_agent("regime_agent", "bullish", 0.6),
        _make_mock_agent("risk_agent", "bearish", 0.6),
    ]
    meta = MetaAgent(agents=agents)
    result = meta.run({"ticker": "MSFT"})
    # Equally opposing signals should produce neutral
    assert result["signal"] == "neutral"


def test_meta_agent_specialist_signals_in_result():
    from agents.meta_agent import MetaAgent

    agents = [
        _make_mock_agent("regime_agent", "bullish", 0.8),
        _make_mock_agent("sentiment_agent", "neutral", 0.5),
    ]
    meta = MetaAgent(agents=agents)
    result = meta.run({"ticker": "NVDA"})
    assert len(result["specialist_signals"]) == 2
    agent_names = {s["agent"] for s in result["specialist_signals"]}
    assert "regime_agent" in agent_names
    assert "sentiment_agent" in agent_names


def test_meta_agent_no_agents_returns_neutral():
    from agents.meta_agent import MetaAgent

    meta = MetaAgent(agents=[])
    result = meta.run({})
    assert result["signal"] == "neutral"
    assert result["confidence"] == 0.0
    assert result["specialist_signals"] == []


def test_meta_agent_custom_weights_affect_score():
    from agents.meta_agent import MetaAgent

    agents = [
        _make_mock_agent("regime_agent", "bullish", 1.0),
        _make_mock_agent("risk_agent", "bearish", 1.0),
    ]
    # Give regime_agent much higher weight → should tip bullish
    meta = MetaAgent(agents=agents, weights={"regime_agent": 10.0, "risk_agent": 1.0})
    result = meta.run({"ticker": "AAPL"})
    assert result["signal"] == "bullish"


def test_meta_agent_confidence_in_valid_range():
    from agents.meta_agent import MetaAgent

    agents = [_make_mock_agent("regime_agent", "bullish", 0.9)]
    meta = MetaAgent(agents=agents)
    result = meta.run({})
    assert 0.0 <= result["confidence"] <= 1.0


def test_meta_agent_failing_agent_is_skipped():
    from unittest.mock import MagicMock

    from agents.meta_agent import MetaAgent

    bad_agent = MagicMock()
    bad_agent.name = "failing_agent"
    bad_agent.run.side_effect = RuntimeError("boom")

    good_agent = _make_mock_agent("regime_agent", "bullish", 0.8)

    meta = MetaAgent(agents=[bad_agent, good_agent])
    result = meta.run({})
    # Should still return a result from the good agent
    assert result["signal"] in ("bullish", "bearish", "neutral")
    # Only the successful agent's signal should be in the list
    assert len(result["specialist_signals"]) == 1


def test_meta_agent_weighted_score_and_reasoning_present():
    from agents.meta_agent import MetaAgent

    agents = [_make_mock_agent("regime_agent", "bullish", 0.7)]
    meta = MetaAgent(agents=agents)
    result = meta.run({"ticker": "AAPL"})
    assert "weighted_score" in result
    assert "reasoning" in result
    assert isinstance(result["reasoning"], str)
    assert len(result["reasoning"]) > 0
