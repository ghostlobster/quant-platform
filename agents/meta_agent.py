"""
agents/meta_agent.py — Meta-agent that aggregates specialist signals.

Collects outputs from all specialist agents, applies configurable weights, and
returns a consensus signal. Optionally calls the LLM to arbitrate when agents
disagree significantly.

ENV vars
--------
    AGENT_WEIGHTS       JSON dict of agent weights, e.g. '{"regime_agent":2.0}'
    AGENT_LLM_ARBITER   set to '1' to enable LLM meta-arbitration (default: 0)
"""
from __future__ import annotations

import json
import os
import re

from agents.base import AgentProvider, AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)

# Valid ticker: 1-6 uppercase letters/digits, optionally with a dot suffix (e.g. BRK.B)
_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$")

_DEFAULT_WEIGHTS: dict[str, float] = {
    "regime_agent": 1.5,
    "risk_agent": 1.2,
    "sentiment_agent": 0.8,
    "screener_agent": 1.0,
    "execution_agent": 0.5,
}

_SIGNAL_SCORES = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


def _load_weights() -> dict[str, float]:
    raw = os.environ.get("AGENT_WEIGHTS", "")
    if raw:
        try:
            return {**_DEFAULT_WEIGHTS, **json.loads(raw)}
        except Exception:
            pass
    return dict(_DEFAULT_WEIGHTS)


class MetaAgent:
    """
    Aggregates specialist agent signals using weighted voting.

    Parameters
    ----------
    agents : list of AgentProvider instances to consult.
             Defaults to all five specialists when not provided.
    weights : optional weight override dict {agent_name: weight}
    """

    def __init__(
        self,
        agents: list[AgentProvider] | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        if agents is None:
            from agents.execution_agent import ExecutionAgent
            from agents.regime_agent import RegimeAgent
            from agents.risk_agent import RiskAgent
            from agents.screener_agent import ScreenerAgent
            from agents.sentiment_agent import SentimentAgent
            agents = [RegimeAgent(), RiskAgent(), SentimentAgent(), ScreenerAgent(), ExecutionAgent()]
        self._agents: list[AgentProvider] = agents
        self._weights = weights or _load_weights()

    def run(self, context: dict) -> dict:
        """
        Run all specialist agents and return a consensus signal dict.

        Returns
        -------
        dict with keys:
            signal       : 'bullish' | 'neutral' | 'bearish'
            confidence   : float [0.0, 1.0]
            weighted_score : float [-1.0, 1.0] (positive = bullish)
            specialist_signals : list of individual AgentSignal dicts
            reasoning    : str summary
        """
        signals: list[AgentSignal] = []
        for agent in self._agents:
            try:
                sig = agent.run(context)
                signals.append(sig)
                logger.debug("Agent %s: %s (%.2f)", sig.agent_name, sig.signal, sig.confidence)
            except Exception as exc:
                logger.warning("Agent %s failed: %s", getattr(agent, "name", "?"), exc)

        if not signals:
            return {
                "signal": "neutral",
                "confidence": 0.0,
                "weighted_score": 0.0,
                "specialist_signals": [],
                "reasoning": "No agents returned signals",
            }

        # Weighted vote
        total_weight = 0.0
        weighted_sum = 0.0
        for sig in signals:
            w = self._weights.get(sig.agent_name, 1.0)
            score = _SIGNAL_SCORES.get(sig.signal, 0.0)
            # Weight by both agent weight and signal confidence
            effective_weight = w * sig.confidence
            weighted_sum += score * effective_weight
            total_weight += effective_weight

        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        if weighted_score > 0.2:
            consensus = "bullish"
        elif weighted_score < -0.2:
            consensus = "bearish"
        else:
            consensus = "neutral"

        confidence = min(1.0, abs(weighted_score))
        reasoning = f"Weighted score={weighted_score:+.3f} → {consensus}"

        # Optional LLM arbitration for close calls
        if os.environ.get("AGENT_LLM_ARBITER", "0") == "1" and abs(weighted_score) < 0.3:
            try:
                reasoning = self._llm_arbitrate(signals, context, weighted_score, reasoning)
            except Exception as exc:
                logger.warning("LLM arbiter failed: %s", exc)
                reasoning = reasoning + " [LLM arbiter unavailable]"

        return {
            "signal": consensus,
            "confidence": round(confidence, 4),
            "weighted_score": round(weighted_score, 4),
            "specialist_signals": [
                {
                    "agent": s.agent_name,
                    "signal": s.signal,
                    "confidence": s.confidence,
                    "reasoning": s.reasoning,
                }
                for s in signals
            ],
            "reasoning": reasoning,
        }

    def _llm_arbitrate(
        self,
        signals: list[AgentSignal],
        context: dict,
        weighted_score: float,
        fallback_reasoning: str,
    ) -> str:
        """Ask the LLM to arbitrate when the vote is close."""
        from providers.llm import get_llm
        summary = "\n".join(
            f"- {s.agent_name}: {s.signal} ({s.confidence:.2f}) — {s.reasoning}"
            for s in signals
        )
        raw_ticker = str(context.get("ticker", "")).upper().strip()
        # Validate ticker to prevent prompt injection
        ticker = raw_ticker if _TICKER_RE.match(raw_ticker) else "UNKNOWN"
        prompt = (
            f"You are a quant portfolio manager. The following specialist agents have "
            f"analysed {ticker} and disagree (weighted score: {weighted_score:+.3f}):\n\n"
            f"{summary}\n\n"
            "Provide a one-sentence synthesis of their views and your recommendation. "
            "Be direct and concise."
        )
        llm = get_llm()
        return llm.complete(prompt)
