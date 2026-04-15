"""agents/regime_agent.py — Specialist agent wrapping the regime classifier."""
from __future__ import annotations

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class RegimeAgent:
    """Returns a directional signal based on the current market regime."""

    name = "regime_agent"

    def run(self, context: dict) -> AgentSignal:
        regime = context.get("regime")
        if regime is None:
            try:
                from analysis.regime import get_cached_live_regime
                data = get_cached_live_regime(use_llm=True)
                regime = data["regime"]
            except Exception as exc:
                logger.warning("RegimeAgent: regime fetch failed: %s", exc)
                return AgentSignal(
                    agent_name=self.name,
                    signal="neutral",
                    confidence=0.3,
                    reasoning="Regime data unavailable",
                )

        signal_map = {
            "trending_bull": ("bullish", 0.8),
            "trending_bear": ("bearish", 0.8),
            "mean_reverting": ("neutral", 0.5),
            "high_vol": ("bearish", 0.6),
        }
        signal, confidence = signal_map.get(regime, ("neutral", 0.4))
        return AgentSignal(
            agent_name=self.name,
            signal=signal,
            confidence=confidence,
            reasoning=f"Market regime is '{regime}'",
            metadata={"regime": regime},
        )
