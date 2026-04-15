"""agents/execution_agent.py — Specialist agent recommending an execution algorithm."""
from __future__ import annotations

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionAgent:
    """Recommends a market | twap | vwap execution algo based on context."""

    name = "execution_agent"

    def run(self, context: dict) -> AgentSignal:
        regime = context.get("regime", "trending_bull")
        order_size = float(context.get("order_size", 0))

        # Large orders in volatile regimes → TWAP; normal → market
        if regime == "high_vol" or order_size > 10_000:
            algo = "twap"
            reasoning = f"High vol or large order (${order_size:,.0f}) → TWAP for lower impact"
        elif order_size > 5_000:
            algo = "vwap"
            reasoning = f"Moderate order (${order_size:,.0f}) → VWAP for volume-weighted fill"
        else:
            algo = "market"
            reasoning = "Small order → market order for immediate fill"

        return AgentSignal(
            agent_name=self.name,
            signal="neutral",       # execution agent doesn't have direction bias
            confidence=0.85,
            reasoning=reasoning,
            metadata={"recommended_algo": algo},
        )
