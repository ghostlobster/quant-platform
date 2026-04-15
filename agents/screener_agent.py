"""agents/screener_agent.py — Specialist agent wrapping the equity screener."""
from __future__ import annotations

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class ScreenerAgent:
    """Returns a signal based on the screener's signal for the target ticker."""

    name = "screener_agent"

    def run(self, context: dict) -> AgentSignal:
        ticker = context.get("ticker", "")
        if not ticker:
            return AgentSignal(
                agent_name=self.name, signal="neutral", confidence=0.3,
                reasoning="No ticker in context",
            )
        try:
            from screener.screener import run_screener
            results = run_screener(tickers=[ticker])
            if results.empty:
                raise ValueError("No screener results")
            row = results.iloc[0]
            signal_label = str(row.get("Signal", "Neutral")).lower()
            if "trending up" in signal_label or "overbought" in signal_label:
                signal, conf = "bullish", 0.7
            elif "trending down" in signal_label or "oversold" in signal_label:
                signal, conf = "bearish", 0.7
            else:
                signal, conf = "neutral", 0.5
            return AgentSignal(
                agent_name=self.name,
                signal=signal,
                confidence=conf,
                reasoning=f"Screener signal for {ticker}: {signal_label}",
                metadata={"screener_signal": signal_label},
            )
        except Exception as exc:
            logger.debug("ScreenerAgent: screener failed for %s: %s", ticker, exc)
            return AgentSignal(
                agent_name=self.name, signal="neutral", confidence=0.3,
                reasoning="Screener data unavailable",
            )
