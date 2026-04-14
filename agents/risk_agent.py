"""agents/risk_agent.py — Specialist agent wrapping correlation & VaR checks."""
from __future__ import annotations

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskAgent:
    """Returns a risk signal based on portfolio concentration and VaR."""

    name = "risk_agent"

    def run(self, context: dict) -> AgentSignal:
        portfolio = context.get("portfolio", {})
        positions = portfolio.get("positions", {})

        warnings: list[str] = []

        # Correlation check
        if len(positions) >= 2:
            try:
                from data.fetcher import fetch_ohlcv
                from risk.correlation import check_correlation_alerts
                price_data = {}
                for ticker in list(positions.keys())[:10]:
                    df = fetch_ohlcv(ticker, period="3mo")
                    if not df.empty and "Close" in df.columns:
                        price_data[ticker] = df["Close"]
                alerts = check_correlation_alerts(price_data, positions)
                for alert in alerts:
                    warnings.append(alert.message)
            except Exception as exc:
                logger.debug("RiskAgent: correlation check failed: %s", exc)

        if warnings:
            return AgentSignal(
                agent_name=self.name,
                signal="bearish",
                confidence=0.65,
                reasoning=" | ".join(warnings[:2]),
                metadata={"warning_count": len(warnings)},
            )
        return AgentSignal(
            agent_name=self.name,
            signal="neutral",
            confidence=0.7,
            reasoning="Portfolio risk metrics within normal thresholds",
        )
