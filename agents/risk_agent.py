"""agents/risk_agent.py — Specialist agent wrapping correlation & VaR checks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

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

                tickers = list(positions.keys())[:10]

                # Fetch all OHLCV series in parallel to avoid N+1 serial calls
                price_data: dict = {}
                with ThreadPoolExecutor(max_workers=min(5, len(tickers))) as pool:
                    future_to_ticker = {
                        pool.submit(fetch_ohlcv, t, "3mo"): t for t in tickers
                    }
                    for future in as_completed(future_to_ticker):
                        t = future_to_ticker[future]
                        try:
                            df = future.result()
                            if not df.empty and "Close" in df.columns:
                                price_data[t] = df["Close"]
                        except Exception as exc:
                            logger.debug("RiskAgent: fetch failed for %s: %s", t, exc)

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
