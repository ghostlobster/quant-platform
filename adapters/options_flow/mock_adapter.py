"""
adapters/options_flow/mock_adapter.py — Synthetic options flow adapter for tests.
"""
from __future__ import annotations

import random

from providers.options_flow import OptionsFlowResult


class MockOptionsFlowAdapter:
    """Returns deterministic synthetic options flow data. No network calls."""

    def get_flow(self, symbol: str, lookback_days: int = 1) -> list[dict]:
        rng = random.Random(hash(symbol) % 2**32)
        records = []
        for _ in range(10):
            side = rng.choice(["call", "put"])
            records.append({
                "symbol": symbol.upper(),
                "side": side,
                "volume": rng.randint(100, 5000),
                "strike": round(rng.uniform(50, 500), 2),
                "expiry": "2026-06-20",
                "premium": round(rng.uniform(1, 20), 2),
            })
        return records

    def unusual_activity_score(self, symbol: str) -> OptionsFlowResult:
        flow = self.get_flow(symbol)
        call_vol = sum(r["volume"] for r in flow if r["side"] == "call")
        put_vol = sum(r["volume"] for r in flow if r["side"] == "put")
        total = call_vol + put_vol or 1
        ratio = call_vol / put_vol if put_vol > 0 else 2.0
        baseline = 1.0
        unusual = ratio > 1.5 or ratio < 0.67
        score = (ratio - baseline) / (baseline + 1e-6)
        score = max(-1.0, min(1.0, score))
        return OptionsFlowResult(
            symbol=symbol.upper(),
            call_volume=call_vol,
            put_volume=put_vol,
            call_put_ratio=round(ratio, 4),
            unusual_score=round(score, 4),
            avg_20d_call_put_ratio=baseline,
            is_unusual=unusual,
        )
