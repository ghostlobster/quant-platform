"""
adapters/market_data/mock_adapter.py — Deterministic mock for tests.

No external SDK required.  Returns synthetic but structurally valid data.
"""
from __future__ import annotations

from datetime import date, timedelta


class MockMarketDataAdapter:
    """MarketDataProvider that returns canned synthetic data."""

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        start_d = date.fromisoformat(start[:10])
        end_d = date.fromisoformat(end[:10])
        records = []
        price = 100.0
        d = start_d
        while d <= end_d:
            records.append({
                "t": d.isoformat(),
                "o": round(price, 2),
                "h": round(price * 1.01, 2),
                "l": round(price * 0.99, 2),
                "c": round(price * 1.005, 2),
                "v": 1_000_000,
            })
            price *= 1.001
            d += timedelta(days=1)
        return records

    def get_quote(self, symbol: str) -> dict:
        return {
            "symbol": symbol.upper(),
            "bid": 99.99,
            "ask": 100.01,
            "last_price": 100.0,
            "previous_close": 99.5,
        }

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {sym: self.get_quote(sym) for sym in symbols}
