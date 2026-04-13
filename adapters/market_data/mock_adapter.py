class MockMarketDataAdapter:
    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]:
        return [
            {"t": start, "o": 100.0, "h": 105.0, "l": 99.0, "c": 102.0, "v": 10000},
        ]

    def get_quote(self, symbol: str) -> dict:
        return {"symbol": symbol, "bid": 99.5, "ask": 100.5, "last": 100.0}

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {s: self.get_quote(s) for s in symbols}
