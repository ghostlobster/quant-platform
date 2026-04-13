from __future__ import annotations


class YFinanceAdapter:
    def __init__(self) -> None:
        try:
            import yfinance

            self._yf = yfinance
        except ImportError as e:
            raise ImportError("yfinance not installed. Run: pip install yfinance") from e

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]:
        ticker = self._yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=timeframe)
        return df.reset_index().rename(columns=str.lower).to_dict(orient="records")

    def get_quote(self, symbol: str) -> dict:
        ticker = self._yf.Ticker(symbol)
        info = ticker.fast_info
        return {
            "symbol": symbol,
            "last": getattr(info, "last_price", None),
            "bid": getattr(info, "last_price", None),
            "ask": getattr(info, "last_price", None),
        }

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {s: self.get_quote(s) for s in symbols}
