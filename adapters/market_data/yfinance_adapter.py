"""
adapters/market_data/yfinance_adapter.py — Wraps data/fetcher.py (yfinance).

Requires:  pip install yfinance  (already in requirements.txt)
"""
from __future__ import annotations

import logging

try:
    import yfinance as _yf
except ImportError:
    _yf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class YFinanceAdapter:
    """MarketDataProvider backed by Yahoo Finance via yfinance."""

    def __init__(self) -> None:
        if _yf is None:
            raise ImportError(
                "yfinance package is required for YFinanceAdapter. "
                "Install it with: pip install yfinance"
            )

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """
        Return OHLCV bars.

        *timeframe* is mapped to yfinance ``interval``:
            1Day → 1d, 1Hour → 1h, 5Min → 5m, 1Min → 1m
        Unrecognised values are passed through as-is.
        """
        interval_map = {
            "1day": "1d", "1d": "1d",
            "1hour": "1h", "1h": "1h",
            "30min": "30m", "15min": "15m",
            "5min": "5m", "1min": "1m",
        }
        interval = interval_map.get(timeframe.lower(), timeframe.lower())
        ticker = _yf.Ticker(symbol.upper())
        df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
        if df.empty:
            return []
        records = []
        for ts, row in df.iterrows():
            records.append({
                "t": ts.isoformat(),
                "o": float(row["Open"]),
                "h": float(row["High"]),
                "l": float(row["Low"]),
                "c": float(row["Close"]),
                "v": int(row["Volume"]),
            })
        return records

    def get_quote(self, symbol: str) -> dict:
        ticker = _yf.Ticker(symbol.upper())
        info = ticker.fast_info
        try:
            return {
                "symbol": symbol.upper(),
                "bid": getattr(info, "bid", None),
                "ask": getattr(info, "ask", None),
                "last_price": getattr(info, "last_price", None),
                "previous_close": getattr(info, "previous_close", None),
            }
        except Exception as exc:
            logger.warning("YFinance get_quote failed for %s: %s", symbol, exc)
            return {"symbol": symbol.upper()}

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {sym: self.get_quote(sym) for sym in symbols}
