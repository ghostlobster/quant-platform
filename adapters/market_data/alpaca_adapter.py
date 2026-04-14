"""
adapters/market_data/alpaca_adapter.py — Alpaca Markets data API.

Uses the Alpaca Data API v2 (no extra SDK needed — pure requests).

ENV vars
--------
    ALPACA_API_KEY, ALPACA_SECRET_KEY
    ALPACA_DATA_BASE_URL  (default: https://data.alpaca.markets)
"""
from __future__ import annotations

import logging
import os

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_DATA_URL = "https://data.alpaca.markets"


class AlpacaMarketDataAdapter:
    """MarketDataProvider backed by Alpaca Data API v2."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if _requests is None:
            raise ImportError(
                "requests package is required for AlpacaMarketDataAdapter. "
                "Install it with: pip install requests"
            )
        self._api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = (
            base_url
            or os.environ.get("ALPACA_DATA_BASE_URL", _DEFAULT_DATA_URL)
        ).rstrip("/")

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._base_url}{path}"
        resp = _requests.get(url, headers=self._headers(), params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """Return OHLCV bars. timeframe examples: ``1Day``, ``1Hour``, ``5Min``."""
        data = self._get(
            f"/v2/stocks/{symbol.upper()}/bars",
            params={
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "adjustment": "raw",
                "feed": "iex",
            },
        )
        return data.get("bars", [])

    def get_quote(self, symbol: str) -> dict:
        data = self._get(f"/v2/stocks/{symbol.upper()}/quotes/latest")
        return data.get("quote", {})

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        joined = ",".join(s.upper() for s in symbols)
        data = self._get("/v2/stocks/quotes/latest", params={"symbols": joined})
        return data.get("quotes", {})
