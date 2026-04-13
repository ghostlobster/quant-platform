from __future__ import annotations

import os


class AlpacaMarketDataAdapter:
    def __init__(self) -> None:
        try:
            import requests

            self._requests = requests
        except ImportError as e:
            raise ImportError("requests not installed. Run: pip install requests") from e
        self._api_key = os.environ.get("ALPACA_API_KEY", "")
        self._secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        self._base_url = "https://data.alpaca.markets"

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
        }

    def get_bars(self, symbol: str, timeframe: str, start: str, end: str) -> list[dict]:
        params = {"timeframe": timeframe, "start": start, "end": end, "limit": 1000}
        r = self._requests.get(
            f"{self._base_url}/v2/stocks/{symbol}/bars",
            headers=self._headers(),
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("bars", [])

    def get_quote(self, symbol: str) -> dict:
        r = self._requests.get(
            f"{self._base_url}/v2/stocks/{symbol}/quotes/latest",
            headers=self._headers(),
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("quote", {})

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        params = {"symbols": ",".join(symbols)}
        r = self._requests.get(
            f"{self._base_url}/v2/stocks/quotes/latest",
            headers=self._headers(),
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        return r.json().get("quotes", {})
