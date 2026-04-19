"""
adapters/market_data/polygon_adapter.py — Polygon.io market data.

Implements the ``MarketDataProvider`` Protocol over Polygon's REST API v2.
No vendor SDK required — uses ``requests`` directly.

Timeframe translation — accepts Alpaca-style strings (``1Day``,
``1Hour``, ``5Min``) as well as Polygon-native ``{multiplier}/{timespan}``
syntax. See :data:`_TIMEFRAME_MAP`.

ENV vars
--------
    POLYGON_API_KEY          required — obtain at https://polygon.io
    POLYGON_BASE_URL         default: https://api.polygon.io
    POLYGON_ADJUSTMENT       ``true`` (split + dividend-adjusted) | ``false``
"""
from __future__ import annotations

import logging
import os
import re

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.polygon.io"

# Alpaca-style timeframe string → (multiplier, timespan) on Polygon.
_TIMEFRAME_MAP: dict[str, tuple[int, str]] = {
    "1Min": (1, "minute"),
    "5Min": (5, "minute"),
    "15Min": (15, "minute"),
    "30Min": (30, "minute"),
    "1Hour": (1, "hour"),
    "4Hour": (4, "hour"),
    "1Day": (1, "day"),
    "1Week": (1, "week"),
    "1Month": (1, "month"),
}

_POLY_NATIVE_PATTERN = re.compile(r"^(\d+)/([a-zA-Z]+)$")


def _parse_timeframe(tf: str) -> tuple[int, str]:
    tf_key = tf.strip()
    if tf_key in _TIMEFRAME_MAP:
        return _TIMEFRAME_MAP[tf_key]
    match = _POLY_NATIVE_PATTERN.match(tf_key)
    if match:
        return int(match.group(1)), match.group(2).lower()
    raise ValueError(
        f"Unknown timeframe {tf!r}. "
        f"Expected one of {sorted(_TIMEFRAME_MAP)} or '{{multiplier}}/{{timespan}}'."
    )


class PolygonAdapter:
    """MarketDataProvider backed by Polygon.io REST v2."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        adjusted: bool | None = None,
    ) -> None:
        if _requests is None:
            raise ImportError(
                "requests package is required for PolygonAdapter. "
                "Install it with: pip install requests"
            )
        self._api_key = api_key or os.environ.get("POLYGON_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("POLYGON_BASE_URL", _DEFAULT_BASE_URL)
        ).rstrip("/")
        if adjusted is None:
            raw = os.environ.get("POLYGON_ADJUSTMENT", "true").lower().strip()
            self._adjusted = raw not in ("0", "false", "no", "off")
        else:
            self._adjusted = bool(adjusted)

    def _params(self, extra: dict | None = None) -> dict:
        out = {"apiKey": self._api_key}
        if extra:
            out.update(extra)
        return out

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{self._base_url}{path}"
        resp = _requests.get(url, params=self._params(params), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """Return OHLCV bars with Alpaca-compatible keys (t, o, h, l, c, v)."""
        multiplier, timespan = _parse_timeframe(timeframe)
        path = (
            f"/v2/aggs/ticker/{symbol.upper()}/range/"
            f"{multiplier}/{timespan}/{start}/{end}"
        )
        data = self._get(
            path,
            params={
                "adjusted": "true" if self._adjusted else "false",
                "sort": "asc",
                "limit": 50000,
            },
        )
        results = data.get("results") or []
        bars: list[dict] = []
        for row in results:
            bars.append(
                {
                    "t": row.get("t"),           # ms epoch
                    "o": row.get("o"),
                    "h": row.get("h"),
                    "l": row.get("l"),
                    "c": row.get("c"),
                    "v": row.get("v"),
                    "n": row.get("n"),           # trade count (Polygon-specific)
                    "vw": row.get("vw"),         # VWAP (Polygon-specific)
                }
            )
        return bars

    def get_quote(self, symbol: str) -> dict:
        """Latest NBBO quote."""
        data = self._get(f"/v2/last/nbbo/{symbol.upper()}")
        quote = data.get("results") or {}
        return {
            "symbol": symbol.upper(),
            "bid": quote.get("p"),
            "ask": quote.get("P"),
            "bid_size": quote.get("s"),
            "ask_size": quote.get("S"),
            "t": quote.get("t"),
        }

    def get_quotes(self, symbols: list[str]) -> dict[str, dict]:
        return {s.upper(): self.get_quote(s) for s in symbols}
