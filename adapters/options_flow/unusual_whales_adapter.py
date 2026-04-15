"""
adapters/options_flow/unusual_whales_adapter.py — Unusual Whales REST adapter.

Docs: https://unusualwhales.com/api

ENV vars
--------
    UNUSUAL_WHALES_TOKEN   API bearer token (required)
    UNUSUAL_WHALES_BASE_URL override base URL (default: https://api.unusualwhales.com)
"""
from __future__ import annotations

import json
import os
import urllib.request

from providers.options_flow import OptionsFlowResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.unusualwhales.com"


class UnusualWhalesAdapter:
    """OptionsFlowProvider backed by the Unusual Whales API."""

    def __init__(self) -> None:
        # Read credentials at instantiation time so the env is fully loaded
        self._base_url = os.environ.get("UNUSUAL_WHALES_BASE_URL", _DEFAULT_BASE_URL)
        self._token = os.environ.get("UNUSUAL_WHALES_TOKEN", "")
        if not self._token:
            logger.warning("UNUSUAL_WHALES_TOKEN is not set; requests will be unauthenticated")

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/json",
                "User-Agent": "quant-platform/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.warning("UnusualWhales request failed %s: %s", path, exc)
            return {}

    def get_flow(self, symbol: str, lookback_days: int = 1) -> list[dict]:
        data = self._get(f"/api/v2/options/flow/{symbol.upper()}")
        return data.get("data", []) if isinstance(data, dict) else []

    def unusual_activity_score(self, symbol: str) -> OptionsFlowResult:
        flow = self.get_flow(symbol)
        call_vol = sum(
            float(r.get("volume", 0)) for r in flow
            if str(r.get("option_type", r.get("type", ""))).lower() in ("call", "c")
        )
        put_vol = sum(
            float(r.get("volume", 0)) for r in flow
            if str(r.get("option_type", r.get("type", ""))).lower() in ("put", "p")
        )
        ratio = call_vol / put_vol if put_vol > 0 else (2.0 if call_vol > 0 else 1.0)
        baseline = 1.0
        unusual = ratio > 1.5 or ratio < 0.67
        score = max(-1.0, min(1.0, (ratio - baseline) / (baseline + 1e-6)))

        return OptionsFlowResult(
            symbol=symbol.upper(),
            call_volume=call_vol,
            put_volume=put_vol,
            call_put_ratio=round(ratio, 4),
            unusual_score=round(score, 4),
            avg_20d_call_put_ratio=baseline,
            is_unusual=unusual,
        )
