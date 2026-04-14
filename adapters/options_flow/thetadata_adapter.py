"""
adapters/options_flow/thetadata_adapter.py — ThetaData free-tier REST adapter.

Docs: https://docs.thetadata.us/

ENV vars
--------
    THETADATA_API_KEY   ThetaData API key (required)
    THETADATA_BASE_URL  override API base URL (default: https://api.thetadata.us/v2)
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

from providers.options_flow import OptionsFlowResult
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = os.environ.get("THETADATA_BASE_URL", "https://api.thetadata.us/v2")
_API_KEY = os.environ.get("THETADATA_API_KEY", "")


class ThetaDataAdapter:
    """OptionsFlowProvider backed by ThetaData REST API."""

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{_BASE_URL}{path}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{qs}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {_API_KEY}",
                "Accept": "application/json",
                "User-Agent": "quant-platform/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())
        except Exception as exc:
            logger.warning("ThetaData request failed %s: %s", path, exc)
            return {}

    def get_flow(self, symbol: str, lookback_days: int = 1) -> list[dict]:
        data = self._get(
            f"/options/trade",
            params={"root": symbol.upper(), "exp": "all", "days_back": lookback_days},
        )
        return data.get("response", []) if isinstance(data, dict) else []

    def unusual_activity_score(self, symbol: str) -> OptionsFlowResult:
        flow = self.get_flow(symbol, lookback_days=1)
        call_vol = sum(float(r.get("size", 0)) for r in flow if str(r.get("right", "")).upper() == "C")
        put_vol = sum(float(r.get("size", 0)) for r in flow if str(r.get("right", "")).upper() == "P")

        # 20-day baseline
        flow_20d = self.get_flow(symbol, lookback_days=20)
        days = max(1, len({str(r.get("date", i)) for i, r in enumerate(flow_20d)}))
        call_20d = sum(float(r.get("size", 0)) for r in flow_20d if str(r.get("right", "")).upper() == "C")
        put_20d = sum(float(r.get("size", 0)) for r in flow_20d if str(r.get("right", "")).upper() == "P")
        baseline = (call_20d / put_20d if put_20d > 0 else 1.0) / days

        ratio = call_vol / put_vol if put_vol > 0 else (2.0 if call_vol > 0 else 1.0)
        unusual = baseline > 0 and (ratio > 1.5 * baseline or ratio < baseline / 1.5)
        score = max(-1.0, min(1.0, (ratio - baseline) / (baseline + 1e-6)))

        return OptionsFlowResult(
            symbol=symbol.upper(),
            call_volume=call_vol,
            put_volume=put_vol,
            call_put_ratio=round(ratio, 4),
            unusual_score=round(score, 4),
            avg_20d_call_put_ratio=round(baseline, 4),
            is_unusual=unusual,
        )
