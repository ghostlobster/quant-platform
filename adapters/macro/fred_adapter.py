"""
adapters/macro/fred_adapter.py — MacroDataProvider backed by the FRED REST API.

Reads ``FRED_API_KEY`` from the environment and pulls observations for
the requested series. Returns an empty :class:`pandas.Series` on any
HTTP / parse failure so callers can degrade gracefully (the same
contract sentiment adapters honour).

API reference: https://fred.stlouisfed.org/docs/api/fred/series_observations.html
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FREDAdapter:
    """Pull macro series from the St. Louis Fed FRED API."""

    def __init__(self, api_key: str | None = None, timeout: float = 10.0) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY", "")
        self._timeout = timeout

    def get_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        if not self._api_key:
            log.warning("fred_adapter: no FRED_API_KEY configured")
            return pd.Series(dtype=float, name=series_id)

        params: dict[str, str] = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        }
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end

        url = f"{_BASE}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "quant-platform/1.0"})
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310 - https only
                payload = json.loads(resp.read())
        except Exception as exc:
            log.warning("fred_adapter: fetch failed", series_id=series_id, error=str(exc))
            return pd.Series(dtype=float, name=series_id)

        observations = payload.get("observations", [])
        if not observations:
            return pd.Series(dtype=float, name=series_id)

        dates: list[pd.Timestamp] = []
        values: list[float] = []
        for obs in observations:
            raw = obs.get("value", ".")
            if raw in (".", "", None):
                continue
            try:
                values.append(float(raw))
                dates.append(pd.Timestamp(obs["date"]))
            except (ValueError, KeyError):
                continue

        return pd.Series(values, index=pd.DatetimeIndex(dates), name=series_id, dtype=float)
