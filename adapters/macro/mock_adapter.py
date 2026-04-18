"""
adapters/macro/mock_adapter.py — deterministic MacroDataProvider for tests.

Returns a synthetic but reproducible time series for any series_id.
Used when ``MACRO_PROVIDER=mock`` (the default when ``FRED_API_KEY`` is
unset) so unit tests, CI runs, and local development never hit the
real FRED endpoint.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


class MockMacroAdapter:
    """Deterministic per-series synthetic macro data."""

    def get_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        start_ts = pd.Timestamp(start) if start else pd.Timestamp("2020-01-01")
        end_ts = pd.Timestamp(end) if end else pd.Timestamp("2024-12-31")
        idx = pd.date_range(start_ts, end_ts, freq="B")
        if len(idx) == 0:
            return pd.Series(dtype=float, name=series_id)

        seed = int(
            hashlib.sha1(series_id.encode("utf-8"), usedforsecurity=False).hexdigest()[:8],
            16,
        ) % 2**32
        rng = np.random.default_rng(seed)
        # Mean-reverting AR(1) around a series-specific level.
        level = (seed % 50) + 1.0
        eps = rng.standard_normal(len(idx))
        values = np.empty(len(idx))
        values[0] = level
        for t in range(1, len(idx)):
            values[t] = 0.95 * values[t - 1] + 0.05 * level + eps[t] * 0.1
        return pd.Series(values, index=idx, name=series_id, dtype=float)
