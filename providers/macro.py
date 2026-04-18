"""
providers/macro.py — MacroDataProvider Protocol and factory.

Macro / alternative data is fetched through this layer so the rest of
the platform can stay agnostic of the upstream source. Currently
supported back-ends are the FRED public API and a deterministic mock
adapter for tests / offline runs.

ENV vars
--------
    MACRO_PROVIDER   fred | mock  (default: fred when FRED_API_KEY set, else mock)
    FRED_API_KEY     personal API key from https://fred.stlouisfed.org
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MacroDataProvider(Protocol):
    """Duck-typed interface for macro / alternative time-series data."""

    def get_series(
        self,
        series_id: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.Series:
        """Return a date-indexed series of observations for ``series_id``.

        Implementations should return an empty Series on failure rather
        than raising — callers are expected to handle missing macro
        data gracefully.
        """
        ...


def get_macro(provider: Optional[str] = None) -> MacroDataProvider:
    """Return a configured :class:`MacroDataProvider`.

    Resolution order: explicit ``provider`` arg → ``MACRO_PROVIDER`` env var
    → ``fred`` (when ``FRED_API_KEY`` set) → ``mock``.
    """
    name = provider or os.environ.get("MACRO_PROVIDER")
    if not name:
        name = "fred" if os.environ.get("FRED_API_KEY") else "mock"
    name = name.lower().strip()

    if name == "fred":
        from adapters.macro.fred_adapter import FREDAdapter
        return FREDAdapter()
    if name == "mock":
        from adapters.macro.mock_adapter import MockMacroAdapter
        return MockMacroAdapter()
    raise ValueError(
        f"Unknown macro provider: {name!r}. Valid options: fred, mock"
    )
