"""
adapters/execution_algo/vwap_adapter.py — Volume-Weighted Average Price execution.

Sizes each child order proportionally to the historical intraday volume profile
over *VWAP_LOOKBACK_DAYS* trading days, then places them on a per-slice schedule.

NOTE: execute() blocks the calling thread for the full execution window. Never
call this directly from a Streamlit render function or an async event loop.
Use a background thread (e.g. concurrent.futures.ThreadPoolExecutor) at the
call site. Call stop() to interrupt an in-progress execution early.

ENV vars
--------
    VWAP_LOOKBACK_DAYS   days of history to build volume profile (default: 5)
    TWAP_SLICE_SECONDS   seconds between slices (shared with TWAP, default: 60)
"""
from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

from adapters.execution_algo.result import ExecutionResult

logger = logging.getLogger(__name__)


def _uniform_weights(n: int) -> list[float]:
    """Fallback: uniform weights when historical data unavailable."""
    return [1.0 / n] * n


class VWAPAdapter:
    """ExecutionAlgoProvider that weights child orders by historical volume profile."""

    def __init__(
        self,
        lookback_days: int | None = None,
        slice_seconds: int | None = None,
    ) -> None:
        self._lookback_days = lookback_days or int(
            os.environ.get("VWAP_LOOKBACK_DAYS", "5")
        )
        self._slice_seconds = slice_seconds or int(
            os.environ.get("TWAP_SLICE_SECONDS", "60")
        )
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Signal any in-progress execute() to abort after the current slice."""
        self._stop_event.set()

    def _get_volume_weights(self, symbol: str, n_slices: int) -> list[float]:
        """
        Build an intraday volume weight vector of length *n_slices*.
        Falls back to uniform weights if data unavailable.
        """
        try:
            from data.fetcher import fetch_ohlcv
            df = fetch_ohlcv(symbol, period="5d")
            if df.empty or "Volume" not in df.columns:
                return _uniform_weights(n_slices)
            vols = df["Volume"].dropna().tolist()
            if not vols:
                return _uniform_weights(n_slices)
            # Sample vols evenly across slices
            step = max(1, len(vols) // n_slices)
            sampled = [vols[i * step] for i in range(n_slices)]
            total = sum(sampled) or 1
            return [v / total for v in sampled]
        except Exception as exc:
            logger.warning("VWAP volume fetch failed (%s), using uniform: %s", symbol, exc)
            return _uniform_weights(n_slices)

    def execute(
        self,
        symbol: str,
        total_qty: float,
        side: str,
        broker: Any,
        *,
        duration_minutes: int = 30,
        decision_price: float = 0.0,
        **kwargs: Any,
    ) -> ExecutionResult:
        self._stop_event.clear()
        duration_seconds = duration_minutes * 60
        n_slices = max(1, math.ceil(duration_seconds / self._slice_seconds))
        weights = self._get_volume_weights(symbol, n_slices)

        logger.info(
            "VWAP: %s %s x%.4f in %d slices over %d min",
            side.upper(), symbol, total_qty, n_slices, duration_minutes,
        )

        fills: list[dict] = []
        placed = 0.0
        for i, weight in enumerate(weights):
            if self._stop_event.is_set():
                logger.warning("VWAP: execution aborted after %d/%d slices", i, n_slices)
                break
            is_last = i == n_slices - 1
            if is_last:
                qty = round(total_qty - placed, 4)
            else:
                qty = round(total_qty * weight, 4)
            if qty <= 0:
                continue
            fill = broker.place_order(
                symbol=symbol,
                qty=qty,
                side=side,
                order_type="market",
            )
            fills.append(fill)
            placed = round(placed + qty, 4)
            if not is_last:
                # Interruptible sleep: wakes immediately if stop() is called
                self._stop_event.wait(timeout=self._slice_seconds)

        return ExecutionResult.from_fills(
            fills=fills,
            symbol=symbol,
            side=side,
            algo="vwap",
            decision_price=decision_price,
        )
