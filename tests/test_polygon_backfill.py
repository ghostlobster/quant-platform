"""
tests/test_polygon_backfill.py — tests for cron/polygon_backfill.py.

Monkeypatches :class:`PolygonAdapter` so no HTTP calls fire.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cron.polygon_backfill as backfill


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def get_bars(self, symbol, timeframe, start, end):  # noqa: D401
        self.calls.append((symbol, timeframe, start, end))
        if symbol == "FAIL":
            raise RuntimeError("boom")
        # Return a couple of mock bars.
        return [{"t": 0, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}] * 3


def test_backfill_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert backfill.backfill(["AAPL"], "1Day", 30) == {}


def test_backfill_calls_adapter_for_each_ticker(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )

    counts = backfill.backfill(["AAPL", "SPY"], "1Day", 30)
    assert counts == {"AAPL": 3, "SPY": 3}
    assert len(fake.calls) == 2
    assert fake.calls[0][0] == "AAPL"


def test_backfill_swallows_per_ticker_errors(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    counts = backfill.backfill(["AAPL", "FAIL", "SPY"], "1Day", 30)
    assert counts == {"AAPL": 3, "FAIL": 0, "SPY": 3}


def test_main_exits_zero_when_no_api_key(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    rc = backfill.main(["--tickers", "AAPL", "--days", "5"])
    assert rc == 0


def test_main_runs_happy_path(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    rc = backfill.main(["--tickers", "AAPL,SPY", "--days", "5", "--timeframe", "1Day"])
    assert rc == 0
    assert {c[0] for c in fake.calls} == {"AAPL", "SPY"}
