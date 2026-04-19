"""
tests/test_polygon_backfill.py — tests for cron/polygon_backfill.py.

Monkeypatches :class:`PolygonAdapter` so no HTTP calls fire. Cache writes
go into a per-test temp SQLite file via ``data.db._DB_PATH``.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cron.polygon_backfill as backfill
import data.db as db_module


def _cache_read(ticker: str, period: str):
    """Lazy-import to avoid yfinance at collect time (optional local dep)."""
    from data.fetcher import _cache_read as real

    return real(ticker, period)


class _FakeAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, str]] = []

    def get_bars(self, symbol, timeframe, start, end):  # noqa: D401
        self.calls.append((symbol, timeframe, start, end))
        if symbol == "FAIL":
            raise RuntimeError("boom")
        # Return a couple of mock bars at one-day intervals.
        return [
            {"t": 1_700_000_000_000, "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
            {"t": 1_700_086_400_000, "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "v": 20},
            {"t": 1_700_172_800_000, "o": 2.0, "h": 3.0, "l": 1.5, "c": 2.5, "v": 30},
        ]


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point data.db._DB_PATH at a per-test SQLite file."""
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    return tmp_path


def test_backfill_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    assert backfill.backfill(["AAPL"], "1Day", 30) == {}


def test_backfill_calls_adapter_for_each_ticker(monkeypatch, temp_db):
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


def test_backfill_swallows_per_ticker_errors(monkeypatch, temp_db):
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


def test_main_runs_happy_path(monkeypatch, temp_db):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    rc = backfill.main(["--tickers", "AAPL,SPY", "--days", "5", "--timeframe", "1Day"])
    assert rc == 0
    assert {c[0] for c in fake.calls} == {"AAPL", "SPY"}


# ── Cache write integration ──────────────────────────────────────────────────

def test_backfill_writes_to_cache_for_daily(monkeypatch, temp_db):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    counts = backfill.backfill(["AAPL"], "1Day", days=30, period="1mo")
    assert counts == {"AAPL": 3}

    # A subsequent read of the same (ticker, period) must now hit the cache.
    cached = _cache_read("AAPL", "1mo")
    assert cached is not None
    assert not cached.empty
    assert list(cached.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(cached) == 3


def test_backfill_skips_cache_for_intraday(monkeypatch, temp_db):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    counts = backfill.backfill(["AAPL"], "5Min", days=5)
    assert counts == {"AAPL": 3}
    # Non-daily timeframes are counted but not cached.
    assert _cache_read("AAPL", "5d") is None


def test_backfill_period_falls_back_to_days_mapping(monkeypatch, temp_db):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    fake = _FakeAdapter()
    monkeypatch.setattr(
        "adapters.market_data.polygon_adapter.PolygonAdapter",
        lambda: fake,
    )
    backfill.backfill(["AAPL"], "1Day", days=365)
    # 365 days → "1y" bucket.
    assert _cache_read("AAPL", "1y") is not None


# ── Period-bucket helper ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (1, "1d"),
        (4, "5d"),
        (7, "1mo"),
        (45, "3mo"),
        (120, "6mo"),
        (300, "1y"),
        (500, "2y"),
        (1500, "5y"),
        (10000, "5y"),
    ],
)
def test_period_for_days(days, expected):
    assert backfill._period_for_days(days) == expected

