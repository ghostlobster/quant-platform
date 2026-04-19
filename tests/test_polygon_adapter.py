"""
tests/test_polygon_adapter.py — tests for the Polygon.io adapter.

Every case monkeypatches ``requests.get`` so no HTTP calls actually fire.
Timeframe translation + quote / bar shape conversion are exercised directly.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import adapters.market_data.polygon_adapter as polymod
from adapters.market_data.polygon_adapter import PolygonAdapter, _parse_timeframe
from providers.market_data import get_market_data


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def _install_fake_requests(monkeypatch, payload, captured=None):
    def _fake_get(url, headers=None, params=None, timeout=None):
        if captured is not None:
            captured["url"] = url
            captured["params"] = params
        return _Resp(payload)

    monkeypatch.setattr(polymod, "_requests", SimpleNamespace(get=_fake_get))


# ── Timeframe parsing ────────────────────────────────────────────────────────

def test_parse_timeframe_alpaca_style():
    assert _parse_timeframe("1Day") == (1, "day")
    assert _parse_timeframe("5Min") == (5, "minute")
    assert _parse_timeframe("1Hour") == (1, "hour")


def test_parse_timeframe_polygon_native():
    assert _parse_timeframe("15/minute") == (15, "minute")
    assert _parse_timeframe("3/DAY") == (3, "day")


def test_parse_timeframe_rejects_garbage():
    with pytest.raises(ValueError, match="timeframe"):
        _parse_timeframe("weekly")


# ── get_bars: URL + payload shape ────────────────────────────────────────────

def test_get_bars_builds_polygon_url_and_maps_results(monkeypatch):
    captured: dict = {}
    _install_fake_requests(
        monkeypatch,
        payload={
            "results": [
                {"t": 1_700_000_000_000, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100.0},
                {"t": 1_700_086_400_000, "o": 1.5, "h": 2.5, "l": 1.0, "c": 2.0, "v": 150.0},
            ]
        },
        captured=captured,
    )
    adapter = PolygonAdapter(api_key="key")
    bars = adapter.get_bars("aapl", "1Day", "2024-01-01", "2024-01-05")
    assert captured["url"].endswith("/v2/aggs/ticker/AAPL/range/1/day/2024-01-01/2024-01-05")
    assert captured["params"]["adjusted"] == "true"
    assert captured["params"]["sort"] == "asc"
    assert len(bars) == 2
    assert bars[0]["o"] == 1.0
    assert bars[1]["c"] == 2.0


def test_get_bars_empty_results(monkeypatch):
    _install_fake_requests(monkeypatch, payload={"results": None})
    adapter = PolygonAdapter(api_key="key")
    assert adapter.get_bars("AAPL", "1Day", "2024-01-01", "2024-01-05") == []


def test_adjustment_false_via_env(monkeypatch):
    captured: dict = {}
    _install_fake_requests(
        monkeypatch,
        payload={"results": []},
        captured=captured,
    )
    monkeypatch.setenv("POLYGON_ADJUSTMENT", "false")
    PolygonAdapter(api_key="key").get_bars("AAPL", "1Day", "2024-01-01", "2024-01-05")
    assert captured["params"]["adjusted"] == "false"


# ── get_quote / get_quotes ───────────────────────────────────────────────────

def test_get_quote_maps_nbbo_payload(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        payload={"results": {"p": 100.0, "P": 100.2, "s": 10, "S": 20, "t": 1700000000000}},
    )
    adapter = PolygonAdapter(api_key="key")
    q = adapter.get_quote("aapl")
    assert q == {
        "symbol": "AAPL",
        "bid": 100.0,
        "ask": 100.2,
        "bid_size": 10,
        "ask_size": 20,
        "t": 1700000000000,
    }


def test_get_quotes_fans_out(monkeypatch):
    _install_fake_requests(
        monkeypatch,
        payload={"results": {"p": 1.0, "P": 1.1, "s": 5, "S": 6, "t": 0}},
    )
    adapter = PolygonAdapter(api_key="key")
    quotes = adapter.get_quotes(["aapl", "tsla"])
    assert set(quotes) == {"AAPL", "TSLA"}


# ── Provider factory ─────────────────────────────────────────────────────────

def test_factory_returns_polygon_adapter(monkeypatch):
    monkeypatch.setenv("POLYGON_API_KEY", "key")
    assert isinstance(get_market_data("polygon"), PolygonAdapter)


def test_factory_rejects_unknown():
    with pytest.raises(ValueError, match="polygon"):
        get_market_data("nonexistent")


# ── Import-time safety: adapter raises a clean error when requests missing ──

def test_adapter_raises_when_requests_missing(monkeypatch):
    monkeypatch.setattr(polymod, "_requests", None)
    with pytest.raises(ImportError, match="requests"):
        PolygonAdapter(api_key="key")
