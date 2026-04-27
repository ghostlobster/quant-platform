"""
tests/test_ibkr_multi_asset.py — IBKR contract factory + adapter routing.

Mocks ``ib_insync`` with a tiny SimpleNamespace so neither a real TWS
connection nor the vendor SDK is required to exercise the multi-asset
plumbing.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.ibkr_bridge as ibkr_bridge
import data.db as db_module
from data.symbols import AssetClass, SymbolMeta, register


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    return tmp_path


@pytest.fixture
def fake_ib_insync(monkeypatch):
    """Replace ib_insync with a minimal stand-in."""

    class _Stock:
        def __init__(self, symbol, exchange, currency):
            self.symbol = symbol
            self.exchange = exchange
            self.currency = currency
            self.kind = "Stock"

    class _Forex:
        def __init__(self, pair):
            self.pair = pair
            self.kind = "Forex"

    class _Future:
        def __init__(self, symbol, expiry, exchange, currency="USD", multiplier=""):
            self.symbol = symbol
            self.expiry = expiry
            self.exchange = exchange
            self.currency = currency
            self.multiplier = multiplier
            self.kind = "Future"

    fake = SimpleNamespace(
        Stock=_Stock,
        Forex=_Forex,
        Future=_Future,
        IB=MagicMock,
        MarketOrder=MagicMock,
        LimitOrder=MagicMock,
    )
    monkeypatch.setattr(ibkr_bridge, "_ib_insync", fake)
    monkeypatch.setattr(ibkr_bridge, "_IB_AVAILABLE", True)
    return fake


# ── make_contract: stock / forex / future ───────────────────────────────────

def test_make_contract_stock(fake_ib_insync):
    c = ibkr_bridge.make_contract("AAPL", asset_class="stock")
    assert c.kind == "Stock"
    assert c.symbol == "AAPL"
    assert c.exchange == "SMART"
    assert c.currency == "USD"


def test_make_contract_lse_stock(fake_ib_insync):
    c = ibkr_bridge.make_contract(
        "HSBA", asset_class="stock", exchange="LSE", currency="GBP",
    )
    assert c.exchange == "LSE"
    assert c.currency == "GBP"


def test_make_contract_forex_split_six_letters(fake_ib_insync):
    c = ibkr_bridge.make_contract("EURUSD", asset_class="forex")
    assert c.kind == "Forex"
    assert c.pair == "EURUSD"


def test_make_contract_forex_with_currency_quote(fake_ib_insync):
    c = ibkr_bridge.make_contract("GBP", asset_class="forex", currency="JPY")
    assert c.kind == "Forex"
    assert c.pair == "GBPJPY"


def test_make_contract_future_requires_expiry(fake_ib_insync):
    with pytest.raises(ValueError, match="expiry"):
        ibkr_bridge.make_contract("ES", asset_class="future")


def test_make_contract_future_full(fake_ib_insync):
    c = ibkr_bridge.make_contract(
        "ES", asset_class="future", expiry="202612", multiplier=50,
    )
    assert c.kind == "Future"
    assert c.symbol == "ES"
    assert c.expiry == "202612"
    assert c.exchange == "GLOBEX"
    assert c.multiplier == "50"


def test_make_contract_unknown_asset_class(fake_ib_insync):
    with pytest.raises(ValueError, match="asset_class"):
        ibkr_bridge.make_contract("BTC", asset_class="crypto")


def test_make_contract_requires_ib_insync(monkeypatch):
    monkeypatch.setattr(ibkr_bridge, "_IB_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="ib_insync"):
        ibkr_bridge.make_contract("AAPL")


# ── IBKRAdapter routing through bridge.place_order ──────────────────────────

def test_ibkr_adapter_routes_forex_via_registered_meta(
    temp_db, fake_ib_insync, monkeypatch,
):
    register(SymbolMeta("EUR", AssetClass.FOREX, "IDEALPRO", "USD"))

    captured: dict = {}

    def _fake_place_order(ticker, qty, side, order_type="MKT",
                          limit_price=None, **kwargs):
        captured.update(
            ticker=ticker, qty=qty, side=side, order_type=order_type,
            limit_price=limit_price, **kwargs,
        )
        return {"order_id": "ib-1", "status": "Filled"}

    monkeypatch.setattr(ibkr_bridge, "place_order", _fake_place_order)
    monkeypatch.delenv("MAX_POSITION_PCT", raising=False)

    from adapters.broker.ibkr_adapter import IBKRAdapter

    broker = IBKRAdapter()
    result = broker.place_order("eur", qty=10_000, side="buy")
    assert result["status"] == "Filled"
    assert result["asset_class"] == AssetClass.FOREX
    assert captured["ticker"] == "EUR"
    assert captured["asset_class"] == AssetClass.FOREX
    assert captured["exchange"] == "IDEALPRO"


def test_ibkr_adapter_falls_back_to_stock_for_unregistered(
    temp_db, fake_ib_insync, monkeypatch,
):
    captured: dict = {}

    def _fake_place_order(ticker, qty, side, **kwargs):
        captured.update(ticker=ticker, qty=qty, side=side, **kwargs)
        return {"order_id": "ib-2", "status": "Filled"}

    monkeypatch.setattr(ibkr_bridge, "place_order", _fake_place_order)
    monkeypatch.delenv("MAX_POSITION_PCT", raising=False)

    from adapters.broker.ibkr_adapter import IBKRAdapter

    broker = IBKRAdapter()
    result = broker.place_order("AAPL", qty=10, side="buy")
    assert result["status"] == "Filled"
    assert result["asset_class"] == AssetClass.STOCK
    assert captured["asset_class"] == AssetClass.STOCK
    assert captured["exchange"] == "SMART"


def test_ibkr_adapter_returns_failed_when_bridge_returns_empty(
    temp_db, fake_ib_insync, monkeypatch,
):
    monkeypatch.setattr(ibkr_bridge, "place_order", lambda *a, **kw: {})
    monkeypatch.delenv("MAX_POSITION_PCT", raising=False)

    from adapters.broker.ibkr_adapter import IBKRAdapter

    result = IBKRAdapter().place_order("AAPL", qty=1, side="buy")
    assert result["status"] == "failed"
    assert result["symbol"] == "AAPL"


def test_ibkr_adapter_guard_rejects_blocklist(temp_db, fake_ib_insync, monkeypatch):
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "AAPL")
    monkeypatch.setattr(ibkr_bridge, "place_order",
                        lambda *a, **kw: pytest.fail("bridge should not be called"))

    from adapters.broker.ibkr_adapter import IBKRAdapter

    result = IBKRAdapter().place_order("AAPL", qty=1, side="buy")
    assert result["status"] == "rejected"
    assert result["reason"] == "symbol_blocklist"
