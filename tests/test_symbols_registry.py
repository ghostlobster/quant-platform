"""
tests/test_symbols_registry.py — multi-asset symbol registry round-trip.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import data.db as db_module
from data.symbols import (
    AssetClass,
    SymbolMeta,
    default_for,
    get,
    list_by_class,
    register,
    resolve,
)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    return tmp_path


# ── SymbolMeta validation ────────────────────────────────────────────────────

def test_symbol_meta_validates_asset_class():
    with pytest.raises(ValueError, match="asset_class"):
        SymbolMeta("AAPL", "crypto", "SMART", "USD")


def test_symbol_meta_validates_currency():
    with pytest.raises(ValueError, match="currency"):
        SymbolMeta("AAPL", AssetClass.STOCK, "SMART", "DOLLARS")


def test_symbol_meta_future_requires_expiry():
    with pytest.raises(ValueError, match="expiry"):
        SymbolMeta("ES", AssetClass.FUTURE, "GLOBEX", "USD")


def test_symbol_meta_future_with_expiry_ok():
    meta = SymbolMeta("ES", AssetClass.FUTURE, "GLOBEX", "USD",
                       expiry="202612", multiplier=50)
    assert meta.expiry == "202612"
    assert meta.multiplier == 50


# ── Register / get round trip ────────────────────────────────────────────────

def test_register_and_get_round_trip(temp_db):
    meta = SymbolMeta("EUR", AssetClass.FOREX, "IDEALPRO", "USD")
    register(meta)
    got = get("eur")
    assert got is not None
    assert got.ticker == "EUR"
    assert got.asset_class == AssetClass.FOREX
    assert got.exchange == "IDEALPRO"


def test_get_returns_none_for_unknown(temp_db):
    assert get("UNKNOWN") is None


def test_register_upserts_existing(temp_db):
    register(SymbolMeta("EUR", AssetClass.FOREX, "IDEALPRO", "USD"))
    # Re-register with a different exchange — should overwrite, not duplicate.
    register(SymbolMeta("EUR", AssetClass.FOREX, "OTHER", "USD"))
    got = get("EUR")
    assert got.exchange == "OTHER"
    assert len(list_by_class(AssetClass.FOREX)) == 1


# ── list_by_class ────────────────────────────────────────────────────────────

def test_list_by_class_filters(temp_db):
    register(SymbolMeta("EUR", AssetClass.FOREX, "IDEALPRO", "USD"))
    register(SymbolMeta("GBP", AssetClass.FOREX, "IDEALPRO", "USD"))
    register(SymbolMeta("ES", AssetClass.FUTURE, "GLOBEX", "USD",
                         expiry="202612", multiplier=50))
    fxs = list_by_class(AssetClass.FOREX)
    futures = list_by_class(AssetClass.FUTURE)
    assert {m.ticker for m in fxs} == {"EUR", "GBP"}
    assert {m.ticker for m in futures} == {"ES"}


def test_list_by_class_rejects_unknown():
    with pytest.raises(ValueError):
        list_by_class("crypto")


# ── default_for / resolve ────────────────────────────────────────────────────

def test_default_for_returns_canonical():
    fx = default_for(AssetClass.FOREX)
    assert fx.exchange == "IDEALPRO"
    assert fx.currency == "USD"
    fut = default_for(AssetClass.FUTURE)
    assert fut.exchange == "GLOBEX"
    assert fut.expiry == "202612"


def test_resolve_uses_registered_when_present(temp_db):
    register(SymbolMeta("AAPL", AssetClass.STOCK, "SMART", "USD"))
    meta = resolve("AAPL")
    assert meta.asset_class == AssetClass.STOCK
    assert meta.exchange == "SMART"


def test_resolve_falls_back_with_substituted_ticker(temp_db):
    meta = resolve("UNKNOWN", fallback_class=AssetClass.STOCK)
    assert meta.ticker == "UNKNOWN"
    assert meta.asset_class == AssetClass.STOCK
    assert meta.exchange == "SMART"


def test_resolve_lse_fallback(temp_db):
    register(SymbolMeta("HSBA", AssetClass.STOCK, "LSE", "GBP"))
    meta = resolve("HSBA")
    assert meta.exchange == "LSE"
    assert meta.currency == "GBP"


def test_default_for_rejects_unknown():
    with pytest.raises(ValueError):
        default_for("crypto")
