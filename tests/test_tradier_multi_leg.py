"""
tests/test_tradier_multi_leg.py — tradier_bridge.place_multi_leg payload.

Mocks ``requests.post`` so no HTTP fires and asserts the multileg payload
matches Tradier's documented form.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.tradier_bridge as tb
from strategies.options_legs import iron_condor, vertical_spread


@pytest.fixture
def configured_tradier(monkeypatch):
    """Inject sentinel credentials so _is_configured() returns True."""
    monkeypatch.setattr(tb, "TRADIER_API_KEY", "test-key")
    monkeypatch.setattr(tb, "TRADIER_ACCOUNT_ID", "test-acct")
    monkeypatch.setattr(tb, "TRADIER_SANDBOX", True)
    monkeypatch.setattr(tb, "BASE_URL", "https://sandbox.example/v1")


def _install_fake_post(monkeypatch, captured: dict, payload_json: dict | None = None):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return payload_json or {"order": {"id": "abc-123", "status": "ok"}}

    def _fake_post(url, headers=None, data=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        return _Resp()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=_fake_post))


# ── OCC symbol helper ────────────────────────────────────────────────────────

def test_occ_symbol_format():
    sym = tb._occ_symbol("SPY", "2026-06-19", "call", 450.0)
    assert sym == "SPY260619C00450000"

    sym_put = tb._occ_symbol("AAPL", "2026-12-18", "put", 175.5)
    assert sym_put == "AAPL261218P00175500"


# ── place_multi_leg payload shape ────────────────────────────────────────────

def test_place_multi_leg_builds_iron_condor_payload(configured_tradier, monkeypatch):
    legs = iron_condor(
        "SPY", "2026-06-19",
        put_long_strike=440, put_short_strike=445,
        call_short_strike=455, call_long_strike=460,
        qty=2,
    )
    captured: dict = {}
    _install_fake_post(monkeypatch, captured)

    result = tb.place_multi_leg(legs, order_type="credit")
    assert result is not None
    assert result["order_id"] == "abc-123"
    assert result["legs"] == 4
    assert result["underlying"] == "SPY"
    assert result["order_type"] == "credit"

    data = captured["data"]
    assert data["class"] == "multileg"
    assert data["symbol"] == "SPY"
    assert data["type"] == "credit"
    assert data["duration"] == "day"
    # Four indexed legs.
    assert data["option_symbol[0]"] == "SPY260619P00440000"
    assert data["side[0]"] == "buy_to_open"
    assert data["quantity[0]"] == "2"
    assert data["option_symbol[3]"] == "SPY260619C00460000"
    assert data["side[3]"] == "buy_to_open"


def test_place_multi_leg_uses_provided_option_symbol(configured_tradier, monkeypatch):
    """Pre-resolved OCC symbols on the leg take precedence over the
    OCC builder so callers can paste in venue-specific identifiers."""
    legs = vertical_spread("SPY", "2026-06-19", 450, 455)
    legs = [
        type(legs[0])(
            underlying=legs[0].underlying, expiry=legs[0].expiry,
            strike=legs[0].strike, option_type=legs[0].option_type,
            side=legs[0].side, qty=legs[0].qty,
            option_symbol="CUSTOM_LONG",
        ),
        legs[1],  # second leg auto-derives OCC
    ]
    captured: dict = {}
    _install_fake_post(monkeypatch, captured)

    tb.place_multi_leg(legs)
    data = captured["data"]
    assert data["option_symbol[0]"] == "CUSTOM_LONG"
    assert data["option_symbol[1]"] == "SPY260619C00455000"


def test_place_multi_leg_rejects_empty_legs():
    with pytest.raises(ValueError, match="at least one leg"):
        tb.place_multi_leg([])


def test_place_multi_leg_rejects_mixed_underlyings(configured_tradier):
    spy = vertical_spread("SPY", "2026-06-19", 450, 455)
    qqq = vertical_spread("QQQ", "2026-06-19", 350, 355)
    with pytest.raises(ValueError, match="same underlying"):
        tb.place_multi_leg(spy + qqq)


def test_place_multi_leg_returns_none_when_unconfigured(monkeypatch):
    monkeypatch.setattr(tb, "TRADIER_API_KEY", "")
    monkeypatch.setattr(tb, "TRADIER_ACCOUNT_ID", "")
    legs = vertical_spread("SPY", "2026-06-19", 450, 455)
    assert tb.place_multi_leg(legs) is None


def test_place_multi_leg_swallows_http_failure(configured_tradier, monkeypatch):
    """A 5xx from Tradier returns None instead of raising — matches the
    project convention used by place_option_order."""
    class _BoomResp:
        def raise_for_status(self):
            raise RuntimeError("503")

        def json(self):
            return {}

    monkeypatch.setitem(
        sys.modules, "requests",
        SimpleNamespace(post=lambda *a, **kw: _BoomResp()),
    )
    legs = vertical_spread("SPY", "2026-06-19", 450, 455)
    assert tb.place_multi_leg(legs) is None
