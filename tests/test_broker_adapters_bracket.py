"""
tests/test_broker_adapters_bracket.py — adapter-layer wiring for P1.3.

Covers PaperBrokerAdapter.place_bracket (happy path + guard reject) and
the Alpaca bracket-order payload (mocks ``requests.post``). IBKR/Schwab
stubs are sanity-checked for the guard-then-NotImplementedError flow.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import broker.alpaca_bridge as alpaca_bridge
import broker.paper_trader as pt
import data.db as db_module
from providers.broker import OrderIntent


@pytest.fixture
def paper_env(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "paper.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    monkeypatch.setattr(pt, "MAX_DRAWDOWN_PCT", 0.99)
    pt.init_paper_tables()
    yield tmp_path


# ── PaperBrokerAdapter.place_bracket ──────────────────────────────────────────

def test_paper_adapter_place_bracket_happy_path(paper_env, monkeypatch):
    monkeypatch.delenv("MAX_POSITION_PCT", raising=False)
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    broker = PaperBrokerAdapter()
    intent = OrderIntent(
        symbol="AAPL", qty=5, side="buy",
        limit_price=100.0, take_profit=110.0, stop_loss=95.0,
    )
    result = broker.place_bracket(intent)
    assert result["status"] == "parent_filled"
    assert result["ticker"] == "AAPL"
    # Child recorded as pending.
    assert len(pt.get_pending_brackets()) == 1


def test_paper_adapter_place_bracket_guard_rejects(paper_env, monkeypatch):
    # Symbol blocklist should stop the order before any fill.
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "AAPL")
    from adapters.broker.paper_adapter import PaperBrokerAdapter

    broker = PaperBrokerAdapter()
    intent = OrderIntent(
        symbol="AAPL", qty=5, side="buy",
        limit_price=100.0, stop_loss=95.0,
    )
    result = broker.place_bracket(intent)
    assert result["status"] == "rejected"
    assert result["reason"] == "symbol_blocklist"
    # No pending bracket created.
    assert pt.get_pending_brackets() == []


# ── AlpacaBrokerAdapter.place_bracket (bridge call mocked) ────────────────────

def test_alpaca_adapter_place_bracket_passes_children_through(monkeypatch):
    from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter

    captured: dict = {}

    def _fake_place_bracket_order(ticker, qty, side, *, take_profit, stop_loss, trail_percent):
        captured.update(
            ticker=ticker, qty=qty, side=side,
            take_profit=take_profit, stop_loss=stop_loss,
            trail_percent=trail_percent,
        )
        return alpaca_bridge.AlpacaOrder(
            order_id="abc", ticker=ticker, qty=qty, side=side,
            order_type="bracket", status="accepted",
        )

    monkeypatch.setattr(alpaca_bridge, "place_bracket_order", _fake_place_bracket_order)
    broker = AlpacaBrokerAdapter()
    intent = OrderIntent(
        symbol="AAPL", qty=10, side="buy",
        take_profit=120.0, trail_percent=0.05,
    )
    result = broker.place_bracket(intent)
    assert result["order_id"] == "abc"
    assert result["status"] == "accepted"
    assert result["children"]["take_profit"] == 120.0
    assert result["children"]["trail_percent"] == 0.05
    assert captured["take_profit"] == 120.0
    assert captured["trail_percent"] == 0.05


def test_alpaca_adapter_place_bracket_handles_bridge_failure(monkeypatch):
    from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter

    monkeypatch.setattr(alpaca_bridge, "place_bracket_order", lambda *a, **kw: None)
    broker = AlpacaBrokerAdapter()
    intent = OrderIntent(symbol="AAPL", qty=10, side="buy", stop_loss=95.0)
    result = broker.place_bracket(intent)
    assert result["status"] == "failed"


def test_alpaca_adapter_place_bracket_guard_rejects(monkeypatch):
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "AAPL")
    from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter

    broker = AlpacaBrokerAdapter()
    intent = OrderIntent(symbol="AAPL", qty=10, side="buy", stop_loss=95.0)
    result = broker.place_bracket(intent)
    assert result["status"] == "rejected"
    assert result["reason"] == "symbol_blocklist"


# ── Alpaca bridge place_bracket_order payload ─────────────────────────────────

def test_alpaca_bridge_place_bracket_builds_payload(monkeypatch):
    monkeypatch.setattr(alpaca_bridge, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(alpaca_bridge, "ALPACA_SECRET_KEY", "secret")
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "bracket-1", "status": "accepted"}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    fake_requests = SimpleNamespace(post=_fake_post)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    order = alpaca_bridge.place_bracket_order(
        "AAPL", qty=5, side="buy",
        take_profit=110.0, stop_loss=95.0,
    )
    assert order is not None
    assert order.order_id == "bracket-1"
    payload = captured["json"]
    assert payload["order_class"] == "bracket"
    assert payload["take_profit"] == {"limit_price": "110.0"}
    assert payload["stop_loss"] == {"stop_price": "95.0"}


def test_alpaca_bridge_place_bracket_uses_trail_percent(monkeypatch):
    monkeypatch.setattr(alpaca_bridge, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(alpaca_bridge, "ALPACA_SECRET_KEY", "secret")
    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"id": "bracket-2", "status": "accepted"}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=_fake_post))
    alpaca_bridge.place_bracket_order(
        "AAPL", qty=5, side="buy",
        take_profit=110.0, trail_percent=0.03,
    )
    # trail_percent takes precedence over stop_loss in the stop leg.
    assert captured["json"]["stop_loss"] == {"trail_percent": "3.0"}


def test_alpaca_bridge_place_bracket_requires_at_least_one_child():
    with pytest.raises(ValueError, match="take_profit"):
        alpaca_bridge.place_bracket_order("AAPL", qty=5, side="buy")


def test_alpaca_bridge_place_bracket_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        alpaca_bridge.place_bracket_order(
            "AAPL", qty=5, side="hold", stop_loss=95.0,
        )


def test_alpaca_bridge_place_bracket_rejects_non_positive_qty():
    with pytest.raises(ValueError, match="qty"):
        alpaca_bridge.place_bracket_order(
            "AAPL", qty=0, side="buy", stop_loss=95.0,
        )


def test_alpaca_bridge_place_bracket_not_configured(monkeypatch):
    monkeypatch.setattr(alpaca_bridge, "ALPACA_API_KEY", "")
    monkeypatch.setattr(alpaca_bridge, "ALPACA_SECRET_KEY", "")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    assert alpaca_bridge.place_bracket_order(
        "AAPL", qty=5, side="buy", stop_loss=95.0,
    ) is None


# ── IBKR / Schwab stubs — guard precedes NotImplementedError ──────────────────

def test_ibkr_adapter_place_bracket_guard_rejects(monkeypatch):
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "AAPL")
    # Force the ibapi availability flag without actually importing ibapi.
    import adapters.broker.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "_IBAPI_AVAILABLE", True)
    broker = ibkr_mod.IBKRAdapter()
    intent = OrderIntent(symbol="AAPL", qty=5, side="buy", stop_loss=95.0)
    result = broker.place_bracket(intent)
    assert result["status"] == "rejected"
    assert result["reason"] == "symbol_blocklist"


def test_ibkr_adapter_place_bracket_not_implemented(monkeypatch):
    import adapters.broker.ibkr_adapter as ibkr_mod

    monkeypatch.setattr(ibkr_mod, "_IBAPI_AVAILABLE", True)
    monkeypatch.delenv("SYMBOL_BLOCKLIST", raising=False)
    broker = ibkr_mod.IBKRAdapter()
    intent = OrderIntent(symbol="AAPL", qty=5, side="buy", stop_loss=95.0)
    with pytest.raises(NotImplementedError):
        broker.place_bracket(intent)


def test_schwab_adapter_place_bracket_guard_rejects(monkeypatch):
    monkeypatch.setenv("SYMBOL_BLOCKLIST", "AAPL")
    import adapters.broker.schwab_adapter as schwab_mod

    monkeypatch.setattr(schwab_mod, "_SCHWAB_AVAILABLE", True)
    broker = schwab_mod.SchwabAdapter()
    intent = OrderIntent(symbol="AAPL", qty=5, side="buy", stop_loss=95.0)
    result = broker.place_bracket(intent)
    assert result["status"] == "rejected"
    assert result["reason"] == "symbol_blocklist"


def test_schwab_adapter_place_bracket_not_implemented(monkeypatch):
    import adapters.broker.schwab_adapter as schwab_mod

    monkeypatch.setattr(schwab_mod, "_SCHWAB_AVAILABLE", True)
    monkeypatch.delenv("SYMBOL_BLOCKLIST", raising=False)
    broker = schwab_mod.SchwabAdapter()
    intent = OrderIntent(symbol="AAPL", qty=5, side="buy", stop_loss=95.0)
    with pytest.raises(NotImplementedError):
        broker.place_bracket(intent)
