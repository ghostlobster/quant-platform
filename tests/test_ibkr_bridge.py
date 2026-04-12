"""
Tests for broker/ibkr_bridge.py

Strategy: mock ib_insync.IB entirely so the tests never require the real
package to be installed. Uses the same reload pattern as test_ccxt_bridge.py.
"""
import sys
import types
import importlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to (re)load ibkr_bridge with a fake ib_insync module injected
# ---------------------------------------------------------------------------

def _make_fake_ib_insync(ib_instance=None):
    """Return a minimal fake ib_insync module."""
    fake = types.ModuleType("ib_insync")
    mock_ib = ib_instance or MagicMock()
    fake.IB = MagicMock(return_value=mock_ib)
    # Contract / order types
    fake.Stock = MagicMock(return_value=MagicMock())
    fake.MarketOrder = MagicMock(return_value=MagicMock())
    fake.LimitOrder = MagicMock(return_value=MagicMock())
    return fake, mock_ib


def _load_bridge(monkeypatch, *, ib_insync_module=None, env=None):
    """
    Import (or re-import) ibkr_bridge with the given environment and
    optional fake ib_insync module injected into sys.modules.
    Returns the freshly loaded module.
    """
    sys.modules.pop("broker.ibkr_bridge", None)
    sys.modules.pop("ib_insync", None)

    if ib_insync_module is not None:
        sys.modules["ib_insync"] = ib_insync_module

    env = env or {}
    for key in ("IBKR_HOST", "IBKR_PORT", "IBKR_CLIENT_ID", "IBKR_PAPER"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    import broker.ibkr_bridge as bridge
    return bridge


# ---------------------------------------------------------------------------
# Safe no-op: ib_insync not installed
# ---------------------------------------------------------------------------

class TestIbInsyncNotInstalled:
    """When ib_insync cannot be imported every function returns a safe empty value."""

    def _bridge(self, monkeypatch):
        sys.modules.pop("ib_insync", None)
        sys.modules.pop("broker.ibkr_bridge", None)
        with patch.dict(sys.modules, {"ib_insync": None}):
            return _load_bridge(monkeypatch, ib_insync_module=None)

    def test_get_account_info_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_account_info() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge(monkeypatch).get_positions() == []

    def test_place_order_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.place_order("AAPL", 1, "BUY") == {}

    def test_cancel_order_returns_false(self, monkeypatch):
        assert self._bridge(monkeypatch).cancel_order("123") is False

    def test_get_market_data_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_market_data("AAPL") == {}


# ---------------------------------------------------------------------------
# Connection error → safe defaults
# ---------------------------------------------------------------------------

class TestConnectionError:
    """When IB.connect raises, every function returns a safe empty value."""

    def _bridge_with_conn_error(self, monkeypatch):
        fake, mock_ib = _make_fake_ib_insync()
        mock_ib.connect.side_effect = ConnectionRefusedError("no TWS")
        return _load_bridge(monkeypatch, ib_insync_module=fake)

    def test_get_account_info_returns_empty_dict(self, monkeypatch):
        assert self._bridge_with_conn_error(monkeypatch).get_account_info() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge_with_conn_error(monkeypatch).get_positions() == []

    def test_place_order_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge_with_conn_error(monkeypatch)
        assert bridge.place_order("AAPL", 1, "BUY") == {}

    def test_cancel_order_returns_false(self, monkeypatch):
        assert self._bridge_with_conn_error(monkeypatch).cancel_order("42") is False

    def test_get_market_data_returns_empty_dict(self, monkeypatch):
        assert self._bridge_with_conn_error(monkeypatch).get_market_data("AAPL") == {}


# ---------------------------------------------------------------------------
# IBKR_PAPER env var controls port
# ---------------------------------------------------------------------------

class TestPaperPortSelection:
    def test_paper_true_uses_port_7497(self, monkeypatch):
        fake, mock_ib = _make_fake_ib_insync()
        mock_ib.accountSummary.return_value = []
        bridge = _load_bridge(
            monkeypatch, ib_insync_module=fake, env={"IBKR_PAPER": "true"}
        )
        bridge.get_account_info()
        mock_ib.connect.assert_called_once()
        _, kwargs = mock_ib.connect.call_args
        args = mock_ib.connect.call_args[0]
        assert args[1] == 7497

    def test_paper_false_uses_port_7496(self, monkeypatch):
        fake, mock_ib = _make_fake_ib_insync()
        mock_ib.accountSummary.return_value = []
        bridge = _load_bridge(
            monkeypatch, ib_insync_module=fake, env={"IBKR_PAPER": "false"}
        )
        bridge.get_account_info()
        mock_ib.connect.assert_called_once()
        args = mock_ib.connect.call_args[0]
        assert args[1] == 7496


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def _account_value(tag, value):
    av = MagicMock()
    av.tag = tag
    av.value = str(value)
    return av


class TestHappyPath:

    def _setup(self, monkeypatch, mock_ib=None):
        fake, instance = _make_fake_ib_insync(mock_ib)
        bridge = _load_bridge(monkeypatch, ib_insync_module=fake)
        return bridge, fake, instance

    # --- get_account_info -----------------------------------------------------

    def test_get_account_info_maps_summary(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.accountSummary.return_value = [
            _account_value("TotalCashValue", 50000.0),
            _account_value("NetLiquidation", 120000.0),
            _account_value("MaintMarginReq", 10000.0),
        ]
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        result = bridge.get_account_info()
        assert result == {"cash": 50000.0, "equity": 120000.0, "margin": 10000.0}

    def test_get_account_info_missing_tags_default_zero(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.accountSummary.return_value = []
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        result = bridge.get_account_info()
        assert result == {"cash": 0.0, "equity": 0.0, "margin": 0.0}

    def test_get_account_info_disconnects_on_success(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.accountSummary.return_value = []
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        bridge.get_account_info()
        mock_ib.disconnect.assert_called_once()

    # --- get_positions --------------------------------------------------------

    def test_get_positions_returns_list(self, monkeypatch):
        mock_ib = MagicMock()
        pos = MagicMock()
        pos.contract.symbol = "AAPL"
        pos.position = 10.0
        pos.avgCost = 150.0
        mock_ib.positions.return_value = [pos]
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        result = bridge.get_positions()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["qty"] == 10.0
        assert result[0]["avg_cost"] == 150.0

    def test_get_positions_empty(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.positions.return_value = []
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        assert bridge.get_positions() == []

    # --- place_order ----------------------------------------------------------

    def test_place_order_market_buy(self, monkeypatch):
        mock_ib = MagicMock()
        trade = MagicMock()
        trade.order.orderId = 42
        trade.orderStatus.status = "Submitted"
        mock_ib.placeOrder.return_value = trade
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        result = bridge.place_order("AAPL", 5, "BUY")
        assert result == {"order_id": "42", "status": "Submitted"}

    def test_place_order_bad_qty_raises(self, monkeypatch):
        bridge, _, _ = self._setup(monkeypatch)
        with pytest.raises(ValueError, match="qty must be positive"):
            bridge.place_order("AAPL", -1, "BUY")

    def test_place_order_bad_side_raises(self, monkeypatch):
        bridge, _, _ = self._setup(monkeypatch)
        with pytest.raises(ValueError, match="side must be"):
            bridge.place_order("AAPL", 1, "LONG")

    # --- cancel_order ---------------------------------------------------------

    def test_cancel_order_found(self, monkeypatch):
        mock_ib = MagicMock()
        trade = MagicMock()
        trade.order.orderId = 99
        mock_ib.openTrades.return_value = [trade]
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        assert bridge.cancel_order("99") is True
        mock_ib.cancelOrder.assert_called_once_with(trade.order)

    def test_cancel_order_not_found_returns_false(self, monkeypatch):
        mock_ib = MagicMock()
        mock_ib.openTrades.return_value = []
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        assert bridge.cancel_order("999") is False

    # --- get_market_data ------------------------------------------------------

    def test_get_market_data_returns_quote(self, monkeypatch):
        mock_ib = MagicMock()
        tick = MagicMock()
        tick.bid = 149.5
        tick.ask = 150.0
        tick.last = 149.8
        tick.volume = 1000000.0
        mock_ib.reqMktData.return_value = tick
        bridge, _, _ = self._setup(monkeypatch, mock_ib)
        result = bridge.get_market_data("AAPL")
        assert result["bid"] == 149.5
        assert result["ask"] == 150.0
        assert result["last"] == 149.8
        assert result["volume"] == 1000000.0
