"""
Tests for broker/schwab_bridge.py

Strategy: mock the schwab import at the module level so the tests never require
the real package to be installed. Uses the same reload pattern as test_ccxt_bridge.py.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to (re)load schwab_bridge with a fake schwab module injected
# ---------------------------------------------------------------------------

def _make_fake_schwab(client_instance=None):
    """Return a minimal fake schwab module with auth and orders sub-modules."""
    fake = types.ModuleType("schwab")

    # auth sub-module
    fake.auth = types.ModuleType("schwab.auth")
    mock_client = client_instance or MagicMock()
    fake.auth.client_from_token_file = MagicMock(return_value=mock_client)

    # orders sub-module
    fake.orders = types.ModuleType("schwab.orders")
    fake.orders.equities = MagicMock()

    return fake, mock_client


def _load_bridge(monkeypatch, *, schwab_module=None, env=None):
    """
    Import (or re-import) schwab_bridge with the given environment and
    optional fake schwab module injected into sys.modules.
    Returns the freshly loaded module.
    """
    sys.modules.pop("broker.schwab_bridge", None)
    sys.modules.pop("schwab", None)
    sys.modules.pop("schwab.auth", None)
    sys.modules.pop("schwab.orders", None)

    if schwab_module is not None:
        sys.modules["schwab"] = schwab_module
        sys.modules["schwab.auth"] = schwab_module.auth
        sys.modules["schwab.orders"] = schwab_module.orders

    env = env or {}
    for key in (
        "SCHWAB_APP_KEY", "SCHWAB_APP_SECRET", "SCHWAB_CALLBACK_URL",
        "SCHWAB_TOKEN_PATH", "SCHWAB_ACCOUNT_HASH", "SCHWAB_SANDBOX",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    import broker.schwab_bridge as bridge
    return bridge


CREDS_ENV = {
    "SCHWAB_APP_KEY": "test_key",
    "SCHWAB_APP_SECRET": "test_secret",
    "SCHWAB_ACCOUNT_HASH": "ABC123HASH",
    "SCHWAB_TOKEN_PATH": "/tmp/fake_token.json",
}


def _ok_response(data: dict):
    """Build a mock response that returns the given dict."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    resp.headers = {"Location": "/v1/accounts/ABC123HASH/orders/9001"}
    resp.status_code = 200
    return resp


# ---------------------------------------------------------------------------
# Safe no-op: schwab not installed
# ---------------------------------------------------------------------------

class TestSchwabNotInstalled:
    """When schwab cannot be imported every function returns a safe empty value."""

    def _bridge(self, monkeypatch):
        sys.modules.pop("schwab", None)
        sys.modules.pop("broker.schwab_bridge", None)
        with patch.dict(sys.modules, {"schwab": None}):
            return _load_bridge(monkeypatch, schwab_module=None)

    def test_get_account_info_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_account_info() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge(monkeypatch).get_positions() == []

    def test_place_order_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.place_order("AAPL", 1, "BUY") == {}

    def test_get_quotes_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.get_quotes(["AAPL"]) == {}


# ---------------------------------------------------------------------------
# Safe no-op: token file missing
# ---------------------------------------------------------------------------

class TestMissingTokenFile:
    """When the token file doesn't exist every function returns a safe value."""

    def _bridge_with_missing_token(self, monkeypatch):
        fake, mock_client = _make_fake_schwab()
        fake.auth.client_from_token_file.side_effect = FileNotFoundError("no token")
        return _load_bridge(monkeypatch, schwab_module=fake, env=CREDS_ENV)

    def test_get_account_info_returns_empty_dict(self, monkeypatch):
        assert self._bridge_with_missing_token(monkeypatch).get_account_info() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge_with_missing_token(monkeypatch).get_positions() == []

    def test_place_order_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge_with_missing_token(monkeypatch)
        assert bridge.place_order("AAPL", 1, "BUY") == {}

    def test_get_quotes_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge_with_missing_token(monkeypatch)
        assert bridge.get_quotes(["AAPL"]) == {}


# ---------------------------------------------------------------------------
# Safe no-op: credentials absent
# ---------------------------------------------------------------------------

class TestCredentialsAbsent:
    def _bridge(self, monkeypatch):
        fake, _ = _make_fake_schwab()
        return _load_bridge(monkeypatch, schwab_module=fake, env={})

    def test_get_account_info_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_account_info() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge(monkeypatch).get_positions() == []

    def test_place_order_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).place_order("AAPL", 1, "BUY") == {}

    def test_get_quotes_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_quotes(["AAPL"]) == {}


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def _bridge(self, monkeypatch):
        fake, _ = _make_fake_schwab()
        return _load_bridge(monkeypatch, schwab_module=fake, env={})

    def test_qty_zero_raises_value_error(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        with pytest.raises(ValueError, match="qty must be positive"):
            bridge.place_order("AAPL", 0, "BUY")

    def test_qty_negative_raises_value_error(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        with pytest.raises(ValueError, match="qty must be positive"):
            bridge.place_order("AAPL", -1, "SELL")

    def test_invalid_side_raises_value_error(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        with pytest.raises(ValueError, match="side must be"):
            bridge.place_order("AAPL", 1, "LONG")


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def _setup(self, monkeypatch, client_instance=None):
        fake, mock_client = _make_fake_schwab(client_instance)
        # Account.Fields enum
        mock_client.Account = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        bridge = _load_bridge(monkeypatch, schwab_module=fake, env=CREDS_ENV)
        return bridge, fake, mock_client

    # --- get_account_info -----------------------------------------------------

    def test_get_account_info_maps_balances(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        mock_client.get_account.return_value = _ok_response({
            "securitiesAccount": {
                "currentBalances": {
                    "cashBalance": 25000.0,
                    "liquidationValue": 75000.0,
                    "maintenanceRequirement": 5000.0,
                }
            }
        })
        bridge, _, _ = self._setup(monkeypatch, mock_client)
        result = bridge.get_account_info()
        assert result == {"cash": 25000.0, "equity": 75000.0, "margin": 5000.0}

    def test_get_account_info_missing_keys_default_zero(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        mock_client.get_account.return_value = _ok_response(
            {"securitiesAccount": {"currentBalances": {}}}
        )
        bridge, _, _ = self._setup(monkeypatch, mock_client)
        result = bridge.get_account_info()
        assert result == {"cash": 0.0, "equity": 0.0, "margin": 0.0}

    # --- get_positions --------------------------------------------------------

    def test_get_positions_returns_list(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        mock_client.get_account.return_value = _ok_response({
            "securitiesAccount": {
                "positions": [
                    {
                        "instrument": {"symbol": "AAPL"},
                        "longQuantity": 10.0,
                        "averagePrice": 150.0,
                        "marketValue": 1600.0,
                    }
                ]
            }
        })
        bridge, _, _ = self._setup(monkeypatch, mock_client)
        result = bridge.get_positions()
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"
        assert result[0]["qty"] == 10.0
        assert result[0]["avg_cost"] == 150.0
        assert result[0]["market_value"] == 1600.0

    def test_get_positions_empty(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        mock_client.get_account.return_value = _ok_response(
            {"securitiesAccount": {"positions": []}}
        )
        bridge, _, _ = self._setup(monkeypatch, mock_client)
        assert bridge.get_positions() == []

    # --- place_order ----------------------------------------------------------

    def test_place_order_market_buy_returns_order_id(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"Location": "/v1/accounts/ABC/orders/9001"}
        mock_client.place_order.return_value = resp
        bridge, fake, _ = self._setup(monkeypatch, mock_client)
        result = bridge.place_order("AAPL", 5, "BUY")
        assert result["order_id"] == "9001"
        assert result["status"] == "ACCEPTED"

    def test_place_order_calls_equity_buy_market(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"Location": "/orders/42"}
        mock_client.place_order.return_value = resp
        bridge, fake, _ = self._setup(monkeypatch, mock_client)
        bridge.place_order("MSFT", 3, "BUY")
        fake.orders.equities.equity_buy_market.assert_called_once_with("MSFT", 3)

    def test_place_order_sell_calls_equity_sell_market(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.headers = {"Location": "/orders/43"}
        mock_client.place_order.return_value = resp
        bridge, fake, _ = self._setup(monkeypatch, mock_client)
        bridge.place_order("MSFT", 3, "SELL")
        fake.orders.equities.equity_sell_market.assert_called_once_with("MSFT", 3)

    # --- get_quotes -----------------------------------------------------------

    def test_get_quotes_returns_dict(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.Account.Fields.POSITIONS = "positions"
        mock_client.get_quotes.return_value = _ok_response({
            "AAPL": {
                "quote": {
                    "bidPrice": 149.5,
                    "askPrice": 150.0,
                    "lastPrice": 149.8,
                    "totalVolume": 5000000.0,
                }
            }
        })
        bridge, _, _ = self._setup(monkeypatch, mock_client)
        result = bridge.get_quotes(["AAPL"])
        assert result["AAPL"]["bid"] == 149.5
        assert result["AAPL"]["ask"] == 150.0
        assert result["AAPL"]["last"] == 149.8
        assert result["AAPL"]["volume"] == 5000000.0

    def test_get_quotes_empty_list_returns_empty_dict(self, monkeypatch):
        bridge, _, _ = self._setup(monkeypatch)
        assert bridge.get_quotes([]) == {}
