import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

import broker.tradier_bridge as tb
from broker.tradier_bridge import (
    cancel_all_orders,
    get_account,
    get_expirations,
    get_options_chain,
    get_positions,
    is_market_open,
    place_market_order,
    place_option_order,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_response(json_data, status_code=200):
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.raise_for_status = MagicMock()
    return m


def _configured():
    """Context managers that inject fake credentials."""
    return (
        patch.object(tb, "TRADIER_API_KEY", "fake-key"),
        patch.object(tb, "TRADIER_ACCOUNT_ID", "fake-account"),
    )


# ---------------------------------------------------------------------------
# No-op when credentials absent
# ---------------------------------------------------------------------------

class TestNoOpWithoutCredentials:
    def _no_creds(self):
        return (
            patch.object(tb, "TRADIER_API_KEY", ""),
            patch.object(tb, "TRADIER_ACCOUNT_ID", ""),
        )

    def test_get_account_returns_none(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert get_account() is None

    def test_get_positions_returns_empty(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert get_positions() == []

    def test_place_market_order_returns_none(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert place_market_order("AAPL", 1, "buy") is None

    def test_cancel_all_orders_no_op(self):
        with self._no_creds()[0], self._no_creds()[1]:
            cancel_all_orders()  # must not raise

    def test_is_market_open_returns_false(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert is_market_open() is False

    def test_get_options_chain_returns_empty(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert get_options_chain("AAPL") == []

    def test_get_expirations_returns_empty(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert get_expirations("AAPL") == []

    def test_place_option_order_returns_none(self):
        with self._no_creds()[0], self._no_creds()[1]:
            assert place_option_order("AAPL240119C00150000", 1, "buy_to_open") is None


# ---------------------------------------------------------------------------
# Validation errors raised BEFORE credential check
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_place_market_order_bad_qty_zero(self):
        with pytest.raises(ValueError, match="qty must be positive"):
            place_market_order("AAPL", 0, "buy")

    def test_place_market_order_bad_qty_negative(self):
        with pytest.raises(ValueError, match="qty must be positive"):
            place_market_order("AAPL", -5, "sell")

    def test_place_market_order_bad_side(self):
        with pytest.raises(ValueError, match="side must be one of"):
            place_market_order("AAPL", 1, "hold")

    def test_place_option_order_bad_qty(self):
        with pytest.raises(ValueError, match="qty must be positive"):
            place_option_order("AAPL240119C00150000", 0, "buy_to_open")

    def test_place_option_order_bad_side(self):
        with pytest.raises(ValueError, match="side must be one of"):
            place_option_order("AAPL240119C00150000", 1, "buy")

    def test_place_option_order_invalid_side_string(self):
        with pytest.raises(ValueError):
            place_option_order("AAPL240119C00150000", 2, "sell")

    def test_validation_fires_even_without_credentials(self):
        """Validation should raise before the credential check."""
        with patch.object(tb, "TRADIER_API_KEY", ""):
            with patch.object(tb, "TRADIER_ACCOUNT_ID", ""):
                with pytest.raises(ValueError):
                    place_market_order("AAPL", -1, "buy")


# ---------------------------------------------------------------------------
# Happy-path with mocked HTTP
# ---------------------------------------------------------------------------

class TestGetAccount:
    def test_parses_balances(self):
        payload = {
            "balances": {
                "cash": {"cash_available": 5000.0},
                "total_equity": 15000.0,
                "margin": {"stock_buying_power": 10000.0},
            }
        }
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            result = get_account()
        assert result == {"cash": 5000.0, "equity": 15000.0, "buying_power": 10000.0}


class TestGetPositions:
    def test_parses_single_position(self):
        payload = {
            "positions": {
                "position": {
                    "symbol": "AAPL",
                    "quantity": 10,
                    "cost_basis": 1750.0,
                }
            }
        }
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            positions = get_positions()
        assert len(positions) == 1
        assert positions[0]["ticker"] == "AAPL"
        assert positions[0]["qty"] == 10.0
        assert positions[0]["avg_entry"] == pytest.approx(175.0)

    def test_parses_multiple_positions(self):
        payload = {
            "positions": {
                "position": [
                    {"symbol": "AAPL", "quantity": 5, "cost_basis": 875.0},
                    {"symbol": "MSFT", "quantity": 2, "cost_basis": 800.0},
                ]
            }
        }
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            positions = get_positions()
        assert len(positions) == 2
        tickers = {p["ticker"] for p in positions}
        assert tickers == {"AAPL", "MSFT"}

    def test_empty_positions(self):
        payload = {"positions": "null"}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            assert get_positions() == []


class TestPlaceMarketOrder:
    def test_successful_order(self):
        payload = {"order": {"id": "999", "status": "ok"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.post", return_value=_mock_response(payload)):
            result = place_market_order("AAPL", 10, "buy")
        assert result is not None
        assert result["order_id"] == "999"
        assert result["ticker"] == "AAPL"
        assert result["qty"] == 10
        assert result["side"] == "buy"
        assert result["status"] == "ok"

    def test_ticker_uppercased(self):
        payload = {"order": {"id": "1", "status": "ok"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.post", return_value=_mock_response(payload)):
            result = place_market_order("aapl", 1, "sell")
        assert result["ticker"] == "AAPL"


class TestCancelAllOrders:
    def test_cancels_open_orders(self):
        get_payload = {
            "orders": {
                "order": [
                    {"id": "10", "status": "open"},
                    {"id": "11", "status": "filled"},
                    {"id": "12", "status": "partially_filled"},
                ]
            }
        }
        delete_response = _mock_response({}, 200)
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(get_payload)), \
             patch("requests.delete", return_value=delete_response) as mock_del:
            cancel_all_orders()
        assert mock_del.call_count == 2  # only open + partially_filled

    def test_no_open_orders(self):
        get_payload = {"orders": "null"}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(get_payload)), \
             patch("requests.delete") as mock_del:
            cancel_all_orders()
        mock_del.assert_not_called()


class TestIsMarketOpen:
    def test_open_state(self):
        payload = {"clock": {"state": "open"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            assert is_market_open() is True

    def test_closed_state(self):
        payload = {"clock": {"state": "closed"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            assert is_market_open() is False


class TestGetExpirations:
    def test_parses_list(self):
        payload = {"expirations": {"date": ["2024-01-19", "2024-02-16", "2024-03-15"]}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            result = get_expirations("AAPL")
        assert result == ["2024-01-19", "2024-02-16", "2024-03-15"]

    def test_single_date_string_coerced_to_list(self):
        payload = {"expirations": {"date": "2024-01-19"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            result = get_expirations("AAPL")
        assert result == ["2024-01-19"]


class TestGetOptionsChain:
    _OPTION = {
        "symbol": "AAPL240119C00150000",
        "strike": 150.0,
        "expiration_date": "2024-01-19",
        "option_type": "call",
        "bid": 2.50,
        "ask": 2.60,
        "volume": 1200,
        "open_interest": 5000,
        "last": 2.55,
    }

    def test_parses_chain(self):
        payload = {"options": {"option": [self._OPTION]}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            chain = get_options_chain("AAPL", expiration="2024-01-19")
        assert len(chain) == 1
        o = chain[0]
        assert o["symbol"] == "AAPL240119C00150000"
        assert o["strike"] == 150.0
        assert o["option_type"] == "call"
        assert o["bid"] == 2.50
        assert o["volume"] == 1200
        assert o["open_interest"] == 5000

    def test_no_expiration_fetches_nearest(self):
        expirations_payload = {"expirations": {"date": ["2024-01-19"]}}
        chain_payload = {"options": {"option": [self._OPTION]}}
        responses = [
            _mock_response(expirations_payload),
            _mock_response(chain_payload),
        ]
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", side_effect=responses):
            chain = get_options_chain("AAPL")
        assert len(chain) == 1

    def test_empty_chain(self):
        payload = {"options": "null"}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.get", return_value=_mock_response(payload)):
            assert get_options_chain("AAPL", expiration="2024-01-19") == []


class TestPlaceOptionOrder:
    def test_successful_option_order(self):
        payload = {"order": {"id": "777", "status": "ok"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.post", return_value=_mock_response(payload)):
            result = place_option_order("AAPL240119C00150000", 2, "buy_to_open")
        assert result is not None
        assert result["order_id"] == "777"
        assert result["option_symbol"] == "AAPL240119C00150000"
        assert result["qty"] == 2
        assert result["side"] == "buy_to_open"
        assert result["order_type"] == "market"

    def test_sell_to_close(self):
        payload = {"order": {"id": "888", "status": "ok"}}
        with patch.object(tb, "TRADIER_API_KEY", "k"), \
             patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
             patch("requests.post", return_value=_mock_response(payload)):
            result = place_option_order("AAPL240119C00150000", 1, "sell_to_close", "limit")
        assert result["side"] == "sell_to_close"
        assert result["order_type"] == "limit"


# ---------------------------------------------------------------------------
# Sandbox vs live URL selection
# ---------------------------------------------------------------------------

class TestUrlSelection:
    def test_sandbox_url(self):
        with patch.object(tb, "TRADIER_SANDBOX", True):
            # Re-evaluate BASE_URL by checking what get_account would call
            with patch.object(tb, "TRADIER_API_KEY", "k"), \
                 patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
                 patch("requests.get") as mock_get:
                mock_get.return_value = _mock_response({
                    "balances": {
                        "cash": {"cash_available": 0},
                        "total_equity": 0,
                        "margin": {"stock_buying_power": 0},
                    }
                })
                get_account()
            url_used = mock_get.call_args[0][0]
            assert "sandbox.tradier.com" in url_used

    def test_live_url(self):
        with patch.object(tb, "TRADIER_SANDBOX", False), \
             patch.object(tb, "BASE_URL", "https://api.tradier.com/v1"):
            with patch.object(tb, "TRADIER_API_KEY", "k"), \
                 patch.object(tb, "TRADIER_ACCOUNT_ID", "a"), \
                 patch("requests.get") as mock_get:
                mock_get.return_value = _mock_response({
                    "balances": {
                        "cash": {"cash_available": 0},
                        "total_equity": 0,
                        "margin": {"stock_buying_power": 0},
                    }
                })
                get_account()
            url_used = mock_get.call_args[0][0]
            assert "api.tradier.com" in url_used
            assert "sandbox" not in url_used
