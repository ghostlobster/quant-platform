import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch

from broker.alpaca_bridge import (
    cancel_all_orders,
    get_account,
    get_positions,
    is_market_open,
    place_market_order,
)


def test_not_configured_returns_none():
    """Without credentials, get_account returns None."""
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""):
        with patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
            assert get_account() is None


def test_get_account_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"status": "ACTIVE", "cash": "10000"}
    mock_resp.raise_for_status = MagicMock()
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", return_value=mock_resp):
        result = get_account()
    assert result == {"status": "ACTIVE", "cash": "10000"}


def test_get_account_exception_returns_none():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", side_effect=ConnectionError("timeout")):
        result = get_account()
    assert result is None


def test_get_positions_not_configured_returns_empty():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
        assert get_positions() == []


def test_get_positions_success():
    mock_resp = MagicMock()
    mock_resp.json.return_value = [{"symbol": "AAPL", "qty": "5"}]
    mock_resp.raise_for_status = MagicMock()
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", return_value=mock_resp):
        result = get_positions()
    assert len(result) == 1
    assert result[0]["symbol"] == "AAPL"


def test_get_positions_exception_returns_empty():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", side_effect=RuntimeError("network")):
        result = get_positions()
    assert result == []


def test_place_order_bad_qty_raises():
    with pytest.raises(ValueError):
        place_market_order("AAPL", qty=-1, side="buy")


def test_place_order_bad_side_raises():
    with pytest.raises(ValueError):
        place_market_order("AAPL", qty=1, side="hold")


def test_place_order_not_configured_returns_none():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""):
        with patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
            result = place_market_order("AAPL", 1, "buy")
    assert result is None


def test_place_order_mocked_success():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "abc-123", "status": "accepted",
        "filled_avg_price": None
    }
    mock_response.raise_for_status = MagicMock()
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.post", return_value=mock_response):
        order = place_market_order("AAPL", 10, "buy")
    assert order is not None
    assert order.__class__.__name__ == "AlpacaOrder"
    assert order.ticker == "AAPL"
    assert order.status == "accepted"


def test_place_order_exception_returns_none():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.post", side_effect=ConnectionError("fail")):
        result = place_market_order("AAPL", 1, "buy")
    assert result is None


def test_cancel_all_orders_not_configured():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
        assert cancel_all_orders() is False


def test_cancel_all_orders_success_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 207
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.delete", return_value=mock_resp):
        assert cancel_all_orders() is True


def test_cancel_all_orders_exception():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.delete", side_effect=OSError("net")):
        assert cancel_all_orders() is False


def test_is_market_open_not_configured():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
        assert is_market_open() is False


def test_is_market_open_true():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"is_open": True}
    mock_resp.raise_for_status = MagicMock()
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", return_value=mock_resp):
        assert is_market_open() is True


def test_is_market_open_exception():
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"), \
         patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"), \
         patch("requests.get", side_effect=TimeoutError("timeout")):
        assert is_market_open() is False
