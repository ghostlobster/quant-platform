import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
from broker.alpaca_bridge import (
    get_account, place_market_order, AlpacaOrder,
    _is_configured, cancel_all_orders
)


def test_not_configured_returns_none():
    """Without credentials, get_account returns None."""
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", ""):
        with patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", ""):
            assert get_account() is None


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
    with patch("broker.alpaca_bridge.ALPACA_API_KEY", "key"):
        with patch("broker.alpaca_bridge.ALPACA_SECRET_KEY", "secret"):
            with patch("requests.post", return_value=mock_response):
                order = place_market_order("AAPL", 10, "buy")
    assert order is not None
    assert order.__class__.__name__ == "AlpacaOrder"
    assert order.ticker == "AAPL"
    assert order.status == "accepted"
