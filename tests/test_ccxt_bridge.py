"""
Tests for broker/ccxt_bridge.py

Strategy: mock the ccxt import at the module level so the tests never require
the real package to be installed.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers to (re)load ccxt_bridge with a fake ccxt module injected
# ---------------------------------------------------------------------------

def _make_fake_ccxt(exchange_id: str = "binance", exchange_instance=None):
    """Return a minimal fake ccxt module with one exchange class."""
    fake = types.ModuleType("ccxt")
    mock_instance = exchange_instance or MagicMock()
    exchange_class = MagicMock(return_value=mock_instance)
    setattr(fake, exchange_id, exchange_class)
    return fake, exchange_class, mock_instance


def _load_bridge(monkeypatch, *, ccxt_module=None, env=None):
    """
    Import (or re-import) ccxt_bridge with the given environment and
    optional fake ccxt module injected into sys.modules.
    Returns the freshly loaded module.
    """
    # Remove any previously loaded version
    sys.modules.pop("broker.ccxt_bridge", None)
    sys.modules.pop("ccxt", None)

    if ccxt_module is not None:
        sys.modules["ccxt"] = ccxt_module

    env = env or {}
    for key in ("CCXT_EXCHANGE", "CCXT_API_KEY", "CCXT_SECRET"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)

    import broker.ccxt_bridge as bridge
    return bridge


# ---------------------------------------------------------------------------
# Safe no-op: ccxt not installed
# ---------------------------------------------------------------------------

class TestCcxtNotInstalled:
    """When ccxt cannot be imported every function returns a safe empty value."""

    def _bridge(self, monkeypatch):
        # Ensure ccxt is NOT in sys.modules
        sys.modules.pop("ccxt", None)
        sys.modules.pop("broker.ccxt_bridge", None)

        # Patch the import inside the module by making importlib raise ImportError
        with patch.dict(sys.modules, {"ccxt": None}):
            return _load_bridge(
                monkeypatch,
                ccxt_module=None,
                env={"CCXT_API_KEY": "key", "CCXT_SECRET": "secret"},
            )

    def test_get_account_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.get_account() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.get_positions() == []

    def test_place_market_order_returns_empty_dict(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        result = bridge.place_market_order("BTC/USDT", 0.1, "buy")
        assert result == {}

    def test_cancel_all_orders_is_noop(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        bridge.cancel_all_orders()  # must not raise

    def test_fetch_ohlcv_returns_empty_dataframe(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        df = bridge.fetch_ohlcv("BTC/USDT")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_ticker_price_returns_zero(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.get_ticker_price("BTC/USDT") == 0.0

    def test_list_symbols_returns_empty_list(self, monkeypatch):
        bridge = self._bridge(monkeypatch)
        assert bridge.list_symbols() == []


# ---------------------------------------------------------------------------
# Safe no-op: ccxt available but credentials absent
# ---------------------------------------------------------------------------

class TestCredentialsAbsent:
    """When CCXT_API_KEY / CCXT_SECRET are not set, functions return safe values."""

    def _bridge(self, monkeypatch):
        fake_ccxt, _, _ = _make_fake_ccxt()
        return _load_bridge(monkeypatch, ccxt_module=fake_ccxt, env={})

    def test_get_account_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).get_account() == {}

    def test_get_positions_returns_empty_list(self, monkeypatch):
        assert self._bridge(monkeypatch).get_positions() == []

    def test_place_market_order_returns_empty_dict(self, monkeypatch):
        assert self._bridge(monkeypatch).place_market_order("BTC/USDT", 1.0, "buy") == {}

    def test_cancel_all_orders_is_noop(self, monkeypatch):
        self._bridge(monkeypatch).cancel_all_orders()  # must not raise

    def test_fetch_ohlcv_returns_empty_dataframe(self, monkeypatch):
        df = self._bridge(monkeypatch).fetch_ohlcv("BTC/USDT")
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_get_ticker_price_returns_zero(self, monkeypatch):
        assert self._bridge(monkeypatch).get_ticker_price("BTC/USDT") == 0.0


# ---------------------------------------------------------------------------
# Input validation — raises BEFORE credential check
# ---------------------------------------------------------------------------

class TestValidation:
    """qty and side validation must fire regardless of credential state."""

    def _bridge_no_creds(self, monkeypatch):
        fake_ccxt, _, _ = _make_fake_ccxt()
        return _load_bridge(monkeypatch, ccxt_module=fake_ccxt, env={})

    def test_qty_zero_raises_value_error(self, monkeypatch):
        bridge = self._bridge_no_creds(monkeypatch)
        with pytest.raises(ValueError, match="qty must be positive"):
            bridge.place_market_order("BTC/USDT", 0, "buy")

    def test_qty_negative_raises_value_error(self, monkeypatch):
        bridge = self._bridge_no_creds(monkeypatch)
        with pytest.raises(ValueError, match="qty must be positive"):
            bridge.place_market_order("BTC/USDT", -5.0, "sell")

    def test_invalid_side_raises_value_error(self, monkeypatch):
        bridge = self._bridge_no_creds(monkeypatch)
        with pytest.raises(ValueError, match="side must be"):
            bridge.place_market_order("BTC/USDT", 1.0, "long")

    def test_validation_fires_before_credential_check(self, monkeypatch):
        """Even with no credentials the ValueError is raised, not a silent no-op."""
        bridge = self._bridge_no_creds(monkeypatch)
        with pytest.raises(ValueError):
            bridge.place_market_order("BTC/USDT", -1.0, "buy")


# ---------------------------------------------------------------------------
# Happy-path: mock exchange object
# ---------------------------------------------------------------------------

CREDS_ENV = {"CCXT_API_KEY": "test_key", "CCXT_SECRET": "test_secret"}


class TestHappyPath:
    """Mock the ccxt exchange object and verify return-value shaping."""

    def _setup(self, monkeypatch, mock_instance=None):
        fake_ccxt, exchange_class, instance = _make_fake_ccxt(
            "binance", exchange_instance=mock_instance
        )
        bridge = _load_bridge(monkeypatch, ccxt_module=fake_ccxt, env=CREDS_ENV)
        return bridge, exchange_class, instance

    # --- get_account ----------------------------------------------------------

    def test_get_account_maps_usdt_balance(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_balance.return_value = {
            "total": {"USDT": 10000.0, "BTC": 0.5},
            "free":  {"USDT": 8000.0,  "BTC": 0.5},
        }
        bridge, _, _ = self._setup(monkeypatch, instance)
        result = bridge.get_account()
        assert result == {"cash": 8000.0, "equity": 10000.0, "buying_power": 8000.0}

    def test_get_account_missing_usdt_defaults_zero(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_balance.return_value = {"total": {}, "free": {}}
        bridge, _, _ = self._setup(monkeypatch, instance)
        result = bridge.get_account()
        assert result == {"cash": 0.0, "equity": 0.0, "buying_power": 0.0}

    # --- get_positions --------------------------------------------------------

    def test_get_positions_returns_nonzero_balances(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_balance.return_value = {
            "total": {"BTC": 1.5, "USDT": 500.0, "ETH": 0.0},
            "free":  {},
        }
        bridge, _, _ = self._setup(monkeypatch, instance)
        positions = bridge.get_positions()
        tickers = {p["ticker"] for p in positions}
        assert tickers == {"BTC", "USDT"}
        btc = next(p for p in positions if p["ticker"] == "BTC")
        assert btc["qty"] == 1.5

    def test_get_positions_all_zero_returns_empty(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_balance.return_value = {"total": {"BTC": 0.0}, "free": {}}
        bridge, _, _ = self._setup(monkeypatch, instance)
        assert bridge.get_positions() == []

    # --- place_market_order ---------------------------------------------------

    def test_place_market_order_calls_create_order(self, monkeypatch):
        instance = MagicMock()
        fake_order = {"id": "abc123", "status": "closed", "filled": 0.1}
        instance.create_order.return_value = fake_order
        bridge, _, _ = self._setup(monkeypatch, instance)

        result = bridge.place_market_order("BTC/USDT", 0.1, "buy")
        instance.create_order.assert_called_once_with("BTC/USDT", "market", "buy", 0.1)
        assert result == fake_order

    def test_place_market_order_sell_side(self, monkeypatch):
        instance = MagicMock()
        instance.create_order.return_value = {"id": "xyz", "status": "open"}
        bridge, _, _ = self._setup(monkeypatch, instance)

        bridge.place_market_order("ETH/USDT", 2.0, "sell")
        instance.create_order.assert_called_once_with("ETH/USDT", "market", "sell", 2.0)

    # --- fetch_ohlcv ----------------------------------------------------------

    def test_fetch_ohlcv_returns_dataframe_with_correct_columns(self, monkeypatch):
        instance = MagicMock()
        # ccxt returns list of [timestamp_ms, open, high, low, close, volume]
        raw = [
            [1_700_000_000_000, 30000.0, 31000.0, 29000.0, 30500.0, 100.0],
            [1_700_086_400_000, 30500.0, 32000.0, 30000.0, 31000.0, 120.0],
        ]
        instance.fetch_ohlcv.return_value = raw
        bridge, _, _ = self._setup(monkeypatch, instance)

        df = bridge.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=2)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(df) == 2
        assert df.index.tz is not None  # UTC timezone attached

    def test_fetch_ohlcv_passes_params_to_exchange(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_ohlcv.return_value = []
        bridge, _, _ = self._setup(monkeypatch, instance)

        bridge.fetch_ohlcv("ETH/USDT", timeframe="1h", limit=100)
        instance.fetch_ohlcv.assert_called_once_with("ETH/USDT", timeframe="1h", limit=100)

    # --- get_ticker_price -----------------------------------------------------

    def test_get_ticker_price_returns_last(self, monkeypatch):
        instance = MagicMock()
        instance.fetch_ticker.return_value = {"last": 42000.5, "bid": 42000.0}
        bridge, _, _ = self._setup(monkeypatch, instance)
        assert bridge.get_ticker_price("BTC/USDT") == 42000.5

    # --- cancel_all_orders ----------------------------------------------------

    def test_cancel_all_orders_no_symbol(self, monkeypatch):
        instance = MagicMock()
        bridge, _, _ = self._setup(monkeypatch, instance)
        bridge.cancel_all_orders()
        instance.cancel_all_orders.assert_called_once_with()

    def test_cancel_all_orders_with_symbol(self, monkeypatch):
        instance = MagicMock()
        bridge, _, _ = self._setup(monkeypatch, instance)
        bridge.cancel_all_orders("BTC/USDT")
        instance.cancel_all_orders.assert_called_once_with("BTC/USDT")

    # --- list_symbols ---------------------------------------------------------

    def test_list_symbols_returns_market_keys(self, monkeypatch):
        instance = MagicMock()
        instance.load_markets.return_value = {
            "BTC/USDT": {}, "ETH/USDT": {}, "SOL/USDT": {}
        }
        bridge, _, _ = self._setup(monkeypatch, instance)
        symbols = bridge.list_symbols()
        assert set(symbols) == {"BTC/USDT", "ETH/USDT", "SOL/USDT"}


# ---------------------------------------------------------------------------
# Exchange selection from CCXT_EXCHANGE env var
# ---------------------------------------------------------------------------

class TestExchangeSelection:
    def test_default_exchange_is_binance(self, monkeypatch):
        fake_ccxt, binance_class, binance_instance = _make_fake_ccxt("binance")
        binance_instance.fetch_balance.return_value = {"total": {}, "free": {}}
        bridge = _load_bridge(monkeypatch, ccxt_module=fake_ccxt, env=CREDS_ENV)
        bridge.get_account()
        binance_class.assert_called_once()

    def test_custom_exchange_env_var(self, monkeypatch):
        fake_ccxt, _, kraken_instance = _make_fake_ccxt("kraken")
        kraken_instance.fetch_balance.return_value = {"total": {}, "free": {}}
        bridge = _load_bridge(
            monkeypatch,
            ccxt_module=fake_ccxt,
            env={**CREDS_ENV, "CCXT_EXCHANGE": "kraken"},
        )
        bridge.get_account()
        kraken_class = getattr(fake_ccxt, "kraken")
        kraken_class.assert_called_once()
