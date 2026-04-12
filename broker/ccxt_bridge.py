"""
CCXT crypto exchange bridge — supports any ccxt-compatible exchange.
Configure via environment variables:
  CCXT_EXCHANGE  — exchange id, e.g. 'binance' (default: 'binance')
  CCXT_API_KEY   — API key
  CCXT_SECRET    — API secret
  CCXT_SANDBOX   — use sandbox/testnet mode (default: true)
Safe no-op when credentials are absent or ccxt is not installed.
"""
import os
import structlog

import pandas as pd

logger = structlog.get_logger(__name__)

try:
    import ccxt as _ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _ccxt = None
    _CCXT_AVAILABLE = False

def _get_config() -> dict:
    return {
        "api_key": os.getenv("CCXT_API_KEY", ""),
        "secret": os.getenv("CCXT_SECRET", ""),
        "exchange": os.getenv("CCXT_EXCHANGE", "binance"),
        "sandbox": os.getenv("CCXT_SANDBOX", "true").lower() == "true",
    }


def _is_configured() -> bool:
    cfg = _get_config()
    return bool(cfg["api_key"] and cfg["secret"])


def _get_exchange():
    """Instantiate and return the configured ccxt exchange object."""
    cfg = _get_config()
    exchange_class = getattr(_ccxt, cfg["exchange"])
    exchange = exchange_class({
        "apiKey": cfg["api_key"],
        "secret": cfg["secret"],
        "enableRateLimit": True,
    })
    if cfg["sandbox"]:
        try:
            exchange.set_sandbox_mode(True)
        except Exception:
            logger.warning("Exchange %s does not support sandbox mode — proceeding in live mode", cfg["exchange"])
    return exchange


def get_account() -> dict:
    """
    Return account summary.
    Returns: {'cash': float, 'equity': float, 'buying_power': float}
    Returns empty dict if not configured or ccxt unavailable.
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — get_account returning empty dict")
        return {}
    if not _is_configured():
        logger.warning("CCXT credentials not configured — get_account returning empty dict")
        return {}
    try:
        exchange = _get_exchange()
        balance = exchange.fetch_balance()
        total_usdt = float(balance.get("total", {}).get("USDT", 0.0) or 0.0)
        free_usdt  = float(balance.get("free",  {}).get("USDT", 0.0) or 0.0)
        return {
            "cash": free_usdt,
            "equity": total_usdt,
            "buying_power": free_usdt,
        }
    except Exception as e:
        logger.error("ccxt get_account failed: %s", type(e).__name__)
        return {}


def get_positions() -> list[dict]:
    """
    Return non-zero balances as position-like dicts.
    Returns: [{'ticker': str, 'qty': float, 'avg_entry': float, 'unrealised_pnl': float}]
    Returns empty list if not configured or ccxt unavailable.
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — get_positions returning []")
        return []
    if not _is_configured():
        logger.warning("CCXT credentials not configured — get_positions returning []")
        return []
    try:
        exchange = _get_exchange()
        balance = exchange.fetch_balance()
        positions = []
        totals = balance.get("total", {})
        for asset, qty in totals.items():
            qty = float(qty or 0.0)
            if qty > 0:
                positions.append({
                    "ticker": asset,
                    "qty": qty,
                    "avg_entry": 0.0,
                    "unrealised_pnl": 0.0,
                })
        return positions
    except Exception as e:
        logger.error("ccxt get_positions failed: %s", type(e).__name__)
        return []


def place_market_order(symbol: str, qty: float, side: str) -> dict:
    """
    Place a market order.
    symbol: ccxt unified format, e.g. 'BTC/USDT'
    side: 'buy' | 'sell'
    qty: must be > 0
    Raises ValueError for invalid qty or side before checking credentials.
    Returns order dict on success, empty dict on failure.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — order not placed")
        return {}
    if not _is_configured():
        logger.warning("CCXT credentials not configured — order not placed")
        return {}
    try:
        exchange = _get_exchange()
        order = exchange.create_order(symbol, "market", side, qty)
        return order
    except Exception as e:
        logger.error("ccxt place_market_order failed: %s", type(e).__name__)
        return {}


def cancel_all_orders(symbol: str = None) -> None:
    """
    Cancel all open orders, optionally filtered by symbol.
    No-op if not configured or ccxt unavailable.
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — cancel_all_orders is a no-op")
        return
    if not _is_configured():
        logger.warning("CCXT credentials not configured — cancel_all_orders is a no-op")
        return
    try:
        exchange = _get_exchange()
        if symbol:
            exchange.cancel_all_orders(symbol)
        else:
            exchange.cancel_all_orders()
    except Exception as e:
        logger.error("ccxt cancel_all_orders failed: %s", type(e).__name__)


def fetch_ohlcv(symbol: str, timeframe: str = "1d", limit: int = 365) -> pd.DataFrame:
    """
    Fetch OHLCV candles.
    Returns DataFrame with columns [open, high, low, close, volume]
    and a UTC DatetimeIndex.
    Returns empty DataFrame if not configured or ccxt unavailable.
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — fetch_ohlcv returning empty DataFrame")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not _is_configured():
        logger.warning("CCXT credentials not configured — fetch_ohlcv returning empty DataFrame")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    try:
        exchange = _get_exchange()
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.index.name = None
        df = df.drop(columns=["timestamp"])
        return df
    except Exception as e:
        logger.error("ccxt fetch_ohlcv failed: %s", type(e).__name__)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def get_ticker_price(symbol: str) -> float:
    """
    Return the current last price for a symbol.
    Returns 0.0 if not configured or ccxt unavailable.
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — get_ticker_price returning 0.0")
        return 0.0
    if not _is_configured():
        logger.warning("CCXT credentials not configured — get_ticker_price returning 0.0")
        return 0.0
    try:
        exchange = _get_exchange()
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker.get("last") or 0.0)
    except Exception as e:
        logger.error("ccxt get_ticker_price failed: %s", type(e).__name__)
        return 0.0


def list_symbols() -> list[str]:
    """
    Return all available trading pairs on the configured exchange.
    Returns empty list if ccxt unavailable (credentials not required).
    """
    if not _CCXT_AVAILABLE:
        logger.warning("ccxt not installed — list_symbols returning []")
        return []
    try:
        cfg = _get_config()
        exchange_class = getattr(_ccxt, cfg["exchange"])
        exchange = exchange_class({"enableRateLimit": True})
        markets = exchange.load_markets()
        return list(markets.keys())
    except Exception as e:
        logger.error("ccxt list_symbols failed: %s", type(e).__name__)
        return []
