"""
Tradier API bridge — paper/sandbox and live trading, with options support.
Requires TRADIER_API_KEY and TRADIER_ACCOUNT_ID in .env.
Set TRADIER_SANDBOX=true (default) for sandbox; false for live.
"""
import os
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Sentinels — set to None so credentials are NOT read at import time.
# Tests may patch these directly; None means "read from os.getenv lazily".
TRADIER_API_KEY = None
TRADIER_ACCOUNT_ID = None
TRADIER_SANDBOX = None
BASE_URL = None

_VALID_EQUITY_SIDES = ("buy", "sell")
_VALID_OPTION_SIDES = ("buy_to_open", "sell_to_close", "buy_to_close", "sell_to_open")


def _get_config() -> dict:
    api_key = TRADIER_API_KEY if TRADIER_API_KEY is not None else os.getenv("TRADIER_API_KEY", "")
    account_id = TRADIER_ACCOUNT_ID if TRADIER_ACCOUNT_ID is not None else os.getenv("TRADIER_ACCOUNT_ID", "")
    if TRADIER_SANDBOX is not None:
        sandbox = TRADIER_SANDBOX
    else:
        sandbox = os.getenv("TRADIER_SANDBOX", "true").lower() == "true"
    if BASE_URL is not None:
        base_url = BASE_URL
    else:
        base_url = (
            "https://sandbox.tradier.com/v1"
            if sandbox else
            "https://api.tradier.com/v1"
        )
    return {"api_key": api_key, "account_id": account_id, "base_url": base_url}


def _get_headers() -> dict:
    cfg = _get_config()
    return {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Accept": "application/json",
    }


def _is_configured() -> bool:
    cfg = _get_config()
    return bool(cfg["api_key"] and cfg["account_id"])


def get_account() -> Optional[dict]:
    """Fetch account balances. Returns {'cash', 'equity', 'buying_power'} or None."""
    if not _is_configured():
        logger.warning("Tradier credentials not configured")
        return None
    try:
        import requests
        cfg = _get_config()
        r = requests.get(
            f"{cfg['base_url']}/accounts/{cfg['account_id']}/balances",
            headers=_get_headers(),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        balances = data.get("balances", {})
        return {
            "cash": float(balances.get("cash", {}).get("cash_available", 0.0)),
            "equity": float(balances.get("total_equity", 0.0)),
            "buying_power": float(balances.get("margin", {}).get("stock_buying_power", 0.0)),
        }
    except Exception as e:
        logger.error("Tradier get_account failed: %s", type(e).__name__)
        return None


def get_positions() -> list[dict]:
    """Return list of current positions as {'ticker', 'qty', 'avg_entry', 'unrealised_pnl'}."""
    if not _is_configured():
        return []
    try:
        import requests
        cfg = _get_config()
        r = requests.get(
            f"{cfg['base_url']}/accounts/{cfg['account_id']}/positions",
            headers=_get_headers(),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("positions", {})
        if not raw or raw == "null":
            return []
        positions = raw.get("position", [])
        if isinstance(positions, dict):
            positions = [positions]
        return [
            {
                "ticker": p.get("symbol", ""),
                "qty": float(p.get("quantity", 0)),
                "avg_entry": float(p.get("cost_basis", 0.0)) / max(float(p.get("quantity", 1)), 1),
                "unrealised_pnl": 0.0,  # Tradier positions endpoint doesn't return live P&L
            }
            for p in positions
        ]
    except Exception as e:
        logger.error("Tradier get_positions failed: %s", type(e).__name__)
        return []


def place_market_order(ticker: str, qty: int, side: str) -> Optional[dict]:
    """
    Place a market equity order. side = 'buy' | 'sell'.
    Validates qty and side before checking credentials.
    Returns order dict on success, None on failure or unconfigured.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    if side not in _VALID_EQUITY_SIDES:
        raise ValueError(f"side must be one of {_VALID_EQUITY_SIDES}, got {side!r}")
    if not _is_configured():
        logger.warning("Tradier not configured — order not placed")
        return None
    try:
        import requests
        cfg = _get_config()
        payload = {
            "class": "equity",
            "symbol": ticker.upper(),
            "side": side,
            "quantity": str(qty),
            "type": "market",
            "duration": "day",
        }
        r = requests.post(
            f"{cfg['base_url']}/accounts/{cfg['account_id']}/orders",
            headers=_get_headers(),
            data=payload,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        order_data = data.get("order", data)
        return {
            "order_id": str(order_data.get("id", "")),
            "status": order_data.get("status", "unknown"),
            "ticker": ticker.upper(),
            "qty": qty,
            "side": side,
        }
    except Exception as e:
        logger.error("Tradier place_market_order failed: %s", type(e).__name__)
        return None


def cancel_all_orders() -> None:
    """Cancel all open orders."""
    if not _is_configured():
        logger.warning("Tradier not configured — cancel_all_orders skipped")
        return
    try:
        import requests
        cfg = _get_config()
        r = requests.get(
            f"{cfg['base_url']}/accounts/{cfg['account_id']}/orders",
            headers=_get_headers(),
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("orders", {})
        if not raw or raw == "null":
            return
        orders = raw.get("order", [])
        if isinstance(orders, dict):
            orders = [orders]
        for order in orders:
            if order.get("status") in ("open", "partially_filled"):
                order_id = order.get("id")
                requests.delete(
                    f"{cfg['base_url']}/accounts/{cfg['account_id']}/orders/{order_id}",
                    headers=_get_headers(),
                    timeout=10,
                )
    except Exception as e:
        logger.error("Tradier cancel_all_orders failed: %s", type(e).__name__)


def is_market_open() -> bool:
    """Check if the equity market is currently open."""
    if not _is_configured():
        return False
    try:
        import requests
        cfg = _get_config()
        r = requests.get(
            f"{cfg['base_url']}/markets/clock",
            headers=_get_headers(),
            timeout=10,
        )
        r.raise_for_status()
        state = r.json().get("clock", {}).get("state", "")
        return state == "open"
    except Exception as e:
        logger.error("Tradier is_market_open failed: %s", type(e).__name__)
        return False


# ---------------------------------------------------------------------------
# Options — Tradier's main differentiator
# ---------------------------------------------------------------------------

def get_options_chain(ticker: str, expiration: str = None) -> list[dict]:
    """
    Fetch the options chain for a ticker.
    expiration: 'YYYY-MM-DD', or None to use the nearest expiration.
    Returns list of {'symbol', 'strike', 'expiration', 'option_type',
                     'bid', 'ask', 'volume', 'open_interest', 'last'}.
    """
    if not _is_configured():
        return []
    try:
        import requests
        cfg = _get_config()
        if expiration is None:
            expirations = get_expirations(ticker)
            if not expirations:
                return []
            expiration = expirations[0]
        params = {"symbol": ticker.upper(), "expiration": expiration}
        r = requests.get(
            f"{cfg['base_url']}/markets/options/chains",
            headers=_get_headers(),
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("options", {})
        if not raw or raw == "null":
            return []
        options = raw.get("option", [])
        if isinstance(options, dict):
            options = [options]
        return [
            {
                "symbol": o.get("symbol", ""),
                "strike": float(o.get("strike", 0.0)),
                "expiration": o.get("expiration_date", ""),
                "option_type": o.get("option_type", ""),
                "bid": float(o.get("bid", 0.0)),
                "ask": float(o.get("ask", 0.0)),
                "volume": int(o.get("volume", 0)),
                "open_interest": int(o.get("open_interest", 0)),
                "last": float(o.get("last", 0.0) or 0.0),
            }
            for o in options
        ]
    except Exception as e:
        logger.error("Tradier get_options_chain failed: %s", type(e).__name__)
        return []


def get_expirations(ticker: str) -> list[str]:
    """
    Return list of available expiration date strings 'YYYY-MM-DD' for the ticker.
    """
    if not _is_configured():
        return []
    try:
        import requests
        cfg = _get_config()
        r = requests.get(
            f"{cfg['base_url']}/markets/options/expirations",
            headers=_get_headers(),
            params={"symbol": ticker.upper()},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("expirations", {})
        if not raw or raw == "null":
            return []
        dates = raw.get("date", [])
        if isinstance(dates, str):
            dates = [dates]
        return dates
    except Exception as e:
        logger.error("Tradier get_expirations failed: %s", type(e).__name__)
        return []


def place_option_order(
    option_symbol: str,
    qty: int,
    side: str,
    order_type: str = "market",
) -> Optional[dict]:
    """
    Place an options order.
    side: 'buy_to_open' | 'sell_to_close' | 'buy_to_close' | 'sell_to_open'
    Validates qty and side before checking credentials.
    Returns order dict on success, None on failure or unconfigured.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    if side not in _VALID_OPTION_SIDES:
        raise ValueError(f"side must be one of {_VALID_OPTION_SIDES}, got {side!r}")
    if not _is_configured():
        logger.warning("Tradier not configured — option order not placed")
        return None
    try:
        import requests
        cfg = _get_config()
        payload = {
            "class": "option",
            "symbol": option_symbol,
            "side": side,
            "quantity": str(qty),
            "type": order_type,
            "duration": "day",
        }
        r = requests.post(
            f"{cfg['base_url']}/accounts/{cfg['account_id']}/orders",
            headers=_get_headers(),
            data=payload,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        order_data = data.get("order", data)
        return {
            "order_id": str(order_data.get("id", "")),
            "status": order_data.get("status", "unknown"),
            "option_symbol": option_symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
        }
    except Exception as e:
        logger.error("Tradier place_option_order failed: %s", type(e).__name__)
        return None
