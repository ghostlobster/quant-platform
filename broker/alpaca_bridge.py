"""
Alpaca Markets API bridge — paper and live trading.
Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env
Set ALPACA_PAPER=true for paper trading (default).
"""
import os
from dataclasses import dataclass
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

# Sentinels — set to None so credentials are NOT read at import time.
# Tests may patch these directly; None means "read from os.getenv lazily".
ALPACA_API_KEY = None
ALPACA_SECRET_KEY = None


@dataclass
class AlpacaOrder:
    order_id: str
    ticker: str
    qty: float
    side: str        # 'buy' or 'sell'
    order_type: str  # 'market', 'limit'
    status: str
    filled_avg_price: Optional[float] = None


def _get_config() -> dict:
    api_key = ALPACA_API_KEY if ALPACA_API_KEY is not None else os.getenv("ALPACA_API_KEY", "")
    secret_key = ALPACA_SECRET_KEY if ALPACA_SECRET_KEY is not None else os.getenv("ALPACA_SECRET_KEY", "")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
    base_url = (
        "https://paper-api.alpaca.markets"
        if paper else
        "https://api.alpaca.markets"
    )
    return {"api_key": api_key, "secret_key": secret_key, "base_url": base_url}


def _get_headers() -> dict:
    cfg = _get_config()
    return {
        "APCA-API-KEY-ID": cfg["api_key"],
        "APCA-API-SECRET-KEY": cfg["secret_key"],
        "Content-Type": "application/json",
    }


def _is_configured() -> bool:
    cfg = _get_config()
    return bool(cfg["api_key"] and cfg["secret_key"])


def get_account() -> Optional[dict]:
    """Fetch account info. Returns None if not configured or request fails."""
    if not _is_configured():
        logger.warning("Alpaca credentials not configured")
        return None
    try:
        import requests
        cfg = _get_config()
        r = requests.get(f"{cfg['base_url']}/v2/account", headers=_get_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Alpaca get_account failed: %s", type(e).__name__)
        return None


def get_positions() -> list:
    """Return list of current positions."""
    if not _is_configured():
        return []
    try:
        import requests
        cfg = _get_config()
        r = requests.get(f"{cfg['base_url']}/v2/positions", headers=_get_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error("Alpaca get_positions failed: %s", type(e).__name__)
        return []


def place_market_order(ticker: str, qty: float, side: str) -> Optional[AlpacaOrder]:
    """
    Place a market order. side = 'buy' or 'sell'.
    Returns AlpacaOrder on success, None on failure.
    Safety: will not execute if ALPACA_API_KEY is not set.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side}")
    if not _is_configured():
        logger.warning("Alpaca not configured — order not placed")
        return None
    try:
        import requests
        cfg = _get_config()
        payload = {
            "symbol": ticker.upper(),
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        r = requests.post(f"{cfg['base_url']}/v2/orders", headers=_get_headers(),
                          json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return AlpacaOrder(
            order_id=data.get("id", ""),
            ticker=ticker,
            qty=qty,
            side=side,
            order_type="market",
            status=data.get("status", "unknown"),
            filled_avg_price=data.get("filled_avg_price"),
        )
    except Exception as e:
        logger.error("Alpaca place_order failed: %s", type(e).__name__)
        return None


def cancel_all_orders() -> bool:
    """Cancel all open orders. Returns True on success."""
    if not _is_configured():
        return False
    try:
        import requests
        cfg = _get_config()
        r = requests.delete(f"{cfg['base_url']}/v2/orders", headers=_get_headers(), timeout=10)
        return r.status_code in (200, 204, 207)
    except Exception as e:
        logger.error("Alpaca cancel_all_orders failed: %s", type(e).__name__)
        return False


def is_market_open() -> bool:
    """Check if the market is currently open."""
    if not _is_configured():
        return False
    try:
        import requests
        cfg = _get_config()
        r = requests.get(f"{cfg['base_url']}/v2/clock", headers=_get_headers(), timeout=10)
        r.raise_for_status()
        return r.json().get("is_open", False)
    except Exception as e:
        logger.error("Alpaca is_market_open failed: %s", type(e).__name__)
        return False
