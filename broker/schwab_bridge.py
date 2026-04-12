"""
Charles Schwab broker bridge via schwab-py.
Configure via environment variables:
  SCHWAB_APP_KEY        — OAuth app key
  SCHWAB_APP_SECRET     — OAuth app secret
  SCHWAB_CALLBACK_URL   — OAuth callback URL (default: https://127.0.0.1)
  SCHWAB_TOKEN_PATH     — path to stored token file (default: .schwab_token.json)
  SCHWAB_ACCOUNT_HASH   — hashed account ID from Schwab API
  SCHWAB_SANDBOX        — sandbox mode guard flag (default: true)
Safe no-op when credentials are absent, token file is missing, or schwab-py is not installed.
"""
import logging
import os

logger = logging.getLogger(__name__)

try:
    import schwab as _schwab
    _SCHWAB_AVAILABLE = True
except ImportError:
    _schwab = None
    _SCHWAB_AVAILABLE = False


def _get_config() -> dict:
    return {
        "app_key": os.getenv("SCHWAB_APP_KEY", ""),
        "app_secret": os.getenv("SCHWAB_APP_SECRET", ""),
        "callback_url": os.getenv("SCHWAB_CALLBACK_URL", "https://127.0.0.1"),
        "token_path": os.getenv("SCHWAB_TOKEN_PATH", ".schwab_token.json"),
        "account_hash": os.getenv("SCHWAB_ACCOUNT_HASH", ""),
        "sandbox": os.getenv("SCHWAB_SANDBOX", "true").lower() == "true",
    }


def _is_configured() -> bool:
    cfg = _get_config()
    return bool(cfg["app_key"] and cfg["app_secret"] and cfg["account_hash"])


def _get_client():
    """Return an authenticated Schwab client from the token file."""
    cfg = _get_config()
    return _schwab.auth.client_from_token_file(
        cfg["token_path"],
        cfg["app_key"],
        cfg["app_secret"],
    )


def get_account_info() -> dict:
    """
    Return account summary: cash, equity, margin.
    Returns empty dict if schwab-py unavailable, credentials missing,
    token file absent, or request fails.
    """
    if not _SCHWAB_AVAILABLE:
        logger.warning("schwab-py not installed — get_account_info returning {}")
        return {}
    if not _is_configured():
        logger.warning("Schwab credentials not configured — get_account_info returning {}")
        return {}
    try:
        cfg = _get_config()
        client = _get_client()
        resp = client.get_account(
            cfg["account_hash"],
            fields=[client.Account.Fields.POSITIONS],
        )
        resp.raise_for_status()
        data = resp.json()
        balances = data.get("securitiesAccount", {}).get("currentBalances", {})
        return {
            "cash": float(balances.get("cashBalance", 0.0)),
            "equity": float(balances.get("liquidationValue", 0.0)),
            "margin": float(balances.get("maintenanceRequirement", 0.0)),
        }
    except FileNotFoundError:
        logger.warning("Schwab token file not found — get_account_info returning {}")
        return {}
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}


def get_positions() -> list[dict]:
    """
    Return current positions.
    Returns: [{'ticker': str, 'qty': float, 'avg_cost': float, 'market_value': float}]
    Returns empty list if unavailable or request fails.
    """
    if not _SCHWAB_AVAILABLE:
        logger.warning("schwab-py not installed — get_positions returning []")
        return []
    if not _is_configured():
        logger.warning("Schwab credentials not configured — get_positions returning []")
        return []
    try:
        cfg = _get_config()
        client = _get_client()
        resp = client.get_account(
            cfg["account_hash"],
            fields=[client.Account.Fields.POSITIONS],
        )
        resp.raise_for_status()
        data = resp.json()
        raw_positions = (
            data.get("securitiesAccount", {}).get("positions", [])
        )
        result = []
        for pos in raw_positions:
            instrument = pos.get("instrument", {})
            result.append({
                "ticker": instrument.get("symbol", ""),
                "qty": float(pos.get("longQuantity", 0.0)),
                "avg_cost": float(pos.get("averagePrice", 0.0)),
                "market_value": float(pos.get("marketValue", 0.0)),
            })
        return result
    except FileNotFoundError:
        logger.warning("Schwab token file not found — get_positions returning []")
        return []
    except Exception as e:
        logger.warning(type(e).__name__)
        return []


def place_order(
    ticker: str,
    qty: float,
    side: str,
    order_type: str = "MARKET",
    limit_price: float = None,
) -> dict:
    """
    Place an order.
    side: 'BUY' or 'SELL'
    order_type: 'MARKET' or 'LIMIT'
    Returns {'order_id': str, 'status': str} on success, empty dict on failure.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")

    if not _SCHWAB_AVAILABLE:
        logger.warning("schwab-py not installed — order not placed")
        return {}
    if not _is_configured():
        logger.warning("Schwab credentials not configured — order not placed")
        return {}
    try:
        cfg = _get_config()
        client = _get_client()
        ot = order_type.upper()
        if ot == "LIMIT" and limit_price is not None:
            if side == "BUY":
                order_spec = _schwab.orders.equities.equity_buy_limit(
                    ticker.upper(), qty, limit_price
                )
            else:
                order_spec = _schwab.orders.equities.equity_sell_limit(
                    ticker.upper(), qty, limit_price
                )
        else:
            if side == "BUY":
                order_spec = _schwab.orders.equities.equity_buy_market(
                    ticker.upper(), qty
                )
            else:
                order_spec = _schwab.orders.equities.equity_sell_market(
                    ticker.upper(), qty
                )
        resp = client.place_order(cfg["account_hash"], order_spec)
        resp.raise_for_status()
        order_id = resp.headers.get("Location", "").split("/")[-1]
        return {"order_id": order_id, "status": "ACCEPTED"}
    except FileNotFoundError:
        logger.warning("Schwab token file not found — order not placed")
        return {}
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}


def get_quotes(tickers: list[str]) -> dict[str, dict]:
    """
    Return current quotes for a list of tickers.
    Returns: {ticker: {'bid': float, 'ask': float, 'last': float, 'volume': float}}
    Returns empty dict if unavailable or request fails.
    """
    if not _SCHWAB_AVAILABLE:
        logger.warning("schwab-py not installed — get_quotes returning {}")
        return {}
    if not _is_configured():
        logger.warning("Schwab credentials not configured — get_quotes returning {}")
        return {}
    if not tickers:
        return {}
    try:
        client = _get_client()
        resp = client.get_quotes(tickers)
        resp.raise_for_status()
        data = resp.json()
        result = {}
        for symbol, info in data.items():
            quote = info.get("quote", {})
            result[symbol] = {
                "bid": float(quote.get("bidPrice", 0.0)),
                "ask": float(quote.get("askPrice", 0.0)),
                "last": float(quote.get("lastPrice", 0.0)),
                "volume": float(quote.get("totalVolume", 0.0)),
            }
        return result
    except FileNotFoundError:
        logger.warning("Schwab token file not found — get_quotes returning {}")
        return {}
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}
