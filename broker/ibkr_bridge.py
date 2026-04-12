"""
Interactive Brokers bridge via ib_insync.
Configure via environment variables:
  IBKR_HOST       — TWS/Gateway host (default: 127.0.0.1)
  IBKR_PORT       — override port (defaults based on IBKR_PAPER)
  IBKR_CLIENT_ID  — client ID (default: 1)
  IBKR_PAPER      — true → port 7497 (paper), false → port 7496 (live); default: true
Safe no-op when credentials are absent or ib_insync is not installed.
"""
import os
import logging

logger = logging.getLogger(__name__)

try:
    import ib_insync as _ib_insync
    _IB_AVAILABLE = True
except ImportError:
    _ib_insync = None
    _IB_AVAILABLE = False


def _get_config() -> dict:
    paper = os.getenv("IBKR_PAPER", "true").lower() == "true"
    default_port = 7497 if paper else 7496
    return {
        "host": os.getenv("IBKR_HOST", "127.0.0.1"),
        "port": int(os.getenv("IBKR_PORT", str(default_port))),
        "client_id": int(os.getenv("IBKR_CLIENT_ID", "1")),
        "paper": paper,
    }


def _connect():
    """Return a connected IB instance."""
    cfg = _get_config()
    ib = _ib_insync.IB()
    ib.connect(cfg["host"], cfg["port"], clientId=cfg["client_id"])
    return ib


def get_account_info() -> dict:
    """
    Return account summary: cash, equity, margin.
    Returns empty dict if ib_insync unavailable or connection fails.
    """
    if not _IB_AVAILABLE:
        logger.warning("ib_insync not installed — get_account_info returning {}")
        return {}
    try:
        ib = _connect()
        try:
            summary = ib.accountSummary()
            def _val(tag):
                entry = next((v for v in summary if v.tag == tag), None)
                return float(entry.value) if entry else 0.0
            return {
                "cash": _val("TotalCashValue"),
                "equity": _val("NetLiquidation"),
                "margin": _val("MaintMarginReq"),
            }
        finally:
            ib.disconnect()
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}


def get_positions() -> list[dict]:
    """
    Return current positions.
    Returns: [{'ticker': str, 'qty': float, 'avg_cost': float, 'market_value': float}]
    Returns empty list if ib_insync unavailable or connection fails.
    """
    if not _IB_AVAILABLE:
        logger.warning("ib_insync not installed — get_positions returning []")
        return []
    try:
        ib = _connect()
        try:
            raw = ib.positions()
            result = []
            for pos in raw:
                result.append({
                    "ticker": pos.contract.symbol,
                    "qty": float(pos.position),
                    "avg_cost": float(pos.avgCost),
                    "market_value": float(pos.position) * float(pos.avgCost),
                })
            return result
        finally:
            ib.disconnect()
    except Exception as e:
        logger.warning(type(e).__name__)
        return []


def place_order(
    ticker: str,
    qty: float,
    side: str,
    order_type: str = "MKT",
    limit_price: float = None,
) -> dict:
    """
    Place an order.
    side: 'BUY' or 'SELL'
    order_type: 'MKT' or 'LMT'
    Returns {'order_id': str, 'status': str} on success, empty dict on failure.
    """
    if qty <= 0:
        raise ValueError(f"qty must be positive, got {qty}")
    side = side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"side must be 'BUY' or 'SELL', got {side!r}")

    if not _IB_AVAILABLE:
        logger.warning("ib_insync not installed — order not placed")
        return {}
    try:
        ib = _connect()
        try:
            contract = _ib_insync.Stock(ticker.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)
            if order_type.upper() == "LMT" and limit_price is not None:
                order = _ib_insync.LimitOrder(side, qty, limit_price)
            else:
                order = _ib_insync.MarketOrder(side, qty)
            trade = ib.placeOrder(contract, order)
            return {
                "order_id": str(trade.order.orderId),
                "status": trade.orderStatus.status,
            }
        finally:
            ib.disconnect()
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}


def cancel_order(order_id: str) -> bool:
    """
    Cancel an open order by order ID.
    Returns True on success, False on failure.
    """
    if not _IB_AVAILABLE:
        logger.warning("ib_insync not installed — cancel_order returning False")
        return False
    try:
        ib = _connect()
        try:
            open_trades = ib.openTrades()
            for trade in open_trades:
                if str(trade.order.orderId) == str(order_id):
                    ib.cancelOrder(trade.order)
                    return True
            logger.warning("cancel_order: order %s not found in open trades", order_id)
            return False
        finally:
            ib.disconnect()
    except Exception as e:
        logger.warning(type(e).__name__)
        return False


def get_market_data(ticker: str) -> dict:
    """
    Return current market data for a ticker.
    Returns: {'bid': float, 'ask': float, 'last': float, 'volume': float}
    Returns empty dict if ib_insync unavailable or connection fails.
    """
    if not _IB_AVAILABLE:
        logger.warning("ib_insync not installed — get_market_data returning {}")
        return {}
    try:
        ib = _connect()
        try:
            contract = _ib_insync.Stock(ticker.upper(), "SMART", "USD")
            ib.qualifyContracts(contract)
            tick = ib.reqMktData(contract)
            ib.sleep(1)
            return {
                "bid": float(tick.bid) if tick.bid == tick.bid else 0.0,
                "ask": float(tick.ask) if tick.ask == tick.ask else 0.0,
                "last": float(tick.last) if tick.last == tick.last else 0.0,
                "volume": float(tick.volume) if tick.volume == tick.volume else 0.0,
            }
        finally:
            ib.disconnect()
    except Exception as e:
        logger.warning(type(e).__name__)
        return {}
