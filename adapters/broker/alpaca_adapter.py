"""
adapters/broker/alpaca_adapter.py — BrokerProvider wrapping broker/alpaca_bridge.py.

No new vendor dependency — delegates to the existing alpaca_bridge module.
"""
from __future__ import annotations

import logging
from typing import Optional

import broker.alpaca_bridge as _bridge
from risk.pretrade_guard import GuardLimits, GuardViolation, PreTradeGuard

logger = logging.getLogger(__name__)


class AlpacaBrokerAdapter:
    """BrokerProvider backed by the existing Alpaca bridge."""

    def __init__(self) -> None:
        self._guard = PreTradeGuard(GuardLimits.from_env(), self)

    def get_account_info(self) -> dict:
        result = _bridge.get_account()
        if result is None:
            return {}
        return result

    def get_positions(self) -> list[dict]:
        raw = _bridge.get_positions()
        out = []
        for p in raw:
            out.append({
                "symbol": p.get("symbol", ""),
                "qty": float(p.get("qty", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "market_value": float(p.get("market_value", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
            })
        return out

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        try:
            self._guard.check(symbol, qty, side, limit_price)
        except GuardViolation as violation:
            logger.warning(
                "pretrade_guard_reject symbol=%s qty=%s side=%s reason=%s",
                symbol, qty, side, violation.reason,
            )
            return {
                "symbol": symbol.upper(),
                "qty": qty,
                "side": side,
                "order_type": order_type,
                "status": "rejected",
                "reason": violation.reason,
            }

        if order_type != "market":
            logger.warning(
                "AlpacaBrokerAdapter: only market orders supported; "
                "ignoring order_type=%r and limit_price=%r",
                order_type, limit_price,
            )
        order = _bridge.place_market_order(symbol, qty, side)
        if order is None:
            return {"status": "failed", "symbol": symbol, "qty": qty, "side": side}
        return {
            "order_id": order.order_id,
            "symbol": order.ticker,
            "qty": order.qty,
            "side": order.side,
            "order_type": order.order_type,
            "status": order.status,
            "filled_avg_price": order.filled_avg_price,
        }

    def cancel_order(self, order_id: str) -> bool:
        # alpaca_bridge only exposes cancel_all; delegate and warn.
        logger.warning(
            "AlpacaBrokerAdapter.cancel_order: cancelling ALL orders "
            "(single-order cancel not exposed in alpaca_bridge)"
        )
        return _bridge.cancel_all_orders()

    def get_orders(self, status: str = "open") -> list[dict]:
        # alpaca_bridge does not expose get_orders; return empty list.
        logger.debug("AlpacaBrokerAdapter.get_orders not implemented in bridge; returning []")
        return []
