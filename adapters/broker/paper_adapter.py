"""
adapters/broker/paper_adapter.py — BrokerProvider wrapping broker/paper_trader.py.

No external dependency.  Uses the existing SQLite-backed paper trading engine.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from risk.pretrade_guard import GuardLimits, GuardViolation, PreTradeGuard

logger = logging.getLogger(__name__)

# Thread-safe singleton init guard
_init_lock = threading.Lock()
_initialised = False


def _ensure_init() -> None:
    global _initialised
    if _initialised:
        return
    with _init_lock:
        if not _initialised:
            from broker.paper_trader import init_paper_tables
            init_paper_tables()
            _initialised = True


class PaperBrokerAdapter:
    """BrokerProvider backed by the in-memory/SQLite paper trading engine."""

    def __init__(self) -> None:
        _ensure_init()
        self._orders: dict[str, dict] = {}  # local order store (in-memory)
        self._order_counter = 0
        self._lock = threading.Lock()
        self._guard = PreTradeGuard(GuardLimits.from_env(), self)

    def get_account_info(self) -> dict:
        from broker.paper_trader import get_account
        return get_account()

    def get_positions(self) -> list[dict]:
        from broker.paper_trader import get_portfolio
        df = get_portfolio()
        if df.empty:
            return []
        records = []
        for _, row in df.iterrows():
            records.append({
                "symbol": row["Ticker"],
                "qty": float(row["Shares"]),
                "avg_entry_price": float(row["Avg Cost"]),
                "market_value": float(row["Market Value"]),
                "unrealized_pl": row.get("Unrealised P&L"),
            })
        return records

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

        from broker.paper_trader import buy, sell
        # For paper trading we need a price — use limit_price or fetch live price.
        price = limit_price
        if price is None:
            try:
                from data.fetcher import fetch_latest_price
                info = fetch_latest_price(symbol)
                price = info.get("price") or 100.0
            except Exception:
                price = 100.0  # absolute fallback

        with self._lock:
            self._order_counter += 1
            order_id = f"paper-{self._order_counter:06d}"

        try:
            if side.lower() == "buy":
                fill = buy(symbol, qty, price)
            elif side.lower() == "sell":
                fill = sell(symbol, qty, price)
            else:
                raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        except Exception as exc:
            logger.error("Paper order failed: %s", exc)
            return {
                "order_id": order_id,
                "symbol": symbol.upper(),
                "qty": qty,
                "side": side,
                "order_type": order_type,
                "status": "rejected",
                "error": str(exc),
            }

        result = {
            "order_id": order_id,
            "symbol": symbol.upper(),
            "qty": qty,
            "side": side,
            "order_type": "market",
            "status": "filled",
            "filled_avg_price": price,
            **fill,
        }
        with self._lock:
            self._orders[order_id] = result
        return result

    def place_bracket(self, intent) -> dict:
        """Fill the parent leg and record the TP/SL/trail children."""
        try:
            self._guard.check(
                intent.symbol, intent.qty, intent.side, intent.limit_price,
            )
        except GuardViolation as violation:
            logger.warning(
                "pretrade_guard_reject symbol=%s qty=%s side=%s reason=%s",
                intent.symbol, intent.qty, intent.side, violation.reason,
            )
            return {
                "symbol": intent.symbol.upper(),
                "qty": intent.qty,
                "side": intent.side,
                "status": "rejected",
                "reason": violation.reason,
            }

        price = intent.limit_price
        if price is None:
            try:
                from data.fetcher import fetch_latest_price
                info = fetch_latest_price(intent.symbol)
                price = info.get("price") or 100.0
            except Exception:
                price = 100.0

        from broker.paper_trader import place_bracket as _place
        return _place(
            intent.symbol,
            intent.qty,
            intent.side,
            entry_price=price,
            take_profit=intent.take_profit,
            stop_loss=intent.stop_loss,
            trail_percent=intent.trail_percent,
        )

    def cancel_order(self, order_id: str) -> bool:
        # Paper orders fill immediately — nothing to cancel.
        logger.debug("PaperBrokerAdapter.cancel_order: paper orders fill instantly; no-op")
        return False

    def get_orders(self, status: str = "open") -> list[dict]:
        with self._lock:
            orders = list(self._orders.values())
        if status == "all":
            return orders
        return [o for o in orders if o.get("status") == status]
