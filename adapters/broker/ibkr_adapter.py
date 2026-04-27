"""
adapters/broker/ibkr_adapter.py — BrokerProvider for Interactive Brokers.

Backed by :mod:`broker.ibkr_bridge` (ib_insync). P1.8 (#146) extended the
bridge with a multi-asset contract factory; this adapter routes each
order through :func:`data.symbols.resolve` so every ticker carries the
right ``asset_class`` / ``exchange`` / ``currency`` / ``expiry`` /
``multiplier`` when it reaches the bridge.

Pre-trade guard (#139) wraps every order — paper and live alike. The
adapter is a no-op when ``ib_insync`` is missing: ``get_*`` methods
return empty values, ``place_*`` returns ``{"status": "failed"}``.

ENV vars (forwarded to broker.ibkr_bridge)
------------------------------------------
    IBKR_HOST       TWS/Gateway host (default 127.0.0.1)
    IBKR_PORT       override port (defaults derived from IBKR_PAPER)
    IBKR_CLIENT_ID  client ID (default 1)
    IBKR_PAPER      true → 7497 (paper), false → 7496 (live)
"""
from __future__ import annotations

import logging
from typing import Optional

import broker.ibkr_bridge as _bridge
from data.symbols import AssetClass, SymbolMeta, resolve
from risk.pretrade_guard import GuardLimits, GuardViolation, PreTradeGuard

logger = logging.getLogger(__name__)


class IBKRAdapter:
    """BrokerProvider backed by broker.ibkr_bridge (ib_insync)."""

    def __init__(self) -> None:
        if not _bridge._IB_AVAILABLE:
            raise ImportError(
                "ib_insync package is required for IBKRAdapter. "
                "Install it with: pip install ib_insync",
            )
        self._guard = PreTradeGuard(GuardLimits.from_env(), self)

    # ── account state ────────────────────────────────────────────────────────

    def get_account_info(self) -> dict:
        return _bridge.get_account_info() or {}

    def get_positions(self) -> list[dict]:
        raw = _bridge.get_positions() or []
        out: list[dict] = []
        for pos in raw:
            out.append(
                {
                    "symbol": pos.get("ticker", ""),
                    "qty": float(pos.get("qty", 0)),
                    "avg_entry_price": float(pos.get("avg_cost", 0)),
                    "market_value": float(pos.get("market_value", 0)),
                }
            )
        return out

    # ── orders ──────────────────────────────────────────────────────────────

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
                "symbol": symbol.upper(), "qty": qty, "side": side,
                "order_type": order_type,
                "status": "rejected", "reason": violation.reason,
            }

        meta = resolve(symbol, fallback_class=AssetClass.STOCK)
        return self._dispatch(meta, qty, side, order_type, limit_price)

    def place_bracket(self, intent) -> dict:
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
                "symbol": intent.symbol.upper(), "qty": intent.qty,
                "side": intent.side,
                "status": "rejected", "reason": violation.reason,
            }
        raise NotImplementedError(
            "IBKRAdapter.place_bracket: bracket support is Phase 2 — "
            "use place_order plus a child stop/limit for now",
        )

    def cancel_order(self, order_id: str) -> bool:
        return _bridge.cancel_order(order_id)

    def get_orders(self, status: str = "open") -> list[dict]:
        # ib_insync's openTrades() is the closest match; no canonical wrapper
        # exists in broker/ibkr_bridge yet, so the adapter returns an empty
        # list rather than guess.
        return []

    # ── internal ────────────────────────────────────────────────────────────

    def _dispatch(
        self,
        meta: SymbolMeta,
        qty: float,
        side: str,
        order_type: str,
        limit_price: Optional[float],
    ) -> dict:
        bridge_order_type = "LMT" if order_type.lower() == "limit" else "MKT"
        bridge_side = "BUY" if side.lower() == "buy" else "SELL"
        result = _bridge.place_order(
            meta.ticker,
            qty=qty,
            side=bridge_side,
            order_type=bridge_order_type,
            limit_price=limit_price,
            asset_class=meta.asset_class,
            exchange=meta.exchange,
            currency=meta.currency,
            expiry=meta.expiry,
            multiplier=meta.multiplier,
        )
        if not result:
            return {
                "status": "failed",
                "symbol": meta.ticker,
                "asset_class": meta.asset_class,
                "qty": qty,
                "side": side,
            }
        return {
            "order_id": result.get("order_id", ""),
            "symbol": meta.ticker,
            "asset_class": meta.asset_class,
            "exchange": meta.exchange,
            "currency": meta.currency,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "status": result.get("status", "unknown"),
        }
