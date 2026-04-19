"""
adapters/broker/schwab_adapter.py — BrokerProvider stub for Charles Schwab.

Requires:  pip install schwab-py
ENV vars:  SCHWAB_API_KEY, SCHWAB_SECRET_KEY, SCHWAB_ACCOUNT_ID
           SCHWAB_TOKEN_PATH  (default: ./schwab_token.json)

Structural stub — raises NotImplementedError for all trading operations.
Full implementation is Phase 2 scope.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from risk.pretrade_guard import GuardLimits, GuardViolation, PreTradeGuard

try:
    import schwab as _schwab  # noqa: F401  (import check only)
    _SCHWAB_AVAILABLE = True
except ImportError:
    _SCHWAB_AVAILABLE = False

logger = logging.getLogger(__name__)


class SchwabAdapter:
    """BrokerProvider stub for Charles Schwab."""

    def __init__(self) -> None:
        if not _SCHWAB_AVAILABLE:
            raise ImportError(
                "schwab-py package is required for SchwabAdapter. "
                "Install it with: pip install schwab-py"
            )
        self._api_key = os.environ.get("SCHWAB_API_KEY", "")
        self._secret_key = os.environ.get("SCHWAB_SECRET_KEY", "")
        self._account_id = os.environ.get("SCHWAB_ACCOUNT_ID", "")
        self._token_path = os.environ.get("SCHWAB_TOKEN_PATH", "./schwab_token.json")
        self._guard = PreTradeGuard(GuardLimits.from_env(), self)

    def get_account_info(self) -> dict:
        raise NotImplementedError("SchwabAdapter: full implementation pending Phase 2")

    def get_positions(self) -> list[dict]:
        raise NotImplementedError("SchwabAdapter: full implementation pending Phase 2")

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
        raise NotImplementedError("SchwabAdapter: full implementation pending Phase 2")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("SchwabAdapter: full implementation pending Phase 2")

    def get_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("SchwabAdapter: full implementation pending Phase 2")
