"""
adapters/broker/ibkr_adapter.py — BrokerProvider stub for Interactive Brokers.

Requires:  pip install ibapi  (IBKR TWS API client)
ENV vars:  IBKR_HOST (default: 127.0.0.1)
           IBKR_PORT (default: 7497)
           IBKR_CLIENT_ID (default: 1)

This is a structural stub.  The ibapi library requires an event-loop-based
EClient/EWrapper pattern; a full async implementation is outside Phase 1 scope.
The stub raises NotImplementedError for all trading operations so that CI passes
and the import chain is valid.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import ibapi as _ibapi  # noqa: F401  (import check only)
    _IBAPI_AVAILABLE = True
except ImportError:
    _IBAPI_AVAILABLE = False


class IBKRAdapter:
    """BrokerProvider stub for Interactive Brokers TWS/Gateway."""

    def __init__(self) -> None:
        if not _IBAPI_AVAILABLE:
            raise ImportError(
                "ibapi package is required for IBKRAdapter. "
                "Install it with: pip install ibapi"
            )
        self._host = os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port = int(os.environ.get("IBKR_PORT", "7497"))
        self._client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))

    def get_account_info(self) -> dict:
        raise NotImplementedError("IBKRAdapter: full implementation pending Phase 2")

    def get_positions(self) -> list[dict]:
        raise NotImplementedError("IBKRAdapter: full implementation pending Phase 2")

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        raise NotImplementedError("IBKRAdapter: full implementation pending Phase 2")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("IBKRAdapter: full implementation pending Phase 2")

    def get_orders(self, status: str = "open") -> list[dict]:
        raise NotImplementedError("IBKRAdapter: full implementation pending Phase 2")
