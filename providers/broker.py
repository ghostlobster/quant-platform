from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class BrokerProvider(Protocol):
    def get_account_info(self) -> dict: ...
    def get_positions(self) -> list[dict]: ...
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict: ...
    def cancel_order(self, order_id: str) -> bool: ...
    def get_orders(self, status: str = "open") -> list[dict]: ...


def get_broker(provider: Optional[str] = None) -> BrokerProvider:
    name = (provider or os.environ.get("BROKER_PROVIDER", "paper")).lower()
    if name == "alpaca":
        from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter
        return AlpacaBrokerAdapter()
    elif name == "ibkr":
        from adapters.broker.ibkr_adapter import IBKRAdapter
        return IBKRAdapter()
    elif name == "schwab":
        from adapters.broker.schwab_adapter import SchwabAdapter
        return SchwabAdapter()
    elif name == "paper":
        from adapters.broker.paper_adapter import PaperBrokerAdapter
        return PaperBrokerAdapter()
    raise ValueError(f"Unknown broker provider: {name!r}. Valid: alpaca, ibkr, schwab, paper")
