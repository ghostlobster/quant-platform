"""
providers/broker.py — BrokerProvider protocol and factory.

ENV vars
--------
    BROKER_PROVIDER   alpaca | ibkr | schwab | paper  (default: paper)
    ALPACA_API_KEY, ALPACA_SECRET_KEY
    IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
    SCHWAB_API_KEY, SCHWAB_SECRET_KEY
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class BrokerProvider(Protocol):
    """Duck-typed interface for brokerage operations."""

    def get_account_info(self) -> dict:
        """Return cash, equity, buying power, etc."""
        ...

    def get_positions(self) -> list[dict]:
        """Return all open positions."""
        ...

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> dict:
        """
        Place an order.

        Parameters
        ----------
        symbol      : ticker, e.g. ``"AAPL"``
        qty         : number of shares (positive)
        side        : ``"buy"`` or ``"sell"``
        order_type  : ``"market"`` or ``"limit"``
        limit_price : required when order_type is ``"limit"``

        Returns
        -------
        dict with at least: order_id, status, symbol, qty, side
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID. Returns True on success."""
        ...

    def get_orders(self, status: str = "open") -> list[dict]:
        """Return orders filtered by *status* (open | closed | all)."""
        ...


def get_broker(provider: Optional[str] = None) -> BrokerProvider:
    """
    Return a configured BrokerProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the BROKER_PROVIDER env var.  One of:
        ``alpaca``, ``ibkr``, ``schwab``, ``paper``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    """
    name = (provider or os.environ.get("BROKER_PROVIDER", "paper")).lower().strip()
    if name == "alpaca":
        from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter
        return AlpacaBrokerAdapter()
    if name == "ibkr":
        from adapters.broker.ibkr_adapter import IBKRAdapter
        return IBKRAdapter()
    if name == "schwab":
        from adapters.broker.schwab_adapter import SchwabAdapter
        return SchwabAdapter()
    if name == "paper":
        from adapters.broker.paper_adapter import PaperBrokerAdapter
        return PaperBrokerAdapter()
    raise ValueError(
        f"Unknown broker provider: {name!r}. "
        "Valid options: alpaca, ibkr, schwab, paper"
    )
