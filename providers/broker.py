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
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class OrderIntent:
    """Declarative order payload supporting single-leg and bracket orders.

    ``take_profit`` / ``stop_loss`` / ``trail_percent`` are optional.
    At least one of them must be set for a bracket order; when all three
    are ``None`` the intent collapses to an ordinary market/limit order.
    """

    symbol: str
    qty: float
    side: str                       # "buy" | "sell"
    order_type: str = "market"      # "market" | "limit"
    limit_price: Optional[float] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    trail_percent: Optional[float] = None

    def is_bracket(self) -> bool:
        return (
            self.take_profit is not None
            or self.stop_loss is not None
            or self.trail_percent is not None
        )

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be > 0, got {self.qty}")
        if self.side.lower() not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {self.side!r}")
        if self.order_type.lower() not in ("market", "limit"):
            raise ValueError(
                f"order_type must be 'market' or 'limit', got {self.order_type!r}"
            )
        if self.order_type.lower() == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required when order_type='limit'")
        if self.trail_percent is not None and self.trail_percent <= 0:
            raise ValueError(
                f"trail_percent must be > 0 when set, got {self.trail_percent}"
            )


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

    def place_bracket(self, intent: OrderIntent) -> dict:
        """Place a bracket order — parent fill plus take-profit / stop-loss /
        trailing-stop children.

        Returns a dict with at least: ``order_id``, ``status``, ``symbol``,
        ``qty``, ``side`` plus ``children`` — a list of child-order dicts
        describing the pending TP/SL/trail legs.
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
