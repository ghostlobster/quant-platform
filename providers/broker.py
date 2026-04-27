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


_LIVE_PROVIDERS = ("alpaca", "ibkr", "schwab")


class LivePromotionRefused(RuntimeError):
    """Raised by ``get_broker`` when the paper→live promotion guard refuses.

    The exception message names the failing precondition so operators can
    fix the missing env var or extend the paper track record before
    flipping ``BROKER_PROVIDER`` to a real venue.
    """


def _live_promotion_check(name: str) -> None:
    """Refuse to instantiate a live broker without explicit confirmation
    AND a minimum paper track record.

    Two preconditions must both hold:

    * ``LIVE_TRADING_CONFIRMED=true`` — explicit operator opt-in. Set
      after reviewing the paper track record + risk limits.
    * Journal shows ≥ ``LIVE_PROMOTION_MIN_DAYS`` (default 30) of distinct
      paper-trading days with realised PnL Sharpe ≥
      ``LIVE_PROMOTION_MIN_SHARPE`` (default 0.5).

    The Sharpe gate is a coarse "did the paper run produce something?"
    check, not a guarantee of profitability. Operators are still
    responsible for the strategy review.

    Bypassed entirely when ``LIVE_PROMOTION_BYPASS=1`` — for emergency
    re-enablement after a kill-switch event. Use sparingly; the bypass
    is logged at error level so audit trails surface it.
    """
    if name not in _LIVE_PROVIDERS:
        return
    if os.environ.get("LIVE_PROMOTION_BYPASS", "").strip().lower() in ("1", "true", "yes"):
        import structlog
        log = structlog.get_logger(__name__)
        log.error(
            "live_promotion: bypass active — skipping confirmation + track-record gate",
            provider=name,
        )
        return

    confirmed = os.environ.get("LIVE_TRADING_CONFIRMED", "").strip().lower() in (
        "1", "true", "yes",
    )
    if not confirmed:
        raise LivePromotionRefused(
            f"refusing to instantiate live broker {name!r}: "
            "LIVE_TRADING_CONFIRMED is not set. Set it to 'true' only "
            "after reviewing the paper track record + risk limits.",
        )

    min_days = int(os.environ.get("LIVE_PROMOTION_MIN_DAYS", "30"))
    min_sharpe = float(os.environ.get("LIVE_PROMOTION_MIN_SHARPE", "0.5"))
    days, sharpe = _paper_track_record()
    if days < min_days:
        raise LivePromotionRefused(
            f"refusing to instantiate live broker {name!r}: paper journal "
            f"only has {days} distinct trading days, need ≥ {min_days}. "
            "Continue paper trading or override with LIVE_PROMOTION_MIN_DAYS.",
        )
    if sharpe < min_sharpe:
        raise LivePromotionRefused(
            f"refusing to instantiate live broker {name!r}: paper Sharpe "
            f"{sharpe:.2f} < required {min_sharpe:.2f}. Iterate on the "
            "strategy or relax LIVE_PROMOTION_MIN_SHARPE if intentional.",
        )


def _paper_track_record() -> tuple[int, float]:
    """Return ``(distinct_paper_days, daily_pnl_sharpe)`` from the journal.

    Sharpe = mean(daily PnL %) / std(daily PnL %) over the realised exits
    in ``journal_trades.db``. Returns ``(0, 0.0)`` if the journal is
    empty / unreadable so a fresh install never accidentally passes.
    """
    try:
        from journal.trading_journal import get_journal
    except Exception:
        return 0, 0.0
    try:
        df = get_journal()
    except Exception:
        return 0, 0.0
    if df is None or df.empty or "pnl" not in df.columns:
        return 0, 0.0
    closed = df.dropna(subset=["pnl", "exit_time"])
    if closed.empty:
        return 0, 0.0

    import pandas as pd  # local import — keeps the import cost off cold paths

    days = pd.to_datetime(closed["exit_time"]).dt.date
    pnl_per_day = closed.groupby(days)["pnl"].sum()
    if len(pnl_per_day) == 0:
        return 0, 0.0
    if len(pnl_per_day) == 1 or pnl_per_day.std() == 0:
        return int(len(pnl_per_day)), 0.0
    sharpe = float(pnl_per_day.mean() / pnl_per_day.std())
    return int(len(pnl_per_day)), sharpe


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
    LivePromotionRefused
        If the resolved provider is a live venue (alpaca / ibkr / schwab)
        and the operator has not satisfied the P1.11 promotion gate
        (``LIVE_TRADING_CONFIRMED`` + paper track record).
    """
    name = (provider or os.environ.get("BROKER_PROVIDER", "paper")).lower().strip()
    _live_promotion_check(name)
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


def is_live_mode(provider: Optional[str] = None) -> bool:
    """Return ``True`` when the configured broker is a live venue.

    Used by the Streamlit sidebar banner to render the live/paper indicator
    in the right colour.
    """
    name = (provider or os.environ.get("BROKER_PROVIDER", "paper")).lower().strip()
    return name in _LIVE_PROVIDERS
