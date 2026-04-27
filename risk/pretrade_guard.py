"""
risk/pretrade_guard.py — deterministic pre-trade risk gate + kill-switch.

Every broker adapter calls ``PreTradeGuard.check()`` before placing an
order. Any limit violation raises ``GuardViolation`` with a machine-readable
reason; adapters convert that into a structured
``{"status": "rejected", "reason": ...}`` response so cron flows stay
non-fatal.

Limits are optional — unset env vars mean "no restriction" on that
dimension. The operator kill-switch is a file (default ``.killswitch``) that
the adapter checks on every order and that a SIGTERM handler creates on
signal. Removing the file resumes trading.

ENV vars
--------
    MAX_POSITION_PCT         fraction of equity a single symbol may occupy
    MAX_DAILY_LOSS_PCT       fraction of equity; realised+unrealised PnL floor
    MAX_GROSS_EXPOSURE       fraction of equity; ``Σ|position value| / equity``
    MAX_ORDERS_PER_DAY       hard cap on accepted orders per UTC day
    SYMBOL_BLOCKLIST         comma-separated tickers (case-insensitive)
    KILLSWITCH_FILE          path to the kill-switch flag (default .killswitch)
"""
from __future__ import annotations

import os
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import structlog

if TYPE_CHECKING:
    from providers.broker import BrokerProvider

logger = structlog.get_logger(__name__)


class GuardViolation(Exception):
    """Raised when an order violates a pre-trade limit or the kill-switch."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _parse_float(env_name: str) -> float | None:
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("pretrade_guard: bad float env var", var=env_name, value=raw)
        return None


def _parse_int(env_name: str) -> int | None:
    raw = os.environ.get(env_name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("pretrade_guard: bad int env var", var=env_name, value=raw)
        return None


def _parse_blocklist(env_name: str) -> frozenset[str]:
    raw = os.environ.get(env_name, "")
    return frozenset(
        t.strip().upper() for t in raw.split(",") if t.strip()
    )


@dataclass(frozen=True)
class GuardLimits:
    """Pre-trade limit configuration. Unset fields mean no restriction."""

    max_position_pct: float | None = None
    max_daily_loss_pct: float | None = None
    max_gross_exposure: float | None = None
    max_orders_per_day: int | None = None
    symbol_blocklist: frozenset[str] = field(default_factory=frozenset)
    killswitch_path: Path = field(default_factory=lambda: Path(".killswitch"))

    @classmethod
    def from_env(cls) -> GuardLimits:
        """Build ``GuardLimits`` from the documented env vars."""
        killswitch_raw = os.environ.get("KILLSWITCH_FILE", ".killswitch")
        return cls(
            max_position_pct=_parse_float("MAX_POSITION_PCT"),
            max_daily_loss_pct=_parse_float("MAX_DAILY_LOSS_PCT"),
            max_gross_exposure=_parse_float("MAX_GROSS_EXPOSURE"),
            max_orders_per_day=_parse_int("MAX_ORDERS_PER_DAY"),
            symbol_blocklist=_parse_blocklist("SYMBOL_BLOCKLIST"),
            killswitch_path=Path(killswitch_raw),
        )

    def any_active(self) -> bool:
        """True when at least one dimension is configured."""
        return (
            self.max_position_pct is not None
            or self.max_daily_loss_pct is not None
            or self.max_gross_exposure is not None
            or self.max_orders_per_day is not None
            or bool(self.symbol_blocklist)
        )


class PreTradeGuard:
    """Stateful pre-trade gate bound to a specific ``BrokerProvider``."""

    def __init__(
        self,
        limits: GuardLimits,
        broker: "BrokerProvider",
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._limits = limits
        self._broker = broker
        self._clock = clock
        self._day_key: str | None = None
        self._orders_today: int = 0

    @property
    def limits(self) -> GuardLimits:
        return self._limits

    def check(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float | None = None,
    ) -> None:
        """Raise :class:`GuardViolation` if the order is rejected.

        Bumps the accepted-order counter on success so callers do not need
        to track it. A violation leaves the counter untouched.
        """
        symbol_u = symbol.upper()
        self._rollover_day()

        # Kill-switch comes first — fail closed.
        if self._limits.killswitch_path.exists():
            raise GuardViolation(
                "killswitch",
                f"{self._limits.killswitch_path} present",
            )

        if symbol_u in self._limits.symbol_blocklist:
            raise GuardViolation("symbol_blocklist", symbol_u)

        if (
            self._limits.max_orders_per_day is not None
            and self._orders_today >= self._limits.max_orders_per_day
        ):
            raise GuardViolation(
                "max_orders_per_day",
                f"{self._orders_today}/{self._limits.max_orders_per_day}",
            )

        # Remaining dimensions need account state.
        if self._needs_account_state():
            equity, positions = self._account_snapshot()
            if equity > 0:
                self._check_position_pct(symbol_u, qty, limit_price, equity, positions)
                self._check_gross_exposure(symbol_u, qty, side, limit_price, equity, positions)
                self._check_daily_loss(equity, positions)

        # Accept.
        self._orders_today += 1

    def _needs_account_state(self) -> bool:
        return (
            self._limits.max_position_pct is not None
            or self._limits.max_gross_exposure is not None
            or self._limits.max_daily_loss_pct is not None
        )

    def _account_snapshot(self) -> tuple[float, list[dict]]:
        try:
            account = self._broker.get_account_info() or {}
            positions = self._broker.get_positions() or []
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("pretrade_guard: account snapshot failed", error=str(exc))
            return 0.0, []

        # Equity key varies by adapter — Alpaca uses ``equity``, the paper
        # broker exposes ``total_value``, and a few legacy paths still
        # return ``portfolio_value``. Try all three so the guard's
        # dollar-sizing dimensions actually fire on every backend.
        equity_raw = (
            account.get("equity")
            or account.get("portfolio_value")
            or account.get("total_value")
            or 0.0
        )
        try:
            equity = float(equity_raw)
        except (TypeError, ValueError):
            equity = 0.0
        return equity, positions

    @staticmethod
    def _order_notional(qty: float, limit_price: float | None) -> float:
        if limit_price is None or limit_price <= 0:
            return 0.0
        return abs(float(qty) * float(limit_price))

    @staticmethod
    def _position_notional(pos: dict) -> float:
        market_value = pos.get("market_value")
        if market_value is not None:
            try:
                return abs(float(market_value))
            except (TypeError, ValueError):
                pass
        qty = pos.get("qty") or 0
        price = pos.get("avg_entry_price") or pos.get("current_price") or 0
        try:
            return abs(float(qty) * float(price))
        except (TypeError, ValueError):
            return 0.0

    def _check_position_pct(
        self,
        symbol: str,
        qty: float,
        limit_price: float | None,
        equity: float,
        positions: list[dict],
    ) -> None:
        if self._limits.max_position_pct is None:
            return
        order_notional = self._order_notional(qty, limit_price)
        if order_notional <= 0:
            return
        existing = 0.0
        for p in positions:
            if (p.get("symbol") or "").upper() == symbol:
                existing = self._position_notional(p)
                break
        post_weight = (existing + order_notional) / equity
        if post_weight > self._limits.max_position_pct:
            raise GuardViolation(
                "max_position_pct",
                f"weight={post_weight:.4f} cap={self._limits.max_position_pct:.4f}",
            )

    def _check_gross_exposure(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float | None,
        equity: float,
        positions: list[dict],
    ) -> None:
        if self._limits.max_gross_exposure is None:
            return
        order_notional = self._order_notional(qty, limit_price)
        if order_notional <= 0:
            return
        gross = sum(self._position_notional(p) for p in positions)
        projected = (gross + order_notional) / equity
        if projected > self._limits.max_gross_exposure:
            raise GuardViolation(
                "max_gross_exposure",
                f"gross={projected:.4f} cap={self._limits.max_gross_exposure:.4f}",
            )

    def _check_daily_loss(self, equity: float, positions: list[dict]) -> None:
        if self._limits.max_daily_loss_pct is None:
            return
        realised = _realised_pnl_today(self._clock)
        unrealised = 0.0
        for p in positions:
            raw = p.get("unrealized_pl") or p.get("unrealised_pl")
            if raw is None:
                continue
            try:
                unrealised += float(raw)
            except (TypeError, ValueError):
                continue
        pnl = realised + unrealised
        floor = -self._limits.max_daily_loss_pct * equity
        if pnl <= floor:
            raise GuardViolation(
                "max_daily_loss_pct",
                f"pnl={pnl:.2f} floor={floor:.2f}",
            )

    def _rollover_day(self) -> None:
        today = self._clock().strftime("%Y-%m-%d")
        if self._day_key != today:
            self._day_key = today
            self._orders_today = 0


def _realised_pnl_today(clock: Callable[[], datetime]) -> float:
    """Sum today's realised P&L from the trading journal (UTC day)."""
    try:
        from journal.trading_journal import get_journal
    except ImportError:  # pragma: no cover — defensive
        return 0.0
    today = clock().strftime("%Y-%m-%d")
    try:
        df = get_journal(start_date=today)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("pretrade_guard: journal read failed", error=str(exc))
        return 0.0
    if df is None or df.empty or "pnl" not in df.columns:
        return 0.0
    return float(df["pnl"].fillna(0.0).sum())


# ── Kill-switch SIGTERM handler ─────────────────────────────────────────────

def install_killswitch_handler(
    path: Path | str | None = None,
    signals: tuple[int, ...] = (signal.SIGTERM,),
) -> Path:
    """Install a SIGTERM handler that creates the kill-switch file.

    Operators remove the file manually to resume trading. The returned path
    is the one that will be created on signal.
    """
    resolved = Path(path) if path is not None else Path(
        os.environ.get("KILLSWITCH_FILE", ".killswitch")
    )

    def _handler(signum, frame):  # noqa: ARG001 — signal API
        try:
            resolved.touch(exist_ok=True)
            logger.error(
                "pretrade_guard: killswitch engaged via signal",
                signum=signum,
                path=str(resolved),
            )
        except OSError as exc:  # pragma: no cover — defensive
            logger.error(
                "pretrade_guard: failed to write killswitch",
                error=str(exc),
                path=str(resolved),
            )

    for sig in signals:
        signal.signal(sig, _handler)
    return resolved
