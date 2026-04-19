"""
risk/metrics_exporter.py — Prometheus gauges for live portfolio risk.

Defines six gauges (equity, gross exposure, net exposure, daily P&L, VaR95,
drawdown) plus a pure ``compute_risk_snapshot`` that reads from the
configured broker / journal and an ``update_risk_gauges`` that pushes the
snapshot onto the gauges. ``scheduler/alerts.py`` schedules a 60s tick
loop that calls ``risk_exporter_job``; a Grafana dashboard under
``deploy/grafana/provisioning/dashboards/risk.json`` visualises the gauges.

Drawdown breaches (below ``-MAX_DRAWDOWN_PCT`` of equity) are emitted once
per ``RISK_ALERT_COOLDOWN`` seconds via ``alerts.channels.broadcast``.

ENV vars
--------
    MAX_DRAWDOWN_PCT       fraction; trigger drawdown alert below ``-value``
    RISK_ALERT_COOLDOWN    seconds between successive drawdown alerts
                            (default 3600)
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import structlog

logger = structlog.get_logger(__name__)


# ── Prometheus gauge registry ──────────────────────────────────────────────

try:
    import prometheus_client  # noqa: F401
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False


class _NoopMetric:
    """Fallback stand-in when prometheus-client is unavailable."""

    def labels(self, **_):
        return self

    def set(self, *_):
        pass

    def inc(self, *_):
        pass


def _make_gauge(name: str, doc: str):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:  # pragma: no cover — exercised only with prom-client installed
        from prometheus_client import Gauge
        return Gauge(name, doc)
    except Exception:
        return _NoopMetric()


RISK_EQUITY = _make_gauge(
    "quant_risk_equity",
    "Current portfolio equity (cash + market value of open positions) in dollars",
)
RISK_GROSS_EXPOSURE = _make_gauge(
    "quant_risk_gross_exposure",
    "Sum of |position market value| / equity",
)
RISK_NET_EXPOSURE = _make_gauge(
    "quant_risk_net_exposure",
    "Net position market value / equity (signed)",
)
RISK_DAILY_PNL = _make_gauge(
    "quant_risk_daily_pnl",
    "Today's realised + unrealised PnL in dollars (UTC day)",
)
RISK_VAR_95 = _make_gauge(
    "quant_risk_var_95",
    "Historical 95% 1-day VaR as a fraction of equity",
)
RISK_DRAWDOWN = _make_gauge(
    "quant_risk_drawdown",
    "Current drawdown from the running peak equity (fraction; negative in a drawdown)",
)


# ── Snapshot dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskSnapshot:
    equity: float
    gross_exposure: float
    net_exposure: float
    daily_pnl: float
    var_95: float
    drawdown: float


def _position_market_value(pos: dict) -> float:
    mv = pos.get("market_value")
    if mv is not None:
        try:
            return float(mv)
        except (TypeError, ValueError):
            pass
    qty = pos.get("qty") or 0
    price = pos.get("current_price") or pos.get("avg_entry_price") or 0
    try:
        return float(qty) * float(price)
    except (TypeError, ValueError):
        return 0.0


def _realised_pnl_today() -> float:
    """Sum today's realised P&L from the trading journal (UTC day)."""
    try:
        from journal.trading_journal import get_journal
    except ImportError:
        return 0.0
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        df = get_journal(start_date=today)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("risk_exporter: journal read failed", error=str(exc))
        return 0.0
    if df is None or df.empty or "pnl" not in df.columns:
        return 0.0
    return float(df["pnl"].fillna(0.0).sum())


def _portfolio_value_series() -> list[float]:
    """Return the portfolio NAV history from ``quant.db`` for drawdown / VaR."""
    try:
        from data.db import get_connection
    except ImportError:
        return []
    try:
        conn = get_connection()
        rows = conn.execute(
            "SELECT total_value FROM portfolio_history ORDER BY record_date ASC"
        ).fetchall()
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("risk_exporter: portfolio_history read failed", error=str(exc))
        return []
    return [float(r[0]) for r in rows if r and r[0] is not None]


def _current_drawdown(history: list[float], current_equity: float) -> float:
    """Drawdown relative to running peak. Zero when at a new high.

    Returned as a signed fraction (e.g. ``-0.12`` = 12% below peak).
    """
    if current_equity <= 0:
        return 0.0
    peak = max([*history, current_equity]) if history else current_equity
    if peak <= 0:
        return 0.0
    return (current_equity - peak) / peak


def _var_95(history: list[float]) -> float:
    """95% historical VaR from the portfolio-history series (positive fraction)."""
    if len(history) < 6:
        return 0.0
    try:
        from analysis.risk_metrics import _pct_returns, historical_var
    except ImportError:
        return 0.0
    returns = _pct_returns(history)
    return float(historical_var(returns, 0.95))


# ── Snapshot + update ──────────────────────────────────────────────────────

def compute_risk_snapshot(broker=None) -> RiskSnapshot:
    """Build a point-in-time :class:`RiskSnapshot` from broker + journal state.

    When ``broker`` is ``None`` the configured broker is resolved via
    :func:`providers.broker.get_broker`. Any provider failure collapses to a
    zero-filled snapshot so the scheduler tick never crashes.
    """
    if broker is None:
        try:
            from providers.broker import get_broker
            broker = get_broker()
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("risk_exporter: broker resolution failed", error=str(exc))
            return RiskSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    try:
        account = broker.get_account_info() or {}
        positions = broker.get_positions() or []
    except Exception as exc:
        logger.warning("risk_exporter: broker snapshot failed", error=str(exc))
        return RiskSnapshot(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    equity_raw = account.get("equity") or account.get("portfolio_value") or 0.0
    try:
        equity = float(equity_raw)
    except (TypeError, ValueError):
        equity = 0.0

    gross_notional = 0.0
    net_notional = 0.0
    unrealised = 0.0
    for pos in positions:
        mv = _position_market_value(pos)
        gross_notional += abs(mv)
        net_notional += mv
        raw = pos.get("unrealized_pl") or pos.get("unrealised_pl")
        if raw is not None:
            try:
                unrealised += float(raw)
            except (TypeError, ValueError):
                pass

    realised = _realised_pnl_today()
    daily_pnl = realised + unrealised

    gross_pct = (gross_notional / equity) if equity > 0 else 0.0
    net_pct = (net_notional / equity) if equity > 0 else 0.0

    history = _portfolio_value_series()
    drawdown = _current_drawdown(history, equity)
    var_95 = _var_95(history)

    return RiskSnapshot(
        equity=equity,
        gross_exposure=gross_pct,
        net_exposure=net_pct,
        daily_pnl=daily_pnl,
        var_95=var_95,
        drawdown=drawdown,
    )


def update_risk_gauges(snapshot: RiskSnapshot) -> None:
    """Push the snapshot onto the Prometheus gauges."""
    RISK_EQUITY.set(snapshot.equity)
    RISK_GROSS_EXPOSURE.set(snapshot.gross_exposure)
    RISK_NET_EXPOSURE.set(snapshot.net_exposure)
    RISK_DAILY_PNL.set(snapshot.daily_pnl)
    RISK_VAR_95.set(snapshot.var_95)
    RISK_DRAWDOWN.set(snapshot.drawdown)


# ── Drawdown breach alerting ───────────────────────────────────────────────

_last_drawdown_alert: dict[str, float] = {"ts": 0.0}


def _alert_cooldown_seconds() -> float:
    raw = os.environ.get("RISK_ALERT_COOLDOWN", "3600")
    try:
        return float(raw)
    except ValueError:
        return 3600.0


def _drawdown_threshold() -> float | None:
    raw = os.environ.get("MAX_DRAWDOWN_PCT")
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def maybe_alert_drawdown(
    snapshot: RiskSnapshot,
    now: float | None = None,
) -> bool:
    """Broadcast a drawdown alert when the threshold is breached.

    Returns ``True`` iff an alert was broadcast. Respects
    ``RISK_ALERT_COOLDOWN`` so we do not spam a channel on every tick.
    """
    threshold = _drawdown_threshold()
    if threshold is None:
        return False
    if snapshot.drawdown > -threshold:
        return False  # not breached

    now = time.time() if now is None else now
    cooldown = _alert_cooldown_seconds()
    last_ts = _last_drawdown_alert["ts"]
    if last_ts > 0.0 and (now - last_ts) < cooldown:
        return False

    subject = "Drawdown alert"
    body = (
        f"Portfolio drawdown {snapshot.drawdown:.2%} breached the "
        f"-{threshold:.2%} threshold (equity=${snapshot.equity:,.2f}, "
        f"daily_pnl=${snapshot.daily_pnl:,.2f})."
    )
    try:
        from alerts.channels import broadcast

        broadcast(subject, body)
    except Exception as exc:  # pragma: no cover — broadcast is best-effort
        logger.warning("risk_exporter: drawdown broadcast failed", error=str(exc))
        return False

    _last_drawdown_alert["ts"] = now
    logger.error(
        "risk_drawdown_alert",
        drawdown=snapshot.drawdown,
        threshold=-threshold,
        equity=snapshot.equity,
    )
    return True


def reset_alert_state() -> None:
    """Reset the module-level alert cooldown — intended for tests."""
    _last_drawdown_alert["ts"] = 0.0


# ── Job entry point ────────────────────────────────────────────────────────

def risk_exporter_job(broker=None) -> dict:
    """Compute, publish, and optionally alert on the latest risk snapshot.

    Intended for APScheduler (60s cadence). Returns the snapshot as a dict
    so the scheduler can log it in structured form and tests can assert
    on it.
    """
    snapshot = compute_risk_snapshot(broker)
    update_risk_gauges(snapshot)
    alerted = maybe_alert_drawdown(snapshot)
    payload = asdict(snapshot)
    payload["alerted"] = alerted
    logger.info("risk_snapshot", **payload)
    return payload
