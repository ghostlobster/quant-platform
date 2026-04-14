"""
monitoring/metrics.py — Shared Prometheus metrics registry.

All Gauge/Counter/Histogram objects are defined at module level so they persist
across HTTP scrapes and can be updated from anywhere in the platform.

Requires (optional): pip install prometheus-client>=0.19.0

Usage
-----
    from monitoring.metrics import update_portfolio_metrics, SIGNAL_COUNT
    update_portfolio_metrics(nav=105_000, open_pnl=5_000)
    SIGNAL_COUNT.inc()
"""
from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    from prometheus_client import (  # type: ignore[import]
        Counter,
        Gauge,
        Histogram,
        REGISTRY,
    )
    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    logger.info(
        "prometheus-client not installed; metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )


def _make_gauge(name: str, doc: str, labels: list[str] | None = None):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        from prometheus_client import Gauge as _Gauge
        return _Gauge(name, doc, labels or [])
    except Exception:
        return _NoopMetric()


def _make_counter(name: str, doc: str):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        from prometheus_client import Counter as _Counter
        return _Counter(name, doc)
    except Exception:
        return _NoopMetric()


def _make_histogram(name: str, doc: str, buckets=None):
    if not _PROM_AVAILABLE:
        return _NoopMetric()
    try:
        from prometheus_client import Histogram as _Histogram
        kwargs = {"buckets": buckets} if buckets else {}
        return _Histogram(name, doc, **kwargs)
    except Exception:
        return _NoopMetric()


class _NoopMetric:
    """Fallback when prometheus-client is not installed."""
    def labels(self, **_): return self
    def set(self, *_): pass
    def inc(self, *_): pass
    def observe(self, *_): pass
    def time(self): return self
    def __enter__(self): return self
    def __exit__(self, *_): pass


# ── Metric definitions ─────────────────────────────────────────────────────────

PORTFOLIO_NAV = _make_gauge(
    "quant_portfolio_nav",
    "Current total portfolio net asset value in dollars",
)

OPEN_PNL = _make_gauge(
    "quant_open_pnl",
    "Sum of unrealised P&L across all open positions",
)

SIGNAL_COUNT = _make_counter(
    "quant_signal_count_total",
    "Total number of trading signals generated since process start",
)

EXECUTION_LATENCY = _make_histogram(
    "quant_execution_latency_seconds",
    "Time from decision to last fill, in seconds",
    buckets=[0.1, 0.5, 1.0, 5.0, 30.0, 60.0, 300.0],
)

FEED_LATENCY = _make_histogram(
    "quant_feed_latency_seconds",
    "Market data feed latency (time since last bar), in seconds",
    buckets=[1, 5, 15, 60, 300, 900],
)

REGIME_LABEL = _make_gauge(
    "quant_regime_label",
    "Current market regime encoded as numeric: 0=trending_bull 1=trending_bear "
    "2=mean_reverting 3=high_vol",
)

_REGIME_MAP = {
    "trending_bull": 0,
    "trending_bear": 1,
    "mean_reverting": 2,
    "high_vol": 3,
}


# ── Helper update functions ────────────────────────────────────────────────────

def update_portfolio_metrics(nav: float, open_pnl: float) -> None:
    """Push current portfolio NAV and open P&L to Prometheus gauges."""
    PORTFOLIO_NAV.set(nav)
    OPEN_PNL.set(open_pnl)


def update_regime_metric(regime: str) -> None:
    """Encode regime string as a numeric gauge value."""
    REGIME_LABEL.set(_REGIME_MAP.get(regime, -1))


def record_execution_latency(seconds: float) -> None:
    """Record how long an execution algo took from decision to last fill."""
    EXECUTION_LATENCY.observe(seconds)


def record_feed_latency(seconds: float) -> None:
    """Record market data feed lag."""
    FEED_LATENCY.observe(seconds)
