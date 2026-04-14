"""Tests for monitoring/metrics.py (Issue #30)."""
from __future__ import annotations


def test_module_imports_without_prometheus():
    """metrics.py should import cleanly even when prometheus-client is absent."""
    import importlib
    from unittest.mock import patch

    with patch.dict("sys.modules", {"prometheus_client": None}):
        import monitoring.metrics as m
        importlib.reload(m)
    # No exception = pass


def test_update_portfolio_metrics_does_not_raise():
    from monitoring.metrics import update_portfolio_metrics

    # Should be callable without raising regardless of whether prometheus is installed
    update_portfolio_metrics(nav=105_000.0, open_pnl=3_200.0)


def test_update_regime_metric_valid_regime():
    from monitoring.metrics import update_regime_metric

    for regime in ["trending_bull", "trending_bear", "mean_reverting", "high_vol"]:
        update_regime_metric(regime)  # Must not raise


def test_update_regime_metric_unknown_regime():
    from monitoring.metrics import update_regime_metric

    update_regime_metric("unknown_regime")  # Should be a no-op, not an error


def test_record_execution_latency():
    from monitoring.metrics import record_execution_latency

    record_execution_latency(0.35)  # Must not raise


def test_record_feed_latency():
    from monitoring.metrics import record_feed_latency

    record_feed_latency(2.5)  # Must not raise


def test_noop_metric_interface():
    """_NoopMetric must support all expected methods."""
    from monitoring.metrics import _NoopMetric

    noop = _NoopMetric()
    noop.set(100.0)
    noop.inc(1)
    noop.observe(0.5)
    noop.labels(job="test")
    with noop.time():
        pass


def test_portfolio_nav_and_open_pnl_attributes_exist():
    from monitoring import metrics

    assert hasattr(metrics, "PORTFOLIO_NAV")
    assert hasattr(metrics, "OPEN_PNL")
    assert hasattr(metrics, "SIGNAL_COUNT")
    assert hasattr(metrics, "EXECUTION_LATENCY")
    assert hasattr(metrics, "FEED_LATENCY")
    assert hasattr(metrics, "REGIME_LABEL")


def test_signal_count_inc_does_not_raise():
    from monitoring.metrics import SIGNAL_COUNT

    SIGNAL_COUNT.inc()
    SIGNAL_COUNT.inc(5)
