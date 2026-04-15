"""Tests for new scheduler functions: run_var_check, run_correlation_check, run_anomaly_checks."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure optional deps that data.fetcher needs are available as mocks
sys.modules.setdefault("yfinance", MagicMock())


def test_run_correlation_check_disabled_by_default():
    os.environ.pop("CORRELATION_MONITOR_ENABLED", None)
    from scheduler.alerts import run_correlation_check
    result = run_correlation_check(price_data={}, positions={"AAPL": 10_000})
    assert result == []


def test_run_correlation_check_no_positions_returns_empty(monkeypatch):
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")
    from scheduler.alerts import run_correlation_check
    result = run_correlation_check(price_data={}, positions={})
    assert result == []


def test_run_correlation_check_fires_alert_on_concentration(monkeypatch):
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")
    import numpy as np
    import pandas as pd

    from risk.correlation import CorrelationAlert

    mock_alert = CorrelationAlert(
        alert_type="position_concentration",
        value=0.80,
        threshold=0.25,
        message="AAPL is 80% of portfolio",
        ticker="AAPL",
    )

    rng = np.random.default_rng(42)
    price_data = {
        "AAPL": pd.Series(100 + rng.normal(0, 1, 30).cumsum()),
        "MSFT": pd.Series(100 + rng.normal(0, 1, 30).cumsum()),
    }
    positions = {"AAPL": 80_000.0, "MSFT": 20_000.0}

    with patch("risk.correlation.check_correlation_alerts", return_value=[mock_alert]), \
         patch("scheduler.alerts.broadcast"):
        from scheduler.alerts import run_correlation_check
        fired = run_correlation_check(price_data=price_data, positions=positions)

    assert len(fired) == 1
    assert fired[0]["alert_type"] == "position_concentration"


def test_run_correlation_check_no_alerts_returns_empty(monkeypatch):
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")

    with patch("risk.correlation.check_correlation_alerts", return_value=[]):
        from scheduler.alerts import run_correlation_check
        result = run_correlation_check(
            price_data={"AAPL": pd.Series([100.0] * 10)},
            positions={"AAPL": 10_000.0},
        )
    assert result == []


def test_run_anomaly_checks_empty_watchlist():
    from scheduler.alerts import run_anomaly_checks
    # Empty watchlist + empty prices = only signal drought check
    with patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=None):
        result = run_anomaly_checks(watchlist=[], current_prices={})
    assert isinstance(result, list)


def test_run_anomaly_checks_fires_on_drought():
    from analysis.anomaly_detector import Anomaly
    from scheduler.alerts import run_anomaly_checks

    drought = Anomaly(
        type="signal_drought",
        severity="warning",
        symbol="",
        message="No signals in last 4h",
    )

    with patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=drought), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_price_spike",
               return_value=None), \
         patch("scheduler.alerts.broadcast"):
        result = run_anomaly_checks(watchlist=["AAPL"], current_prices={"AAPL": 180.0})

    assert any(a.get("type") == "signal_drought" for a in result)


def test_run_anomaly_checks_broadcasts_anomaly():
    from analysis.anomaly_detector import Anomaly
    from scheduler.alerts import run_anomaly_checks

    anomaly = Anomaly(
        type="price_spike",
        severity="warning",
        symbol="TSLA",
        message="TSLA spiked 15%",
    )

    mock_broadcast = MagicMock()
    with patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=None), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_price_spike",
               return_value=anomaly), \
         patch("scheduler.alerts.broadcast", mock_broadcast):
        result = run_anomaly_checks(watchlist=["TSLA"], current_prices={"TSLA": 250.0})

    assert mock_broadcast.called
    assert len(result) == 1
    assert result[0]["type"] == "price_spike"


# ── run_var_check ──────────────────────────────────────────────────────────────

def test_run_var_check_no_rows_returns_none():
    """When there are no portfolio snapshots in DB, run_var_check returns None."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []

    with patch("scheduler.alerts.get_connection", return_value=mock_conn):
        from scheduler.alerts import run_var_check
        result = run_var_check()

    assert result is None


def test_run_var_check_insufficient_data_returns_none():
    """compute_risk_metrics returns None for < 30 data points → run_var_check returns None."""
    mock_conn = MagicMock()
    # Provide only 5 rows — too few for meaningful risk metrics
    mock_conn.execute.return_value.fetchall.return_value = [(100_000.0,)] * 5

    with patch("scheduler.alerts.get_connection", return_value=mock_conn), \
         patch("analysis.risk_metrics.compute_risk_metrics", return_value=None):
        from scheduler.alerts import run_var_check
        result = run_var_check()

    assert result is None


def test_run_var_check_below_threshold_returns_none():
    """VaR below threshold → run_var_check returns None without broadcasting."""
    from dataclasses import dataclass

    @dataclass
    class _RM:
        var_95: float = 0.01  # below 0.03 threshold
        var_99: float = 0.015
        cvar_95: float = 0.012
        cvar_99: float = 0.018
        volatility_annual: float = 0.15
        n_observations: int = 100

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [(100_000.0 + i,) for i in range(50)]

    with patch("scheduler.alerts.get_connection", return_value=mock_conn), \
         patch("analysis.risk_metrics.compute_risk_metrics", return_value=_RM()):
        from scheduler.alerts import run_var_check
        result = run_var_check(var_threshold=0.03)

    assert result is None


def test_run_var_check_above_threshold_returns_dict():
    """VaR above threshold → run_var_check fires alert and returns metrics dict."""
    from dataclasses import dataclass

    @dataclass
    class _RM:
        var_95: float = 0.05  # above 0.03 threshold
        var_99: float = 0.07
        cvar_95: float = 0.06
        cvar_99: float = 0.08
        volatility_annual: float = 0.25
        n_observations: int = 100

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [(100_000.0 + i,) for i in range(50)]

    with patch("scheduler.alerts.get_connection", return_value=mock_conn), \
         patch("analysis.risk_metrics.compute_risk_metrics", return_value=_RM()), \
         patch("scheduler.alerts.broadcast") as mock_broadcast:
        from scheduler.alerts import run_var_check
        result = run_var_check(var_threshold=0.03)

    assert result is not None
    assert result["var_95"] == pytest.approx(0.05)
    assert "message" in result
    assert mock_broadcast.called


def test_run_var_check_db_exception_returns_none():
    """DB exception is swallowed and function returns None gracefully."""
    mock_conn = MagicMock()
    mock_conn.execute.side_effect = Exception("table not found")

    with patch("scheduler.alerts.get_connection", return_value=mock_conn):
        from scheduler.alerts import run_var_check
        result = run_var_check()

    assert result is None


# ── run_correlation_check null-fallback paths ─────────────────────────────────

def test_run_correlation_check_fetches_positions_when_none(monkeypatch):
    """When positions=None, should fetch from paper_trader.get_portfolio."""
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")

    mock_df = pd.DataFrame([{"Ticker": "AAPL", "Market Value": 10_000.0}])

    with patch("broker.paper_trader.get_portfolio", return_value=mock_df), \
         patch("risk.correlation.check_correlation_alerts", return_value=[]):
        from scheduler.alerts import run_correlation_check
        result = run_correlation_check(positions=None, price_data={})

    assert result == []


def test_run_correlation_check_fetches_portfolio_exception(monkeypatch):
    """Exception in portfolio fetch → positions defaults to empty → returns []."""
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")

    with patch("broker.paper_trader.get_portfolio", side_effect=RuntimeError("db error")):
        from scheduler.alerts import run_correlation_check
        result = run_correlation_check(positions=None, price_data={})

    assert result == []


def test_run_correlation_check_fetches_price_data_when_none(monkeypatch):
    """When price_data=None, should fetch from data.fetcher for each position."""
    monkeypatch.setenv("CORRELATION_MONITOR_ENABLED", "1")

    mock_df = pd.DataFrame({"Close": [100.0] * 30})

    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df), \
         patch("risk.correlation.check_correlation_alerts", return_value=[]):
        from scheduler.alerts import run_correlation_check
        result = run_correlation_check(positions={"AAPL": 10_000.0}, price_data=None)

    assert result == []


# ── run_anomaly_checks null-fallback paths ────────────────────────────────────

def test_run_anomaly_checks_fetches_watchlist_when_none():
    """When watchlist=None, should fetch from data.watchlist."""
    with patch("data.watchlist.get_watchlist", return_value=["AAPL", "MSFT"]), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=None), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_price_spike",
               return_value=None):
        from scheduler.alerts import run_anomaly_checks
        result = run_anomaly_checks(watchlist=None, current_prices={"AAPL": 180.0, "MSFT": 400.0})

    assert isinstance(result, list)


def test_run_anomaly_checks_watchlist_fetch_exception():
    """Exception in watchlist fetch → watchlist defaults to empty."""
    with patch("data.watchlist.get_watchlist", side_effect=RuntimeError("db error")), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=None):
        from scheduler.alerts import run_anomaly_checks
        result = run_anomaly_checks(watchlist=None, current_prices={})

    assert result == []


def test_run_anomaly_checks_fetches_prices_when_none():
    """When current_prices=None, should fetch from data.fetcher for each ticker."""
    mock_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]})

    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_signal_drought",
               return_value=None), \
         patch("analysis.anomaly_detector.AnomalyDetector.check_price_spike",
               return_value=None):
        from scheduler.alerts import run_anomaly_checks
        result = run_anomaly_checks(watchlist=["AAPL"], current_prices=None)

    assert isinstance(result, list)
