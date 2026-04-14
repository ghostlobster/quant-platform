"""Tests for analysis/anomaly_detector.py (Issue #33)."""
from __future__ import annotations

import sys
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Ensure yfinance is available as a mock so data.fetcher can be imported
_yf_mock = MagicMock()
sys.modules.setdefault("yfinance", _yf_mock)


def test_signal_drought_detected_when_log_empty():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    # Provide an empty signal log — drought should be reported
    result = det.check_signal_drought(signal_log=[], window_hours=4)
    assert result is not None
    assert result.type == "signal_drought"
    assert result.severity == "warning"


def test_signal_drought_not_triggered_with_recent_signal():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    recent_ts = time.time() - 60  # 1 minute ago
    log = [{"entry_time": recent_ts, "ticker": "AAPL", "action": "buy"}]
    result = det.check_signal_drought(signal_log=log, window_hours=4)
    assert result is None


def test_signal_drought_old_signals_trigger():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    old_ts = time.time() - 8 * 3600  # 8 hours ago
    log = [{"entry_time": old_ts, "ticker": "SPY", "action": "sell"}]
    result = det.check_signal_drought(signal_log=log, window_hours=4)
    assert result is not None
    assert result.type == "signal_drought"


def test_price_spike_detected_above_threshold():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    # Stable recent prices around 100; current price is 130 → 30% spike
    mock_close = pd.Series([100.0, 100.5, 99.8, 100.2, 100.1])
    mock_df = pd.DataFrame({"Close": mock_close})

    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df):
        result = det.check_price_spike("AAPL", 130.0, threshold_pct=0.10)
    assert result is not None
    assert result.type == "price_spike"
    assert result.symbol == "AAPL"


def test_price_spike_not_triggered_within_threshold():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    mock_close = pd.Series([100.0, 100.5, 99.8, 100.2, 100.1])
    mock_df = pd.DataFrame({"Close": mock_close})

    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df):
        result = det.check_price_spike("AAPL", 101.0, threshold_pct=0.10)
    assert result is None


def test_price_spike_critical_severity():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    mock_close = pd.Series([100.0] * 5)
    mock_df = pd.DataFrame({"Close": mock_close})

    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df):
        result = det.check_price_spike("TSLA", 130.0, threshold_pct=0.10)
    assert result is not None
    assert result.severity == "critical"  # 30% > 20% critical threshold


def test_pnl_divergence_detected():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    live = [100.0, -200.0, 50.0, -300.0, 80.0]    # net: -270
    paper = [100.0, 100.0, 50.0, 100.0, 80.0]      # net: +430
    result = det.check_pnl_divergence(live, paper, threshold_pct=0.10)
    assert result is not None
    assert result.type == "pnl_divergence"


def test_pnl_divergence_not_triggered_when_similar():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    live = [100.0, 50.0, -10.0, 80.0, 20.0]
    paper = [100.0, 52.0, -11.0, 82.0, 21.0]  # Very close values
    result = det.check_pnl_divergence(live, paper, threshold_pct=0.10)
    assert result is None


def test_pnl_divergence_empty_inputs():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    assert det.check_pnl_divergence([], [], threshold_pct=0.05) is None
    assert det.check_pnl_divergence([100.0], [], threshold_pct=0.05) is None


def test_pnl_divergence_single_element():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    result = det.check_pnl_divergence([100.0], [100.0], threshold_pct=0.05)
    assert result is None  # Need at least 2 points


def test_run_all_checks_returns_list():
    from analysis.anomaly_detector import AnomalyDetector

    det = AnomalyDetector()
    with patch.object(det, "check_signal_drought", return_value=None), \
         patch("data.fetcher.fetch_ohlcv", side_effect=Exception("no data")):
        results = det.run_all_checks(
            watchlist=["AAPL", "MSFT"],
            current_prices={"AAPL": 180.0, "MSFT": 320.0},
        )
    assert isinstance(results, list)
