"""Tests for risk/correlation.py monitoring functions (Issue #32)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _price_series(n: int = 30, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.Series(prices)


def test_rolling_correlation_returns_dataframe():
    from risk.correlation import rolling_correlation

    data = {
        "AAPL": _price_series(30, seed=1),
        "MSFT": _price_series(30, seed=2),
    }
    result = rolling_correlation(data, window=10)
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == result.shape[1]  # square


def test_rolling_correlation_empty_input():
    from risk.correlation import rolling_correlation

    result = rolling_correlation({}, window=20)
    assert result.empty


def test_rolling_correlation_single_ticker():
    from risk.correlation import rolling_correlation

    result = rolling_correlation({"AAPL": _price_series(30)}, window=10)
    assert result.empty


def test_check_correlation_alerts_no_alerts_normal_portfolio():
    from risk.correlation import check_correlation_alerts

    # Uncorrelated price series, balanced positions
    data = {
        "AAPL": _price_series(30, seed=1),
        "MSFT": _price_series(30, seed=99),   # different seed = low corr
        "JPM": _price_series(30, seed=77),
    }
    positions = {"AAPL": 10_000.0, "MSFT": 10_000.0, "JPM": 10_000.0}
    alerts = check_correlation_alerts(data, positions, avg_corr_threshold=0.95)
    # No alert for avg correlation since we set a high threshold
    assert all(a.alert_type != "avg_correlation" for a in alerts)


def test_check_correlation_alerts_position_concentration():
    from risk.correlation import check_correlation_alerts

    data = {
        "AAPL": _price_series(30, seed=1),
        "MSFT": _price_series(30, seed=2),
    }
    # AAPL holds 80% of portfolio — should trigger position concentration alert
    positions = {"AAPL": 80_000.0, "MSFT": 20_000.0}
    alerts = check_correlation_alerts(data, positions, position_weight_threshold=0.25)
    types = {a.alert_type for a in alerts}
    assert "position_concentration" in types


def test_check_correlation_alerts_sector_concentration():
    from risk.correlation import check_correlation_alerts

    data = {
        "AAPL": _price_series(30, seed=1),
        "MSFT": _price_series(30, seed=2),
        "NVDA": _price_series(30, seed=3),
    }
    positions = {"AAPL": 40_000.0, "MSFT": 40_000.0, "NVDA": 20_000.0}
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology"}
    # All three are tech → sector concentration = 100% > threshold 40%
    alerts = check_correlation_alerts(
        data, positions, sector_map=sector_map, sector_weight_threshold=0.40
    )
    types = {a.alert_type for a in alerts}
    assert "sector_concentration" in types


def test_check_correlation_alerts_empty_positions():
    from risk.correlation import check_correlation_alerts

    data = {"AAPL": _price_series(30)}
    alerts = check_correlation_alerts(data, {})
    assert isinstance(alerts, list)


def test_correlation_alert_dataclass_fields():
    from risk.correlation import CorrelationAlert

    alert = CorrelationAlert(
        alert_type="avg_correlation",
        value=0.82,
        threshold=0.70,
        message="High average correlation detected",
        ticker="",
    )
    assert alert.alert_type == "avg_correlation"
    assert alert.value == pytest.approx(0.82)
    assert alert.threshold == pytest.approx(0.70)


def test_no_alerts_for_balanced_diversified_portfolio():
    from risk.correlation import check_correlation_alerts

    rng = np.random.default_rng(0)
    # Build near-orthogonal price series
    data = {}
    positions = {}
    sectors = {}
    tickers = ["A", "B", "C", "D", "E", "F"]
    for i, t in enumerate(tickers):
        prices = 100.0 + np.cumsum(rng.normal(0, 1, 30))
        data[t] = pd.Series(prices)
        positions[t] = 10_000.0  # equal weight
        sectors[t] = f"Sector{i}"  # all different sectors

    alerts = check_correlation_alerts(
        data, positions, sector_map=sectors,
        avg_corr_threshold=0.70,
        position_weight_threshold=0.25,
        sector_weight_threshold=0.40,
    )
    # With equal weights of ~16.7% each and different sectors, no concentration alerts
    conc_alerts = [a for a in alerts if a.alert_type in ("position_concentration", "sector_concentration")]
    assert len(conc_alerts) == 0
