"""Tests for portfolio stress testing module (Issue #31)."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_portfolio():
    return pd.DataFrame({
        "Ticker": ["AAPL", "MSFT", "TSLA"],
        "Market Value": [40_000.0, 35_000.0, 25_000.0],
    })


def test_apply_gfc_scenario(sample_portfolio):
    from analysis.stress_test import HISTORICAL_SCENARIOS, apply_scenario
    gfc = next(s for s in HISTORICAL_SCENARIOS if s.name == "2008_gfc")
    result = apply_scenario(sample_portfolio, gfc)
    assert result.pre_nav == pytest.approx(100_000.0)
    # -52% shock: post_nav ≈ 48000
    assert result.post_nav == pytest.approx(48_000.0, rel=1e-3)
    assert result.nav_change_pct == pytest.approx(-0.52, rel=1e-3)


def test_apply_covid_scenario(sample_portfolio):
    from analysis.stress_test import HISTORICAL_SCENARIOS, apply_scenario
    covid = next(s for s in HISTORICAL_SCENARIOS if s.name == "2020_covid")
    result = apply_scenario(sample_portfolio, covid)
    assert result.nav_change_pct == pytest.approx(-0.34, rel=1e-3)
    assert result.worst_position in ["AAPL", "MSFT", "TSLA"]


def test_run_all_historical_scenarios(sample_portfolio):
    from analysis.stress_test import run_stress_tests
    results = run_stress_tests(sample_portfolio)
    assert len(results) == 3
    scenario_names = {r.scenario_name for r in results}
    assert "2008_gfc" in scenario_names
    assert "2020_covid" in scenario_names
    assert "2022_rate_hike" in scenario_names


def test_custom_shock_zero_no_change(sample_portfolio):
    from analysis.stress_test import apply_custom_shock
    result = apply_custom_shock(sample_portfolio, equity_pct=0.0)
    assert result.nav_change == pytest.approx(0.0)
    assert result.post_nav == pytest.approx(result.pre_nav)


def test_custom_shock_positive(sample_portfolio):
    from analysis.stress_test import apply_custom_shock
    result = apply_custom_shock(sample_portfolio, equity_pct=0.10)
    assert result.post_nav > result.pre_nav
    assert result.nav_change_pct == pytest.approx(0.10, rel=1e-3)


def test_apply_scenario_empty_portfolio():
    from analysis.stress_test import HISTORICAL_SCENARIOS, apply_scenario
    empty = pd.DataFrame(columns=["Ticker", "Market Value"])
    result = apply_scenario(empty, HISTORICAL_SCENARIOS[0])
    assert result.pre_nav == 0.0
    assert result.post_nav == 0.0


def test_llm_scenario_disabled_by_default():
    import os
    os.environ.pop("LLM_STRESS_SCENARIOS", None)
    from analysis.stress_test import generate_llm_scenarios
    results = generate_llm_scenarios("test portfolio", "trending_bull")
    assert results == []
