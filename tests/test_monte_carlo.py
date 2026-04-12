import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.monte_carlo import run_monte_carlo, MonteCarloResult


def make_returns(n=252, seed=1):
    np.random.seed(seed)
    return pd.Series(np.random.normal(0.0003, 0.012, n))


def test_monte_carlo_returns_result():
    r = make_returns()
    mc = run_monte_carlo(r, n_simulations=100, n_periods=60)
    assert isinstance(mc, MonteCarloResult)

def test_prob_profit_in_range():
    r = make_returns()
    mc = run_monte_carlo(r, n_simulations=200, n_periods=60)
    assert 0.0 <= mc.prob_profit <= 1.0

def test_percentile_ordering():
    r = make_returns()
    mc = run_monte_carlo(r, n_simulations=200, n_periods=120)
    assert mc.percentile_5 <= mc.percentile_25 <= mc.median_return <= mc.percentile_75 <= mc.percentile_95

def test_short_returns_empty_result():
    r = pd.Series([0.01, -0.02])
    mc = run_monte_carlo(r)
    assert mc.median_return == 0
