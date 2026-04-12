import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.var import historical_var, parametric_var, conditional_var, portfolio_var


def make_returns(n=252, seed=42):
    np.random.seed(seed)
    return pd.Series(np.random.normal(-0.0002, 0.015, n))


def test_historical_var_positive():
    r = make_returns()
    v = historical_var(r)
    assert v > 0
    assert v < 0.10  # sanity: daily VaR < 10%

def test_parametric_var_positive():
    r = make_returns()
    v = parametric_var(r)
    assert v > 0

def test_cvar_ge_var():
    r = make_returns()
    var = historical_var(r)
    cvar = conditional_var(r)
    assert cvar >= var  # CVaR >= VaR by definition

def test_short_series_returns_zero():
    r = pd.Series([0.01, -0.02, 0.005])
    assert historical_var(r) == 0.0
    assert parametric_var(r) == 0.0

def test_portfolio_var():
    weights = np.array([0.6, 0.4])
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    v = portfolio_var(weights, cov)
    assert v > 0
