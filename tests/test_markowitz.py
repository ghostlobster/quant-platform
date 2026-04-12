import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.markowitz import (
    compute_efficient_frontier, get_max_sharpe_portfolio,
    get_min_volatility_portfolio, OptimalPortfolio
)


def make_price_data(n=120, seed=5):
    np.random.seed(seed)
    tickers = ["AAPL", "MSFT", "GOOG"]
    return {
        t: pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        for t in tickers
    }


def test_efficient_frontier_shapes():
    data = make_price_data()
    rets, vols, sharpes, weights = compute_efficient_frontier(data, n_portfolios=100)
    assert len(rets) == 100
    assert weights.shape == (100, 3)

def test_weights_sum_to_one():
    data = make_price_data()
    _, _, _, weights = compute_efficient_frontier(data, n_portfolios=50)
    for w in weights:
        assert abs(w.sum() - 1.0) < 1e-9

def test_max_sharpe_portfolio():
    data = make_price_data()
    p = get_max_sharpe_portfolio(data)
    assert isinstance(p, OptimalPortfolio)
    assert abs(sum(p.weights.values()) - 1.0) < 1e-6
    assert p.expected_volatility > 0

def test_min_vol_portfolio():
    data = make_price_data()
    p_min = get_min_volatility_portfolio(data)
    p_max = get_max_sharpe_portfolio(data)
    assert p_min.expected_volatility <= p_max.expected_volatility + 0.001

def test_single_asset_returns_empty():
    data = {"AAPL": pd.Series([100, 101, 102, 103])}
    rets, vols, _, _ = compute_efficient_frontier(data)
    assert len(rets) == 0
