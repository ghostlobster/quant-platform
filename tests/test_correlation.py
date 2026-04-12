import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk.correlation import compute_correlation_matrix, build_heatmap


def make_prices(n=60, seed=42):
    np.random.seed(seed)
    return {
        "AAPL": pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5)),
        "MSFT": pd.Series(200 + np.cumsum(np.random.randn(n) * 0.7)),
        "GOOG": pd.Series(150 + np.cumsum(np.random.randn(n) * 0.6)),
    }


def test_corr_matrix_shape():
    prices = make_prices()
    corr = compute_correlation_matrix(prices)
    assert corr.shape == (3, 3)


def test_corr_diagonal_is_one():
    prices = make_prices()
    corr = compute_correlation_matrix(prices)
    for i in range(len(corr)):
        assert abs(corr.iloc[i, i] - 1.0) < 1e-9


def test_corr_values_in_range():
    prices = make_prices()
    corr = compute_correlation_matrix(prices)
    assert (corr.values >= -1.0).all()
    assert (corr.values <= 1.0).all()


def test_empty_input_returns_empty():
    corr = compute_correlation_matrix({})
    assert corr.empty


def test_heatmap_returns_figure():
    prices = make_prices()
    corr = compute_correlation_matrix(prices)
    fig = build_heatmap(corr)
    assert fig is not None
    assert len(fig.data) > 0
