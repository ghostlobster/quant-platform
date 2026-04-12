"""Markowitz Mean-Variance Portfolio Optimization."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class OptimalPortfolio:
    weights: dict          # ticker -> weight
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


def compute_efficient_frontier(
    price_data: dict,
    n_portfolios: int = 3000,
    risk_free_rate: float = 0.05,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo simulation of random portfolios to approximate efficient frontier.
    Returns (returns_arr, vols_arr, sharpes_arr, weights_arr).
    """
    tickers = list(price_data.keys())
    n = len(tickers)
    if n < 2:
        empty = np.array([])
        return empty, empty, empty, np.array([]).reshape(0, n)

    df = pd.DataFrame(price_data).pct_change().dropna()
    mean_returns = df.mean() * 252
    cov_matrix = df.cov() * 252

    rng = np.random.default_rng(seed)
    all_returns, all_vols, all_sharpes = [], [], []
    all_weights = []

    for _ in range(n_portfolios):
        w = rng.random(n)
        w /= w.sum()
        ret = float(w @ mean_returns.values)
        vol = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
        all_returns.append(ret)
        all_vols.append(vol)
        all_sharpes.append(sharpe)
        all_weights.append(w)

    return (np.array(all_returns), np.array(all_vols),
            np.array(all_sharpes), np.array(all_weights))


def get_max_sharpe_portfolio(
    price_data: dict,
    risk_free_rate: float = 0.05,
) -> Optional[OptimalPortfolio]:
    """Return the portfolio with the highest Sharpe ratio."""
    tickers = list(price_data.keys())
    rets, vols, sharpes, weights = compute_efficient_frontier(price_data, risk_free_rate=risk_free_rate)
    if len(sharpes) == 0:
        return None
    idx = np.argmax(sharpes)
    return OptimalPortfolio(
        weights={t: float(w) for t, w in zip(tickers, weights[idx])},
        expected_return=float(rets[idx]),
        expected_volatility=float(vols[idx]),
        sharpe_ratio=float(sharpes[idx]),
    )


def get_min_volatility_portfolio(
    price_data: dict,
) -> Optional[OptimalPortfolio]:
    """Return the minimum variance portfolio."""
    tickers = list(price_data.keys())
    rets, vols, sharpes, weights = compute_efficient_frontier(price_data)
    if len(vols) == 0:
        return None
    idx = np.argmin(vols)
    return OptimalPortfolio(
        weights={t: float(w) for t, w in zip(tickers, weights[idx])},
        expected_return=float(rets[idx]),
        expected_volatility=float(vols[idx]),
        sharpe_ratio=float(sharpes[idx]),
    )


def build_efficient_frontier_chart(price_data: dict, risk_free_rate: float = 0.05) -> go.Figure:
    """Scatter plot of simulated portfolios coloured by Sharpe ratio."""
    rets, vols, sharpes, _ = compute_efficient_frontier(price_data, risk_free_rate=risk_free_rate)
    if len(rets) == 0:
        return go.Figure()
    fig = go.Figure(data=go.Scatter(
        x=vols * 100, y=rets * 100,
        mode="markers",
        marker=dict(color=sharpes, colorscale="Viridis", showscale=True,
                    colorbar=dict(title="Sharpe"), size=4, opacity=0.6),
        text=[f"Sharpe: {s:.2f}" for s in sharpes],
    ))
    fig.update_layout(
        title="Efficient Frontier (Monte Carlo)",
        xaxis_title="Volatility (%)", yaxis_title="Expected Return (%)",
    )
    return fig
