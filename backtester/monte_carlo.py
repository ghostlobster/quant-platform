"""Monte Carlo simulation for return path analysis."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass
class MonteCarloResult:
    median_return: float
    percentile_5: float
    percentile_25: float
    percentile_75: float
    percentile_95: float
    prob_profit: float        # fraction of paths ending positive
    n_simulations: int
    n_periods: int


def run_monte_carlo(
    returns: pd.Series,
    n_simulations: int = 1000,
    n_periods: int = 252,
    initial_value: float = 100_000.0,
) -> MonteCarloResult:
    """Bootstrap Monte Carlo from historical returns."""
    if returns.empty or len(returns) < 20:
        return MonteCarloResult(0,0,0,0,0,0, n_simulations, n_periods)

    r = returns.dropna().values
    rng = np.random.default_rng(seed=42)
    # Sample returns with replacement
    sampled = rng.choice(r, size=(n_simulations, n_periods), replace=True)
    # Compound growth
    paths = initial_value * np.cumprod(1 + sampled, axis=1)
    final_values = paths[:, -1]
    final_returns = (final_values - initial_value) / initial_value

    return MonteCarloResult(
        median_return=float(np.median(final_returns)),
        percentile_5=float(np.percentile(final_returns, 5)),
        percentile_25=float(np.percentile(final_returns, 25)),
        percentile_75=float(np.percentile(final_returns, 75)),
        percentile_95=float(np.percentile(final_returns, 95)),
        prob_profit=float(np.mean(final_returns > 0)),
        n_simulations=n_simulations,
        n_periods=n_periods,
    )


def build_monte_carlo_chart(
    returns: pd.Series,
    n_simulations: int = 200,
    n_periods: int = 252,
    initial_value: float = 100_000.0,
) -> go.Figure:
    """Plot a fan chart of simulated equity paths (shows 50 paths + percentile bands)."""
    if returns.empty or len(returns) < 20:
        return go.Figure()

    r = returns.dropna().values
    rng = np.random.default_rng(seed=42)
    sampled = rng.choice(r, size=(n_simulations, n_periods), replace=True)
    paths = initial_value * np.cumprod(1 + sampled, axis=1)
    x = list(range(n_periods))

    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    # 90% confidence band
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p95)+list(p5[::-1]),
        fill='toself', fillcolor='rgba(0,100,255,0.1)', line=dict(color='rgba(0,0,0,0)'),
        name='5–95th pct'))
    # 50% band
    fig.add_trace(go.Scatter(x=x+x[::-1], y=list(p75)+list(p25[::-1]),
        fill='toself', fillcolor='rgba(0,100,255,0.2)', line=dict(color='rgba(0,0,0,0)'),
        name='25–75th pct'))
    # Median
    fig.add_trace(go.Scatter(x=x, y=p50, line=dict(color='royalblue', width=2), name='Median'))
    fig.update_layout(title=f'Monte Carlo ({n_simulations} paths, {n_periods} days)',
                      yaxis_title='Portfolio Value ($)', xaxis_title='Trading Days')
    return fig
