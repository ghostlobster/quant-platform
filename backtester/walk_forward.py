"""Walk-forward backtesting — rolling train/test windows."""
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from backtester.engine import run_backtest, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    windows: List[BacktestResult] = field(default_factory=list)
    avg_return: float = 0.0
    avg_sharpe: float = 0.0
    avg_sortino: float = 0.0
    avg_max_drawdown: float = 0.0
    total_trades: int = 0
    consistency_score: float = 0.0   # fraction of windows with positive return


def walk_forward(
    df: pd.DataFrame,
    strategy: str = "sma_crossover",
    ticker: str = "TEST",
    train_periods: int = 120,   # bars for in-sample optimisation (unused for now)
    test_periods: int = 60,     # bars for out-of-sample test
    step: int = 30,             # how many bars to slide the window each iteration
) -> WalkForwardResult:
    """
    Slide a test window across the full price series and run the backtest
    on each segment. Returns aggregated stats.
    """
    results: List[BacktestResult] = []
    n = len(df)
    start = train_periods  # skip the first train_periods bars (reserved for future optimisation)

    while start + test_periods <= n:
        segment = df.iloc[start: start + test_periods].copy()
        if len(segment) >= 30:
            try:
                r = run_backtest(segment, strategy=strategy, ticker=ticker,
                                 start_date=None, end_date=None)
                results.append(r)
            except Exception as exc:
                logger.warning("Walk-forward segment [%d:%d] failed — skipped: %s", start, start + test_periods, exc)
        start += step

    if not results:
        return WalkForwardResult()

    returns = [r.total_return_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    sortinos = [r.sortino_ratio for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]
    trades = [r.num_trades for r in results]

    return WalkForwardResult(
        windows=results,
        avg_return=float(np.mean(returns)),
        avg_sharpe=float(np.mean(sharpes)),
        avg_sortino=float(np.mean(sortinos)),
        avg_max_drawdown=float(np.mean(drawdowns)),
        total_trades=int(np.sum(trades)),
        consistency_score=float(np.mean([1 if r > 0 else 0 for r in returns])),
    )


def build_walk_forward_chart(wf_result: WalkForwardResult):
    """Bar chart of per-window returns."""
    import plotly.graph_objects as go
    if not wf_result.windows:
        return go.Figure()
    returns = [r.total_return_pct * 100 for r in wf_result.windows]
    colors = ["green" if r >= 0 else "red" for r in returns]
    fig = go.Figure(data=go.Bar(
        x=[f"Window {i+1}" for i in range(len(returns))],
        y=returns,
        marker_color=colors,
        name="Return %",
    ))
    fig.update_layout(
        title="Walk-Forward: Per-Window Returns",
        yaxis_title="Return (%)",
        xaxis_title="Test Window",
        showlegend=False,
    )
    return fig
