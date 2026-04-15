"""
backtester/walk_forward.py — Walk-forward backtesting with optional parallelism.

Rolling train/test windows over the full price series. The standard
`walk_forward()` is single-threaded. `walk_forward_parallel()` distributes
segments across CPU cores (multiprocessing) or a Ray cluster when available.

ENV vars
--------
    RAY_ENABLED   '1' to use Ray cluster for segment parallelism (default: 0)
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult, run_backtest

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


def _run_segment(args: tuple) -> Optional[BacktestResult]:
    """
    Top-level picklable function for multiprocessing.

    Parameters
    ----------
    args : (segment_df, strategy, ticker, start_idx)
    """
    segment_df, strategy, ticker, start_idx = args
    try:
        return run_backtest(segment_df, strategy=strategy, ticker=ticker,
                            start_date=None, end_date=None)
    except Exception as exc:
        logger.warning("Parallel segment [start=%d] failed — skipped: %s", start_idx, exc)
        return None


def walk_forward_parallel(
    df: pd.DataFrame,
    strategy: str = "sma_crossover",
    ticker: str = "TEST",
    train_periods: int = 120,
    test_periods: int = 60,
    step: int = 30,
    n_workers: Optional[int] = None,
) -> WalkForwardResult:
    """
    Parallelised walk-forward backtest using multiprocessing.

    When RAY_ENABLED=1 and ray is installed, uses a Ray cluster instead of
    multiprocessing.Pool.  Falls back to single-threaded walk_forward() if
    parallelism is unavailable.

    Parameters
    ----------
    df            : full OHLCV DataFrame
    strategy      : backtest strategy name
    ticker        : ticker label
    train_periods : bars reserved for training (skipped)
    test_periods  : bars per test segment
    step          : slide step between windows
    n_workers     : number of parallel workers (default: os.cpu_count())

    Returns
    -------
    WalkForwardResult with results from all segments.
    """
    # Build segment list
    segments: list[tuple] = []
    n = len(df)
    start = train_periods
    while start + test_periods <= n:
        segment = df.iloc[start: start + test_periods].copy()
        if len(segment) >= 30:
            segments.append((segment, strategy, ticker, start))
        start += step

    if not segments:
        return WalkForwardResult()

    # Ray path
    use_ray = os.environ.get("RAY_ENABLED", "0") == "1"
    if use_ray:
        try:
            import ray  # type: ignore[import]
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)

            @ray.remote
            def _ray_segment(args):
                return _run_segment(args)

            futures = [_ray_segment.remote(seg) for seg in segments]
            raw_results = ray.get(futures)
            results = [r for r in raw_results if r is not None]
        except ImportError:
            logger.info("ray not installed; falling back to multiprocessing")
            use_ray = False
        except Exception as exc:
            logger.warning("Ray execution failed (%s); falling back to multiprocessing", exc)
            use_ray = False

    # Multiprocessing path
    if not use_ray:
        workers = n_workers or min(os.cpu_count() or 1, len(segments))
        results: list[BacktestResult] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_segment, seg): seg for seg in segments}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

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
