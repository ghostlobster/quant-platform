"""
backtester/walk_forward.py — Walk-forward backtesting with optional parallelism.

Rolling train/test windows over the full price series. The standard
`walk_forward()` is single-threaded. `walk_forward_parallel()` distributes
segments across executors:

* ``auto`` (default — P1.12) — prefer Ray when importable, otherwise
  ``ProcessPoolExecutor``. Single-process serial when only one segment.
* ``ray`` — force the Ray path (raises if ``ray`` is not installed).
* ``mp``  — force ``ProcessPoolExecutor``.
* ``serial`` — single-threaded (useful for debugging).

ENV vars
--------
    WF_EXECUTOR   auto | ray | mp | serial  (default: auto)
    RAY_ENABLED   legacy alias — when '1' it forces ``ray`` if WF_EXECUTOR is unset
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


_VALID_EXECUTORS = ("auto", "ray", "mp", "serial")


def _resolve_executor(executor: Optional[str]) -> str:
    """Pick the concrete executor name based on the request and the environment.

    ``executor`` argument takes precedence over the ``WF_EXECUTOR`` env var,
    which takes precedence over the legacy ``RAY_ENABLED`` flag.
    """
    requested = (executor or os.environ.get("WF_EXECUTOR", "auto")).lower().strip()
    if requested not in _VALID_EXECUTORS:
        raise ValueError(
            f"executor must be one of {_VALID_EXECUTORS}, got {requested!r}",
        )
    if requested == "auto":
        # Honour the legacy env-var if the operator hasn't picked WF_EXECUTOR.
        if os.environ.get("RAY_ENABLED", "0") == "1":
            return "ray"
        try:
            import ray  # noqa: F401
            return "ray"
        except ImportError:
            return "mp"
    return requested


def walk_forward_parallel(
    df: pd.DataFrame,
    strategy: str = "sma_crossover",
    ticker: str = "TEST",
    train_periods: int = 120,
    test_periods: int = 60,
    step: int = 30,
    n_workers: Optional[int] = None,
    executor: Optional[str] = None,
) -> WalkForwardResult:
    """Parallelised walk-forward backtest.

    When ``executor`` is unset / ``"auto"`` the function prefers Ray (if
    importable) and otherwise falls back to ``ProcessPoolExecutor``. Pass
    ``executor="serial"`` for deterministic single-threaded debugging.

    Parameters
    ----------
    df            : full OHLCV DataFrame
    strategy      : backtest strategy name
    ticker        : ticker label
    train_periods : bars reserved for training (skipped)
    test_periods  : bars per test segment
    step          : slide step between windows
    n_workers     : number of parallel workers (default: ``os.cpu_count()``)
    executor      : ``auto | ray | mp | serial`` — override the env-var
                    selection.

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

    chosen = _resolve_executor(executor)

    # Ray path
    if chosen == "ray":
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
            chosen = "mp"
        except Exception as exc:
            logger.warning("Ray execution failed (%s); falling back to multiprocessing", exc)
            chosen = "mp"

    # Multiprocessing path
    if chosen == "mp":
        workers = n_workers or min(os.cpu_count() or 1, len(segments))
        results: list[BacktestResult] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_segment, seg): seg for seg in segments}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)

    # Serial path
    if chosen == "serial":
        results = []
        for seg in segments:
            r = _run_segment(seg)
            if r is not None:
                results.append(r)

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


def purged_walk_forward(
    strategy_fn,
    feature_matrix: pd.DataFrame,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> WalkForwardResult:
    """
    Purged walk-forward cross-validation with an embargo gap.

    Prevents label leakage from overlapping forward-return labels by inserting
    an embargo gap between the end of each training period and the start of
    the corresponding test period.

    Parameters
    ----------
    strategy_fn    : callable(segment: pd.DataFrame) -> pd.Series
                     Accepts a MultiIndex (date, ticker) feature matrix segment
                     and returns a pd.Series of daily portfolio returns indexed
                     by date.
    feature_matrix : MultiIndex (date, ticker) DataFrame from build_feature_matrix
    n_splits       : number of train/test folds (default: 5)
    embargo_pct    : fraction of total date count used as the embargo gap
                     (default: 0.01 = 1% of total bars ≈ ~5 trading days on 2y data)

    Returns
    -------
    WalkForwardResult (same structure as walk_forward())

    Notes
    -----
    For fold k (0-indexed) of n_splits:
        train_end  = dates[int(total * (k + 1) / n_splits)]
        gap_end    = train_end + embargo_bars
        test_start = dates[gap_end]
        test_end   = dates[int(total * (k + 2) / n_splits)] or last date
    Fold k=0 uses the first (1/n_splits) of dates as training.
    """
    if feature_matrix.empty:
        logger.warning("purged_walk_forward: empty feature matrix")
        return WalkForwardResult()

    all_dates = sorted(feature_matrix.index.get_level_values(0).unique())
    total = len(all_dates)
    embargo_bars = max(1, int(embargo_pct * total))

    results: list[BacktestResult] = []

    for k in range(n_splits - 1):
        train_end_idx = int(total * (k + 1) / n_splits)
        test_start_idx = min(train_end_idx + embargo_bars, total - 1)
        test_end_idx = min(int(total * (k + 2) / n_splits), total)

        if test_start_idx >= test_end_idx:
            logger.warning("purged_walk_forward: fold %d has empty test window — skipped", k)
            continue

        train_dates = set(all_dates[:train_end_idx])
        test_dates = set(all_dates[test_start_idx:test_end_idx])

        train_seg = feature_matrix[feature_matrix.index.get_level_values(0).isin(train_dates)]
        test_seg = feature_matrix[feature_matrix.index.get_level_values(0).isin(test_dates)]

        if train_seg.empty or test_seg.empty:
            continue

        try:
            # strategy_fn receives (train_seg, test_seg); returns a returns Series
            daily_returns = strategy_fn(train_seg, test_seg)
            if daily_returns is None or daily_returns.empty:
                continue

            # Compute backtest-compatible metrics from the returns Series
            rets = daily_returns.dropna()
            if len(rets) < 2:
                continue

            total_return = float((1 + rets).prod() - 1)
            ann_vol = float(rets.std() * np.sqrt(252))
            sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

            # Sortino: downside deviation
            neg = rets[rets < 0]
            downside_std = float(neg.std() * np.sqrt(252)) if len(neg) > 1 else ann_vol
            sortino = float(rets.mean() * 252 / downside_std) if downside_std > 0 else 0.0

            # Max drawdown from equity curve
            equity = (1 + rets).cumprod()
            roll_max = equity.cummax()
            drawdown = (equity - roll_max) / roll_max
            max_dd = float(drawdown.min())

            calmar = float(total_return / abs(max_dd)) if max_dd != 0 else 0.0

            # Wrap in a BacktestResult for compatibility with WalkForwardResult
            result = BacktestResult(
                ticker="portfolio",
                strategy="purged_wf",
                start_date=rets.index[0],
                end_date=rets.index[-1],
                total_return_pct=total_return,
                buy_hold_return_pct=0.0,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                max_drawdown_pct=max_dd,
                num_trades=0,
                win_rate_pct=float((rets > 0).mean() * 100),
                avg_trade_pct=float(rets.mean()),
                stop_losses_triggered=0,
            )
            results.append(result)

        except Exception as exc:
            logger.warning("purged_walk_forward: fold %d failed — skipped: %s", k, exc)

    if not results:
        return WalkForwardResult()

    returns = [r.total_return_pct for r in results]
    sharpes = [r.sharpe_ratio for r in results]
    sortinos = [r.sortino_ratio for r in results]
    drawdowns = [r.max_drawdown_pct for r in results]

    return WalkForwardResult(
        windows=results,
        avg_return=float(np.mean(returns)),
        avg_sharpe=float(np.mean(sharpes)),
        avg_sortino=float(np.mean(sortinos)),
        avg_max_drawdown=float(np.mean(drawdowns)),
        total_trades=0,
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
