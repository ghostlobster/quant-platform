"""
tests/factories.py — synthetic-data factories shared across the
test suite (closes #239 / T2.7).

The audit (#227 et al.) found the same 4-5 factory functions
re-rolled across every other test file with subtle drift between
them: one uses ``np.random.seed(42)``, another uses
``np.random.default_rng(0)``; one uses ``freq="B"``, another
``freq="D"``; column-name capitalisation varies.

This module is the **single source of truth** for new tests. Pull
the helper that fits, customise via keyword arguments. Each
function is fully deterministic given its ``seed`` argument so
tests don't depend on the determinism trio (#227) for stability.

Usage:

    from tests.factories import make_ohlcv, make_returns

    df = make_ohlcv(n=60, seed=7)
    r = make_returns(n=252, sigma=0.02)

Legacy in-file helpers in ``tests/test_*.py`` will migrate over
time. New tests should reach for this module first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    n: int = 100,
    *,
    seed: int = 42,
    base: float = 100.0,
    drift: float = 0.0,
    vol: float = 0.5,
    freq: str = "B",
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Random-walk OHLCV with capitalised column names.

    The canonical shape: ``Open / High / Low / Close / Volume`` —
    matches the dataframe ``data/fetcher.py:fetch_ohlcv`` returns and
    every backtester / strategy / e2e test expects.

    Defaults are tuned to be cheap (n=100, weekday-only frequency)
    and reproducible. Override ``seed`` / ``base`` / ``drift`` /
    ``vol`` for sensitivity tests; override ``freq`` to ``"D"`` for
    calendar-day series; override ``start`` to control the index.

    Parameters
    ----------
    n
        Number of bars. Most caller use 60-300; <30 will trip the
        risk modules' "too short" guards (which is sometimes the
        point — pass deliberately).
    seed
        Pinned for reproducibility. Defaults to 42 (matches the
        legacy ``test_features.py`` helper).
    base
        Starting price level. Default 100.0.
    drift
        Daily drift added to each step (in price units, not %).
        Default 0.0.
    vol
        Per-step Gaussian std. Default 0.5 — produces realistic
        single-name daily moves for ``base=100``.
    freq
        Pandas frequency string for the index. Default ``"B"``
        (business days). Use ``"D"`` for daily-incl-weekends.
    start
        ISO date for the first bar.
    """
    rng = np.random.default_rng(seed)
    close = base + np.cumsum(rng.normal(drift, vol, n))
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame(
        {
            "Open":   close * 0.99,
            "High":   close * 1.01,
            "Low":    close * 0.98,
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def make_prices(
    n: int = 250,
    *,
    base: float = 100.0,
    last: float | None = None,
) -> pd.Series:
    """Constant-price series with an optional endpoint shift.

    Used by regime tests (``analysis/regime.py``) and SMA-200 tests
    where the test wants a fully-deterministic 200-day mean and a
    controlled position relative to it. ``last`` overrides the final
    value so a single-line test can flip "above 200d SMA" vs
    "below 200d SMA".

    Returns ``pd.Series`` (no DatetimeIndex — regime tests don't
    care about dates, only relative magnitudes).
    """
    prices = np.full(n, float(base))
    if last is not None:
        prices[-1] = float(last)
    return pd.Series(prices)


def make_returns(
    n: int = 252,
    *,
    mu: float = -0.0002,
    sigma: float = 0.015,
    seed: int = 42,
) -> pd.Series:
    """Normally-distributed daily returns.

    Defaults match the typical equity-index distribution used by
    risk/var.py and analysis/garch.py tests: slight negative drift
    (consistent with intraday mean-reversion bias) and ~24 % annual
    volatility. Override ``sigma`` to test high/low-vol regimes.
    """
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mu, sigma, n))


def make_feature_matrix(
    n_dates: int = 60,
    n_tickers: int = 10,
    *,
    seed: int = 0,
    freq: str = "B",
    fwd_signal_strength: float = 0.5,
) -> pd.DataFrame:
    """MultiIndex (date, ticker) feature matrix used by ML signal tests.

    Builds a frame indexed by (date, ticker) with the project's
    standard ``_FEATURE_COLS`` plus a synthetic ``fwd_ret_5d``
    target whose linear dependence on the first feature is
    controlled by ``fwd_signal_strength``. Returns to a flat
    DataFrame (callers re-set the index where needed).
    """
    from data.features import _FEATURE_COLS

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq=freq)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            row: dict = {"date": d, "ticker": t}
            for col in _FEATURE_COLS:
                row[col] = rng.normal(0.0, 1.0)
            row["fwd_ret_5d"] = (
                fwd_signal_strength * row[_FEATURE_COLS[0]]
                + rng.normal(0.0, 0.5)
            )
            rows.append(row)
    return pd.DataFrame(rows)
