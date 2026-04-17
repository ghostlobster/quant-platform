"""
analysis/cointegration.py — Engle-Granger cointegration for pairs discovery.

The existing :mod:`strategies.pairs` module computes hedge ratios and
z-score signals but relies on hand-picked pairs.  This module provides
the two primitives AFML / Jansen Ch 9 recommend for *discovering* pairs
instead of assuming them:

  * :func:`engle_granger` — Engle-Granger cointegration test between
    two price series; returns the p-value, OLS hedge ratio and
    AR(1)-style half-life of the spread.
  * :func:`screen_cointegrated_pairs` — iterate all 2-subsets of a
    ticker → price-series mapping and return the pairs with
    ``p < significance``.

Optional deps
-------------
    statsmodels >= 0.14  (``pip install statsmodels``)

When ``statsmodels`` isn't installed, :func:`engle_granger` returns a
result with ``p_value = None`` and ``converged=False`` so callers can
degrade gracefully.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading*, Ch 9.5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.stattools import coint  # type: ignore[import]
    _STATSMODELS_AVAILABLE = True
except ImportError:
    coint = None  # type: ignore[assignment]
    _STATSMODELS_AVAILABLE = False


@dataclass
class CointegrationResult:
    """Per-pair output of :func:`engle_granger`."""

    p_value: Optional[float]
    hedge_ratio: Optional[float]
    half_life: Optional[float]
    converged: bool


def _hedge_ratio_ols(a: np.ndarray, b: np.ndarray) -> float:
    """OLS slope of ``a`` regressed on ``b`` (no intercept)."""
    num = float(np.dot(a, b))
    den = float(np.dot(b, b))
    return num / den if den > 0 else 0.0


def _half_life(spread: np.ndarray) -> float:
    """AR(1) half-life from the lagged-spread regression."""
    if len(spread) < 10:
        return float("inf")
    lag = spread[:-1]
    delta = spread[1:] - lag
    if np.var(lag) == 0:
        return float("inf")
    beta = float(np.dot(delta, lag) / np.dot(lag, lag))
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / beta)


def engle_granger(
    y: pd.Series, x: pd.Series,
) -> CointegrationResult:
    """Engle-Granger two-step cointegration test.

    Parameters
    ----------
    y, x :
        Two aligned ``pd.Series`` of prices.  The series are inner-
        joined on their index before testing, so different lengths are
        tolerated.  At least 30 overlapping observations are required.

    Returns
    -------
    :class:`CointegrationResult` with ``p_value``, the OLS
    ``hedge_ratio`` (``y ~ hedge_ratio * x``), the spread's
    ``half_life`` (days), and a ``converged`` flag.

    ``p_value`` is ``None`` when the test could not run (e.g.
    insufficient data or ``statsmodels`` missing); callers should
    interpret ``converged=False`` as "treat as non-cointegrated".
    """
    if y is None or x is None:
        return CointegrationResult(None, None, None, False)
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(df) < 30:
        return CointegrationResult(None, None, None, False)

    y_arr = df.iloc[:, 0].to_numpy(dtype=float)
    x_arr = df.iloc[:, 1].to_numpy(dtype=float)

    hedge = _hedge_ratio_ols(y_arr, x_arr)
    spread = y_arr - hedge * x_arr
    hl = _half_life(spread)

    if not _STATSMODELS_AVAILABLE:
        return CointegrationResult(None, hedge, hl, False)

    try:
        _, p_value, _ = coint(y_arr, x_arr)
    except Exception:
        return CointegrationResult(None, hedge, hl, False)

    return CointegrationResult(
        p_value=float(p_value),
        hedge_ratio=float(hedge),
        half_life=float(hl),
        converged=True,
    )


def screen_cointegrated_pairs(
    price_data: dict[str, pd.Series],
    significance: float = 0.05,
    max_half_life: Optional[float] = None,
) -> list[dict]:
    """Screen all 2-subsets of ``price_data`` for cointegration.

    Parameters
    ----------
    price_data :
        ``{ticker: price_series}`` mapping.  Series are Engle-Granger
        tested pairwise (O(n²)).
    significance :
        Threshold on the p-value — pairs below this value are kept.
    max_half_life :
        Optional upper bound on the AR(1) half-life of the spread.
        Useful for filtering out pairs that mean-revert too slowly for
        practical trading.  ``None`` (default) means "no half-life
        filter".

    Returns
    -------
    List of dicts ``{"a", "b", "p_value", "hedge_ratio", "half_life"}``
    sorted ascending by p-value (tightest cointegration first).  Empty
    list when ``statsmodels`` isn't installed or no pair passes the
    filter.
    """
    if not price_data or len(price_data) < 2:
        return []

    tickers = list(price_data.keys())
    results: list[dict] = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1 :]:
            res = engle_granger(price_data[a], price_data[b])
            if not res.converged or res.p_value is None:
                continue
            if res.p_value >= significance:
                continue
            if max_half_life is not None and (
                res.half_life is None or res.half_life > max_half_life
            ):
                continue
            results.append(
                {
                    "a": a,
                    "b": b,
                    "p_value": res.p_value,
                    "hedge_ratio": res.hedge_ratio,
                    "half_life": res.half_life,
                }
            )
    results.sort(key=lambda r: r["p_value"])
    return results
