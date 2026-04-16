"""
data/frac_diff.py — Fixed-width fractional differentiation.

Price levels are typically non-stationary (unit-root) but integer differencing
destroys their informational memory. Fractional differentiation preserves
memory while driving the series toward stationarity — the key compromise
identified by López de Prado for ML-friendly features.

This module implements the **fixed-width window (FFD)** variant (Ch 5.5 of
*Advances in Financial Machine Learning*), which truncates weights once they
fall below a threshold to keep the effective memory constant across rolling
windows.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def frac_diff_weights(d: float, thresh: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Recursive fractional differentiation weights, truncated at ``thresh``.

    The weights obey ``w_0 = 1`` and
    ``w_k = -w_{k-1} * (d - k + 1) / k``.

    Returns a 1-D numpy array ordered from newest (index 0 = current bar)
    to oldest.
    """
    if d < 0:
        raise ValueError("d must be non-negative")
    weights = [1.0]
    for k in range(1, max_size):
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        weights.append(w_k)
    return np.array(weights, dtype=float)


def frac_diff_ffd(
    series: pd.Series,
    d: float,
    thresh: float = 1e-5,
) -> pd.Series:
    """Fixed-width-window fractional differentiation of ``series``.

    Parameters
    ----------
    series : pandas Series of prices (or log prices) indexed by datetime.
    d      : fractional differentiation order in [0, 1]. ``0`` = identity,
             ``1`` = first-difference.
    thresh : weight truncation threshold; lower values use longer windows.

    Returns
    -------
    pandas Series of the same index, with the first ``len(weights) - 1``
    observations NaN (the window burn-in).
    """
    if series.empty:
        return series.copy()

    weights = frac_diff_weights(d, thresh=thresh)
    window = len(weights)
    values = series.astype(float).to_numpy()
    out = np.full_like(values, np.nan, dtype=float)

    if window > len(values):
        return pd.Series(out, index=series.index, name=series.name)

    # Convolve: weights[0] is the newest bar, weights[-1] the oldest in window.
    for i in range(window - 1, len(values)):
        block = values[i - window + 1 : i + 1][::-1]  # newest → oldest
        out[i] = float(np.dot(weights, block))

    return pd.Series(out, index=series.index, name=series.name)


def find_min_d(
    series: pd.Series,
    d_values: list[float] | None = None,
    p_threshold: float = 0.05,
    thresh: float = 1e-5,
) -> float:
    """Smallest ``d`` that makes ``frac_diff_ffd(series, d)`` ADF-stationary.

    Falls back to returning ``0.5`` (middle of the typical range) if
    statsmodels is not installed or no tested ``d`` passes the ADF test.
    """
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore[import]
    except ImportError:
        return 0.5

    candidates = d_values or [round(x, 2) for x in np.arange(0.0, 1.01, 0.1)]
    for d in candidates:
        diff = frac_diff_ffd(series, d=d, thresh=thresh).dropna()
        if len(diff) < 20:
            continue
        try:
            p_value = float(adfuller(diff.values, autolag="AIC")[1])
        except Exception:
            continue
        if p_value < p_threshold:
            return float(d)
    return 0.5
