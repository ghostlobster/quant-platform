"""
analysis/factor_ic.py — Information Coefficient (IC) analysis for alpha factors.

Evaluates whether each feature in a feature matrix has statistically
significant predictive power over forward returns, using the standard
Spearman-rank IC methodology from factor research.

Key metrics
-----------
    IC      : Spearman rank correlation of feature vs forward return,
              computed cross-sectionally per date then averaged
    IC Std  : Standard deviation of the per-date IC series
    ICIR    : IC Information Ratio = IC Mean / IC Std
              Values > 0.5 (or < -0.5) indicate a consistent signal
    p-value : One-sample t-test on the IC series (H₀: mean = 0)

Usage
-----
    from data.features import build_feature_matrix
    from analysis.factor_ic import compute_ic

    fm = build_feature_matrix(["AAPL", "MSFT", "GOOGL"], period="2y")
    results = compute_ic(fm)
    for feature, stats in results.items():
        print(feature, stats)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

# ICIR threshold above which a factor is considered meaningfully consistent
MEANINGFUL_ICIR = 0.5

# Minimum number of ticker observations per date to compute a valid IC
_MIN_OBS_PER_DATE = 5


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without scipy dependency."""
    n = len(x)
    if n < 2:
        return np.nan
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    # Pearson on ranks == Spearman
    mx, my = rx.mean(), ry.mean()
    num = ((rx - mx) * (ry - my)).sum()
    den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
    if den == 0:
        return np.nan
    return float(num / den)


def _ttest_1samp_pvalue(series: np.ndarray) -> float:
    """Two-sided p-value for H₀: mean = 0, no scipy required."""
    clean = series[~np.isnan(series)]
    n = len(clean)
    if n < 2:
        return np.nan
    mean = clean.mean()
    std = clean.std(ddof=1)
    if std == 0:
        return 0.0
    t_stat = mean * np.sqrt(n) / std
    # Approximate p-value using normal distribution (valid for n >= 30)
    # |t| → Φ(-|t|) × 2
    from math import erfc, sqrt
    p = float(erfc(abs(t_stat) / sqrt(2)))
    return p


def compute_ic(
    feature_matrix: pd.DataFrame,
    forward_return_col: str = "fwd_ret_5d",
    feature_cols: list[str] | None = None,
) -> dict[str, dict]:
    """
    Compute IC statistics for each feature column.

    Parameters
    ----------
    feature_matrix     : MultiIndex (date, ticker) DataFrame from build_feature_matrix
    forward_return_col : name of the target forward-return column
    feature_cols       : subset of features to evaluate; None = all non-fwd_ret columns

    Returns
    -------
    dict mapping feature_name → {
        "ic_mean" : float  — mean IC across dates
        "ic_std"  : float  — std of per-date IC series
        "icir"    : float  — ic_mean / ic_std (0 if ic_std == 0)
        "p_value" : float  — two-sided p-value (H₀: mean IC = 0)
        "n_dates" : int    — number of dates with ≥ _MIN_OBS_PER_DATE observations
    }
    """
    if feature_matrix.empty:
        log.warning("compute_ic: empty feature matrix")
        return {}

    if forward_return_col not in feature_matrix.columns:
        log.warning("compute_ic: target column missing", col=forward_return_col)
        return {}

    all_cols = list(feature_matrix.columns)
    if feature_cols is None:
        feature_cols = [c for c in all_cols if not c.startswith("fwd_ret_")]
    else:
        feature_cols = [c for c in feature_cols if c in all_cols]

    if not feature_cols:
        log.warning("compute_ic: no valid feature columns found")
        return {}

    results: dict[str, dict] = {}

    for feat in feature_cols:
        ic_series: list[float] = []
        valid_dates = 0

        for _date, group in feature_matrix.groupby(level="date"):
            sub = group[[feat, forward_return_col]].dropna()
            if len(sub) < _MIN_OBS_PER_DATE:
                continue
            ic = _spearman_corr(sub[feat].values, sub[forward_return_col].values)
            if not np.isnan(ic):
                ic_series.append(ic)
                valid_dates += 1

        if not ic_series:
            results[feat] = {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan,
                             "p_value": np.nan, "n_dates": 0}
            continue

        arr = np.array(ic_series)
        ic_mean = float(arr.mean())
        ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        icir = ic_mean / ic_std if ic_std != 0 else 0.0
        p_value = _ttest_1samp_pvalue(arr)

        results[feat] = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "p_value": p_value,
            "n_dates": valid_dates,
        }

    log.info("compute_ic: complete", features=len(results), target=forward_return_col)
    return results


def compute_ic_decay(
    feature_matrix: pd.DataFrame,
    feature_col: str,
    horizons: list[int] | None = None,
) -> dict[int, dict]:
    """
    Compute IC at multiple forward-return horizons to measure signal decay.

    Parameters
    ----------
    feature_matrix : MultiIndex (date, ticker) frame containing fwd_ret_{h}d columns
    feature_col    : the single feature column to analyse
    horizons       : list of horizon days (default: [1, 5, 10, 21])

    Returns
    -------
    dict mapping horizon_days → compute_ic() result dict
    """
    if horizons is None:
        horizons = [1, 5, 10, 21]

    decay: dict[int, dict] = {}
    for h in horizons:
        col = f"fwd_ret_{h}d"
        result = compute_ic(feature_matrix, forward_return_col=col, feature_cols=[feature_col])
        decay[h] = result.get(feature_col, {"ic_mean": np.nan, "ic_std": np.nan,
                                             "icir": np.nan, "p_value": np.nan, "n_dates": 0})
    return decay
