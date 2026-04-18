"""
analysis/microstructure.py — AFML Ch 19 microstructural features.

Two daily-bar estimators of informed trading / liquidity that can be
computed without tick-level data:

  * :func:`bvc_buy_fraction` — Bulk Volume Classification (Easley, López
    de Prado, O'Hara 2012): the estimated buy-side fraction of a bar's
    volume, derived from the standardised return via the Normal CDF.
  * :func:`vpin` — Volume-Synchronized Probability of Informed Trading
    over a rolling window of ``window`` daily bars, using BVC to split
    each bar's volume into buy / sell halves.
  * :func:`kyle_lambda` — rolling estimate of Kyle's price-impact
    coefficient ``λ`` regressing ``|r_t|`` on signed dollar volume.

All three are O(n) vectorised pandas operations and return a
date-indexed ``pd.Series`` aligned with the input.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 19.
    Easley, López de Prado & O'Hara, "Flow Toxicity and Liquidity in a
    High-Frequency World," *Review of Financial Studies* 25 (2012).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.logger import get_logger

log = get_logger(__name__)


def bvc_buy_fraction(returns: pd.Series, window: int = 50) -> pd.Series:
    """Estimate the buy-side fraction of each bar's volume from returns.

    For bar ``t`` the BVC rule (Easley et al. 2012) is::

        buy_frac_t = Φ(r_t / (σ_r · √2))

    where ``σ_r`` is the rolling standard deviation of returns over the
    last ``window`` bars. Interpretation: if the bar's return is much
    more positive than its recent volatility allows, the bar was likely
    driven by buyers.

    Parameters
    ----------
    returns : pd.Series
        Simple per-bar returns, date-indexed.
    window  : int, default 50
        Rolling window length for the volatility scaler.

    Returns
    -------
    pd.Series
        Values in ``[0, 1]``. NaN for the first ``window - 1`` bars or
        wherever rolling std is zero/NaN.
    """
    r = pd.Series(returns, dtype=float)
    sigma = r.rolling(window, min_periods=window).std(ddof=0)
    sigma = sigma.replace(0.0, np.nan)
    z = r / (sigma * np.sqrt(2.0))
    frac = pd.Series(norm.cdf(z.to_numpy()), index=r.index)
    frac[z.isna()] = np.nan
    return frac


def vpin(close: pd.Series, volume: pd.Series, window: int = 50) -> pd.Series:
    """VPIN over a rolling window of daily bars using BVC classification.

    Each bar contributes ``|buy_vol - sell_vol| = |2·buy_frac - 1| · V``.
    VPIN is the ratio of that order-flow imbalance to total volume over
    the trailing ``window`` bars — a toxicity / informed-flow proxy.

    Parameters
    ----------
    close  : pd.Series
        Close prices, date-indexed.
    volume : pd.Series
        Share volume per bar, date-indexed, aligned with ``close``.
    window : int, default 50
        Rolling window length (number of bars per VPIN observation).

    Returns
    -------
    pd.Series
        VPIN in ``[0, 1]``. NaN until ``window`` bars have accumulated.
    """
    close = pd.Series(close, dtype=float)
    volume = pd.Series(volume, dtype=float)
    returns = close.pct_change()
    buy_frac = bvc_buy_fraction(returns, window=window)

    imbalance = (2.0 * buy_frac - 1.0).abs() * volume
    roll_imbalance = imbalance.rolling(window, min_periods=window).sum()
    roll_volume = volume.rolling(window, min_periods=window).sum()
    out = roll_imbalance / roll_volume.replace(0.0, np.nan)
    return out


def kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Rolling Kyle's λ from daily OHLCV.

    Estimates the price-impact coefficient ``λ`` via the OLS slope of
    bar returns on signed dollar-volume over a trailing window::

        q_t = sgn(r_t) · p_t · V_t                 # tick-rule order flow
        λ   = Cov(r, q) / Var(q)

    Because ``sgn(r_t)`` matches ``sgn(r_t)`` by construction, this
    recovers ``λ ≈ E[|r|·p·V] / E[(p·V)²]`` — the daily-bar analogue of
    Kyle's λ (AFML §19.4.3). A larger ``λ`` means a given order flow
    moves the price more — i.e. lower liquidity.

    Parameters
    ----------
    close  : pd.Series
        Close prices, date-indexed.
    volume : pd.Series
        Share volume per bar, date-indexed, aligned with ``close``.
    window : int, default 21
        Rolling window (~1 trading month).

    Returns
    -------
    pd.Series
        λ estimates, date-indexed. NaN until ``window`` bars are
        available or when the regressor has zero variance.
    """
    close = pd.Series(close, dtype=float)
    volume = pd.Series(volume, dtype=float)
    returns = close.pct_change()
    signed_dv = np.sign(returns) * volume * close

    cov = returns.rolling(window, min_periods=window).cov(signed_dv)
    var = signed_dv.rolling(window, min_periods=window).var(ddof=0)
    out = cov / var.replace(0.0, np.nan)
    return out
