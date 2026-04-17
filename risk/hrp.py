"""
risk/hrp.py — Hierarchical Risk Parity portfolio allocation.

Implements López de Prado (AFML Ch 16)'s HRP algorithm: allocates capital
by recursively bisecting a quasi-diagonalised correlation matrix instead
of inverting the covariance matrix.  Avoids Markowitz's sensitivity to
ill-conditioned covariance matrices and typically produces more stable
out-of-sample weights.

Algorithm
---------
1. Distance matrix:        d_ij = sqrt(0.5 * (1 - corr_ij))
2. Hierarchical clustering: scipy.cluster.hierarchy.linkage
3. Quasi-diagonalisation:   re-order assets so correlated ones are adjacent
4. Recursive bisection:     split each cluster in half, allocate via
                            inverse-variance weighting

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 16.4.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from risk.markowitz import OptimalPortfolio


def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """AFML Ch 16 distance metric: d_ij = sqrt(0.5 * (1 - corr_ij))."""
    # Clip to [-1, 1] to avoid sqrt(negative) from tiny float noise.
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    return dist


def _quasi_diag(link: np.ndarray) -> list[int]:
    """Re-order the linkage matrix so correlated items sit adjacent.

    Follows AFML Code Snippet 16.2 verbatim (apart from int casts).
    """
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]

    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]          # clusters that must expand
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df_right = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df_right])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])

    return sort_ix.tolist()


def _inverse_variance_weights(cov_slice: pd.DataFrame) -> pd.Series:
    """Inverse-variance sub-weights inside a cluster (AFML Eq 16.3)."""
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp /= ivp.sum()
    return pd.Series(ivp, index=cov_slice.index)


def _cluster_variance(cov: pd.DataFrame, items: list) -> float:
    """Variance of a sub-portfolio using inverse-variance sub-weights."""
    cov_slice = cov.loc[items, items]
    w = _inverse_variance_weights(cov_slice).values.reshape(-1, 1)
    return float((w.T @ cov_slice.values @ w).item())


def _recursive_bisection(cov: pd.DataFrame, sorted_tickers: list) -> pd.Series:
    """Top-down recursive bisection weighting (AFML Code Snippet 16.4)."""
    weights = pd.Series(1.0, index=sorted_tickers)
    clusters: list[list] = [sorted_tickers]

    while clusters:
        next_clusters: list[list] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            total = var_left + var_right
            # alpha is the fraction allocated to the LEFT cluster.  Higher
            # variance on the right → more to the left (and vice-versa).
            alpha = 1.0 - var_left / total if total > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    return weights


def hrp_weights(
    returns: pd.DataFrame,
    method: str = "single",
) -> pd.Series:
    """Compute Hierarchical Risk Parity weights from a returns DataFrame.

    Parameters
    ----------
    returns : pd.DataFrame
        Columns are tickers, rows are time-ordered returns.  At least two
        assets and two observations are required.
    method : str
        Linkage method forwarded to ``scipy.cluster.hierarchy.linkage``.
        Defaults to ``"single"``, matching the AFML reference
        implementation.

    Returns
    -------
    pd.Series indexed by ticker, weights summing to 1.0.  Single-asset
    input returns a one-element series with weight 1.0.  Empty or
    degenerate input returns an empty series.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float)

    n = returns.shape[1]
    if n == 0:
        return pd.Series(dtype=float)
    if n == 1:
        return pd.Series([1.0], index=returns.columns)

    corr = returns.corr().fillna(0.0)
    cov = returns.cov().fillna(0.0)

    # Guard against singular correlation: clamp diagonal to exactly 1.
    # pandas >=3.0 returns read-only .values views, so work on a copy.
    corr_arr = corr.to_numpy(copy=True)
    np.fill_diagonal(corr_arr, 1.0)
    corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)

    dist = _correlation_distance(corr)
    # squareform requires a strictly-symmetric zero-diagonal matrix; tiny
    # float noise can break this, so we explicitly zero the diagonal.
    dist_values = dist.to_numpy(copy=True)
    np.fill_diagonal(dist_values, 0.0)
    condensed = squareform(dist_values, checks=False)

    try:
        link = linkage(condensed, method=method)
    except ValueError:
        # Degenerate case: fall back to equal weights so callers don't crash.
        return pd.Series(1.0 / n, index=returns.columns)

    sort_indices = _quasi_diag(link)
    sorted_tickers = [returns.columns[i] for i in sort_indices]

    weights = _recursive_bisection(cov, sorted_tickers)
    # Re-align to the input column order so callers can reason about order.
    return weights.reindex(returns.columns).astype(float)


def get_hrp_portfolio(
    price_data: dict,
    risk_free_rate: float = 0.05,
    method: str = "single",
) -> Optional[OptimalPortfolio]:
    """Return an HRP-allocated ``OptimalPortfolio`` for the given price data.

    Mirrors ``risk.markowitz.get_max_sharpe_portfolio`` so the UI can
    swap the two without special-casing.  Expected return and volatility
    are computed on the resulting weights using annualised mean / cov of
    daily returns.
    """
    if price_data is None or len(price_data) == 0:
        return None
    if len(price_data) < 2:
        # Single-asset portfolio trivially has weight 1 — keep the shape.
        ticker = next(iter(price_data.keys()))
        series = price_data[ticker]
        if series is None or len(series) < 2:
            return None
        daily_ret = series.pct_change().dropna()
        exp_ret = float(daily_ret.mean() * 252)
        exp_vol = float(daily_ret.std() * np.sqrt(252))
        sharpe = (exp_ret - risk_free_rate) / exp_vol if exp_vol > 0 else 0.0
        return OptimalPortfolio(
            weights={ticker: 1.0},
            expected_return=exp_ret,
            expected_volatility=exp_vol,
            sharpe_ratio=float(sharpe),
        )

    df = pd.DataFrame(price_data).pct_change().dropna()
    if df.empty or df.shape[1] < 2:
        return None

    weights = hrp_weights(df, method=method)
    if weights.empty:
        return None

    mean_returns = df.mean() * 252
    cov_matrix = df.cov() * 252

    w_vec = weights.reindex(df.columns).fillna(0.0).values
    exp_ret = float(w_vec @ mean_returns.values)
    exp_vol = float(np.sqrt(w_vec @ cov_matrix.values @ w_vec))
    sharpe = (exp_ret - risk_free_rate) / exp_vol if exp_vol > 0 else 0.0

    return OptimalPortfolio(
        weights={t: float(weights[t]) for t in df.columns},
        expected_return=exp_ret,
        expected_volatility=exp_vol,
        sharpe_ratio=float(sharpe),
    )
