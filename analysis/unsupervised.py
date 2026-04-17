"""
analysis/unsupervised.py — PCA + k-means risk-factor decomposition.

Surfaces latent risk factors and asset clusters via unsupervised learning:
  * PCA on asset returns yields principal "statistical factors" — the top
    few components typically explain the bulk of variance and look like
    market / sector / style factors.
  * k-means on a correlation-distance matrix groups assets that move
    together.  Useful for pairs discovery and cluster-aware risk budgets.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.), Ch 13.

Dependencies
------------
    scikit-learn >= 1.3.0, numpy, pandas.  Both are hard deps already
    pulled in by the rest of the platform.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Minimum observations per asset before PCA / k-means will run.
_MIN_OBS = 20


@dataclass
class PCAFactors:
    """Output of :func:`pca_risk_factors`.

    Attributes
    ----------
    loadings :
        ``n_assets × n_components`` DataFrame — each column is an
        eigenvector of the returns covariance, indexed by ticker.
    explained_variance :
        Series indexed ``PC1, PC2, …`` giving the fraction of total
        variance explained by each component.
    components :
        ``n_observations × n_components`` DataFrame — the returns
        projected onto the PCA axes (handy for plotting factor
        timeseries).
    """

    loadings: pd.DataFrame
    explained_variance: pd.Series
    components: pd.DataFrame


def pca_risk_factors(
    returns: pd.DataFrame,
    n_components: int = 5,
) -> PCAFactors:
    """Run PCA on a returns matrix and return loadings + explained-variance.

    Parameters
    ----------
    returns :
        DataFrame with tickers as columns and a DatetimeIndex of
        observations (typically daily returns).  Missing values are
        filled with ``0.0`` so the estimator sees a dense matrix.
    n_components :
        Number of principal components to keep.  Automatically clipped
        to ``min(n_assets, n_observations)`` when the requested value
        would exceed the matrix rank.

    Returns
    -------
    :class:`PCAFactors` with `loadings`, `explained_variance` and
    `components`.  An empty :class:`PCAFactors` is returned when the
    input does not have enough rows / columns to fit.
    """
    if returns is None or returns.empty:
        return PCAFactors(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    cleaned = returns.fillna(0.0)
    n_obs, n_assets = cleaned.shape
    if n_obs < _MIN_OBS or n_assets < 2:
        return PCAFactors(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    k = max(1, min(int(n_components), n_assets, n_obs))
    pca = PCA(n_components=k)
    projected = pca.fit_transform(cleaned.values)

    component_names = [f"PC{i + 1}" for i in range(k)]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=cleaned.columns,
        columns=component_names,
    )
    explained = pd.Series(
        np.asarray(pca.explained_variance_ratio_, dtype=float),
        index=component_names,
        name="explained_variance_ratio",
    )
    components = pd.DataFrame(
        projected,
        index=cleaned.index,
        columns=component_names,
    )
    return PCAFactors(loadings=loadings, explained_variance=explained, components=components)


def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """AFML-style distance in correlation space: d = sqrt(0.5 * (1 − ρ)).

    Same metric as :mod:`risk.hrp`; reproduced here instead of importing
    to keep this module free of intra-analysis cross-dependencies.
    """
    arr = corr.to_numpy(copy=True)
    np.fill_diagonal(arr, 1.0)
    return np.sqrt(np.clip(0.5 * (1.0 - arr), 0.0, 1.0))


def cluster_assets(
    returns: pd.DataFrame,
    n_clusters: int = 8,
    random_state: int = 42,
) -> pd.Series:
    """Group assets by k-means on a correlation-distance matrix.

    Parameters
    ----------
    returns :
        DataFrame of asset returns (tickers as columns).
    n_clusters :
        Target cluster count.  Automatically clipped to ``n_assets``.
    random_state :
        Seed forwarded to :class:`~sklearn.cluster.KMeans` for
        deterministic tests.

    Returns
    -------
    pd.Series indexed by ticker, values are integer cluster labels
    ``0 … n_clusters − 1``.  Returns an empty series when the input is
    unusable (fewer than 2 tickers or insufficient observations).
    """
    if returns is None or returns.empty or returns.shape[1] < 2:
        return pd.Series(dtype=int)
    if returns.shape[0] < _MIN_OBS:
        return pd.Series(dtype=int)

    corr = returns.corr().fillna(0.0)
    dist = _correlation_distance(corr)

    k = max(1, min(int(n_clusters), corr.shape[0]))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(dist)
    return pd.Series(labels, index=corr.index, name="cluster").astype(int)


def cluster_members(cluster_labels: pd.Series) -> dict[int, list[str]]:
    """Invert a ticker → cluster series into ``{cluster_id: [tickers]}``."""
    if cluster_labels is None or cluster_labels.empty:
        return {}
    out: dict[int, list[str]] = {}
    for ticker, cid in cluster_labels.items():
        out.setdefault(int(cid), []).append(str(ticker))
    return {k: sorted(v) for k, v in sorted(out.items())}
