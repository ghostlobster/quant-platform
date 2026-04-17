"""
backtester/combinatorial_cv.py — AFML Ch 12 combinatorial purged CV.

`purged_walk_forward` (AFML Ch 7) yields a single OOS path by sliding a
single test window across the timeline.  Combinatorial-purged CV (Ch 12)
splits the same timeline into ``N`` equal groups and, for every
``C(N, k)`` combination of ``k`` groups, designates that combination as
the test set — producing many more backtest paths from the same data.

The extra paths are exactly what the Deflated Sharpe Ratio (AFML Eq 11.6)
and the Probability of Backtest Overfitting metric (AFML Eq 11.9) need
to estimate tail risk from hyper-parameter search.

Algorithm (AFML 12.4)
---------------------
1. Partition ``[0, n_samples)`` into ``n_splits`` contiguous groups.
2. Each combination of ``n_test_splits`` groups is the test set; the
   complement minus an embargo is the train set.
3. The number of paths emitted is ``C(n_splits, n_test_splits)``.

Embargo is applied *symmetrically* around every test group to prevent
leakage from overlapping forward-return labels.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 12.4.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Iterable, Iterator

import numpy as np
import pandas as pd


def _group_ranges(n_samples: int, n_splits: int) -> list[tuple[int, int]]:
    """Return ``n_splits`` contiguous ``[start, end)`` ranges covering
    ``[0, n_samples)`` as evenly as possible."""
    if n_samples <= 0 or n_splits <= 0:
        return []
    # np.array_split gives groups whose sizes differ by at most 1.
    boundaries = np.array_split(np.arange(n_samples), n_splits)
    return [(int(g[0]), int(g[-1]) + 1) for g in boundaries if len(g) > 0]


def _apply_embargo(
    train_mask: np.ndarray,
    test_ranges: Iterable[tuple[int, int]],
    embargo: int,
    n_samples: int,
) -> np.ndarray:
    """Zero out train bars that fall within ``embargo`` of any test bar.

    Applied on both sides so a backward-looking or forward-looking label
    cannot leak across the train/test boundary.
    """
    if embargo <= 0:
        return train_mask
    out = train_mask.copy()
    for start, end in test_ranges:
        lo = max(0, start - embargo)
        hi = min(n_samples, end + embargo)
        out[lo:hi] = False
    return out


def num_combinatorial_paths(n_splits: int, n_test_splits: int) -> int:
    """Expected number of ``(train, test)`` splits emitted — i.e.
    ``C(n_splits, n_test_splits)``.  Zero if arguments are invalid."""
    if n_splits <= 0 or n_test_splits <= 0 or n_test_splits > n_splits:
        return 0
    return comb(n_splits, n_test_splits)


def combinatorial_purged_splits(
    n_samples: int,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` positional arrays for every
    combination of ``n_test_splits`` groups.

    Parameters
    ----------
    n_samples : total observation count (e.g. ``len(feature_matrix)``).
    n_splits : partition the timeline into this many contiguous groups.
    n_test_splits : size of each test combination (must be ``<= n_splits``).
    embargo : number of bars excluded on each side of every test group.

    Yields
    ------
    ``(train_idx, test_idx)`` ndarrays of ``int`` positional indices.

    Raises
    ------
    ValueError on invalid argument combinations.
    """
    if n_samples <= 0:
        return
    if n_splits <= 1:
        raise ValueError("n_splits must be at least 2")
    if n_test_splits < 1 or n_test_splits > n_splits:
        raise ValueError(
            f"n_test_splits must be in [1, {n_splits}], got {n_test_splits}"
        )
    if n_splits > n_samples:
        raise ValueError(
            f"n_splits ({n_splits}) cannot exceed n_samples ({n_samples})"
        )
    if embargo < 0:
        raise ValueError("embargo must be non-negative")

    groups = _group_ranges(n_samples, n_splits)
    all_idx = np.arange(n_samples)

    for test_group_indices in combinations(range(n_splits), n_test_splits):
        test_ranges = [groups[i] for i in test_group_indices]
        test_mask = np.zeros(n_samples, dtype=bool)
        for start, end in test_ranges:
            test_mask[start:end] = True

        # Train = everything outside test, then purge the embargo bands.
        train_mask = ~test_mask
        train_mask = _apply_embargo(train_mask, test_ranges, embargo, n_samples)

        yield all_idx[train_mask], all_idx[test_mask]


def combinatorial_purged_cv(
    data: pd.DataFrame | int,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Materialise :func:`combinatorial_purged_splits` into a list.

    Accepts either an integer ``n_samples`` or a DataFrame-like whose
    ``len`` is the sample count (a MultiIndex ``(date, ticker)`` is fine;
    positional indices address its rows).  Returns ``[]`` when ``data``
    is empty rather than raising — this keeps caller code simple.
    """
    if isinstance(data, (int, np.integer)):
        n_samples = int(data)
    else:
        if data is None:
            return []
        n_samples = len(data)
        if n_samples == 0:
            return []

    return list(
        combinatorial_purged_splits(
            n_samples=n_samples,
            n_splits=n_splits,
            n_test_splits=n_test_splits,
            embargo=embargo,
        )
    )


def paths_dataframe(
    fold_results: list[dict],
    n_splits: int,
    n_test_splits: int,
) -> pd.DataFrame:
    """Stack per-fold result dicts into a DataFrame keyed by path id.

    Primarily a convenience so downstream consumers (e.g. the Deflated
    Sharpe / PBO metrics planned in issue #72) can read all paths with
    a single ``.loc`` lookup.

    Parameters
    ----------
    fold_results :
        List of result dictionaries in the same order as the splits
        yielded by :func:`combinatorial_purged_splits`.  Each dict
        typically contains ``sharpe``, ``total_return``, etc.
    n_splits, n_test_splits : the parameters the splits were generated
        with.  Used to validate the length of ``fold_results`` and to
        label the rows.

    Returns
    -------
    ``pd.DataFrame`` indexed by integer path id ``0 .. C(N, k) - 1``.
    """
    expected = num_combinatorial_paths(n_splits, n_test_splits)
    if expected == 0 or not fold_results:
        return pd.DataFrame(fold_results)
    if len(fold_results) != expected:
        raise ValueError(
            f"paths_dataframe: expected {expected} fold results "
            f"(C({n_splits}, {n_test_splits})), got {len(fold_results)}"
        )

    df = pd.DataFrame(fold_results)
    df.index = pd.Index(range(len(df)), name="path")
    return df
