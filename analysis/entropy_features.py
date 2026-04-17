"""
analysis/entropy_features.py — AFML Ch 18 entropy-based features.

Moments (mean, variance, skew, kurtosis) miss structure that simple
transformations surface: whether returns *repeat* the same pattern,
whether they're drawn from a quasi-random sequence, whether bursts of
predictability are interleaved with noise.  AFML Ch 18 proposes three
estimators that operate on symbolised return series:

  * :func:`plug_in_entropy`      — classical Shannon estimator.
  * :func:`lempel_ziv_entropy`   — compression-ratio proxy.
  * :func:`konto_entropy`        — Kontoyiannis 1998 best-in-class
                                   estimator for stationary ergodic
                                   sources.

All three take a sequence of floats, symbolise it into ``n_bins``
equal-width bins, and return a non-negative float.  Higher values mean
more informationally rich (closer to iid uniform); lower values mean
more repetition / compressibility.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 18.3-18.7.
"""
from __future__ import annotations

from collections import Counter
from math import log

import numpy as np
import pandas as pd


def _symbolise(series, n_bins: int) -> str:
    """Bin ``series`` into ``n_bins`` equal-width buckets → string of digits.

    Returns an empty string for empty / all-NaN input.  When the range
    is zero (constant series) everything collapses to a single symbol,
    which then yields zero entropy — the correct answer.
    """
    arr = np.asarray(pd.Series(series).dropna().to_numpy(dtype=float))
    if arr.size == 0:
        return ""
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return "0" * arr.size
    edges = np.linspace(lo, hi, int(n_bins) + 1)
    # np.digitize gives 1..n_bins; subtract 1 to zero-index, clip the
    # final right-edge.
    raw = np.clip(np.digitize(arr, edges[1:-1], right=False), 0, n_bins - 1)
    # Prefer hex-ish single-character symbols for readable strings.
    return "".join(str(int(s) % 10) for s in raw)


def plug_in_entropy(series, n_bins: int = 10) -> float:
    """Classical Shannon (plug-in / ML) entropy estimator.

    Symbolises the series into ``n_bins`` and returns
    ``-Σ p_i log p_i`` using natural logarithm.  Deterministic inputs
    return ``0.0``; uniform inputs approach ``log(n_bins)``.
    """
    s = _symbolise(series, n_bins=n_bins)
    if not s:
        return 0.0
    counts = Counter(s)
    total = float(sum(counts.values()))
    return float(-sum((c / total) * log(c / total) for c in counts.values()))


def _lz_distinct_substring_count(s: str) -> int:
    """Greedy Lempel-Ziv factorisation: count the distinct phrases."""
    if not s:
        return 0
    dictionary: set[str] = set()
    count = 0
    buf = ""
    for ch in s:
        buf += ch
        if buf not in dictionary:
            dictionary.add(buf)
            count += 1
            buf = ""
    if buf:
        count += 1     # trailing partial phrase
    return count


def lempel_ziv_entropy(series, n_bins: int = 10) -> float:
    """Entropy estimated from LZ compression ratio.

    Symbolises ``series`` into ``n_bins`` then returns ``W / n`` where
    ``W`` is the number of distinct phrases in the LZ parsing and ``n``
    is the input length.  Range ``[0, 1]``: lower = more compressible,
    higher = more random-looking.
    """
    s = _symbolise(series, n_bins=n_bins)
    if not s:
        return 0.0
    return _lz_distinct_substring_count(s) / float(len(s))


def _kontoyiannis_match_length(s: str, i: int) -> int:
    """Longest prefix of ``s[i:]`` that matches somewhere in ``s[:i]``, plus 1.

    AFML Eq 18.8 — implementation matches López de Prado's reference
    code (López de Prado 2018).  Runs in ``O(n²)`` worst-case which is
    fine for per-window entropy features on daily return series.
    """
    n = len(s)
    # Longest match length starting at position i.
    max_match = 0
    for j in range(1, min(n - i, i) + 1):
        if s[i : i + j] in s[:i]:
            if j > max_match:
                max_match = j
        else:
            break
    return max_match + 1


def konto_entropy(series, n_bins: int = 10) -> float:
    """Kontoyiannis 1998 entropy estimator.

    ``H ≈ 1 / ((1 / (n - 1)) Σ L_i / log₂(i + 1))`` where ``L_i`` is the
    match-length at position ``i`` (AFML Eq 18.8).  Returns a non-
    negative float; larger = more entropy.  For inputs shorter than 4
    symbols returns 0.0 (too short for the match-length trick to be
    meaningful).
    """
    s = _symbolise(series, n_bins=n_bins)
    n = len(s)
    if n < 4:
        return 0.0

    sum_match = 0.0
    count = 0
    for i in range(1, n):
        L = _kontoyiannis_match_length(s, i)
        # log2(i + 1) so the denominator is non-zero.
        sum_match += L / log(i + 1, 2)
        count += 1
    if count == 0 or sum_match == 0:
        return 0.0
    avg = sum_match / count
    if avg <= 0:
        return 0.0
    return float(1.0 / avg)


def entropy_features(
    returns: pd.Series,
    n_bins: int = 10,
) -> dict[str, float]:
    """Convenience: compute all three entropy features on one return series.

    Returns a dict with keys ``plug_in``, ``lempel_ziv``, ``konto`` —
    handy for the feature-engineering layer to populate three columns
    in a single call.
    """
    return {
        "plug_in":     plug_in_entropy(returns, n_bins=n_bins),
        "lempel_ziv":  lempel_ziv_entropy(returns, n_bins=n_bins),
        "konto":       konto_entropy(returns, n_bins=n_bins),
    }
