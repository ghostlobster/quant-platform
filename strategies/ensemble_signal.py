"""
strategies/ensemble_signal.py — Weighted ensemble blending for alpha score dicts.

Combines multiple model score dictionaries into a single ensemble signal by
computing a weighted average across the union of tickers.  Missing tickers in
any individual model's output are treated as a neutral score of 0.0.
"""
from __future__ import annotations


def blend_signals(
    *score_dicts: dict[str, float],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """
    Blend multiple alpha score dicts into a weighted ensemble signal.

    Parameters
    ----------
    *score_dicts : variable number of {ticker: score} dicts where scores are
                  in [-1, 1].  At least one dict is required.
    weights      : blend weights, one per score_dict.  Must sum to a positive
                   value; will be normalised internally.  None → equal weights.

    Returns
    -------
    dict mapping ticker → weighted-average score, clipped to [-1.0, 1.0].
    Empty dict if no score_dicts are provided.

    Raises
    ------
    ValueError if weights is provided but its length differs from score_dicts,
               or if all weights are zero.
    """
    if not score_dicts:
        return {}

    n = len(score_dicts)

    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(
                f"weights length ({len(weights)}) must match number of score dicts ({n})"
            )
        total = sum(weights)
        if total == 0:
            raise ValueError("weights must not all be zero")
        w = [wt / total for wt in weights]

    # Union of all tickers
    all_tickers: set[str] = set()
    for sd in score_dicts:
        all_tickers.update(sd.keys())

    blended: dict[str, float] = {}
    for ticker in all_tickers:
        score = sum(w[i] * sd.get(ticker, 0.0) for i, sd in enumerate(score_dicts))
        blended[ticker] = max(-1.0, min(1.0, score))

    return blended
