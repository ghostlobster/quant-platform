"""
strategies/ensemble_signal.py — Weighted ensemble blending for alpha score dicts.

Combines multiple model score dictionaries into a single ensemble signal by
computing a weighted average across the union of tickers.  Missing tickers in
any individual model's output are treated as a neutral score of 0.0.

Optional adoption-aware weighting (``model_names``) downweights stale models
via ``KnowledgeAdaptionAgent``'s recommendation tag
(``fresh=1.0 / monitor=0.7 / retrain=0.4``) so blended output shrinks away
from stale members without losing coverage.  Weights are re-normalised after
the multiplier so the total blend mass stays at 1.0.
"""
from __future__ import annotations


def _get_model_recommendation(model_name: str) -> str:
    """Return the adoption recommendation tag for ``model_name``.

    Today the ``KnowledgeAdaptionAgent`` only assesses the LGBM baseline +
    regime pickles, so every model name receives the same verdict — the
    overall ML knowledge-state. Once the agent is zoo-aware (see #123) this
    hook becomes per-model. On any failure (import error, agent exception)
    we fail open with ``"fresh"`` so the blend is unaffected.
    """
    try:
        from agents.knowledge_agent import KnowledgeAdaptionAgent

        sig = KnowledgeAdaptionAgent().run({})
        recommendation = (sig.metadata or {}).get("recommendation", "fresh")
        return str(recommendation)
    except Exception:
        return "fresh"


def blend_signals(
    *score_dicts: dict[str, float],
    weights: list[float] | None = None,
    model_names: list[str] | None = None,
) -> dict[str, float]:
    """
    Blend multiple alpha score dicts into a weighted ensemble signal.

    Parameters
    ----------
    *score_dicts : variable number of {ticker: score} dicts where scores are
                  in [-1, 1].  At least one dict is required.
    weights      : blend weights, one per score_dict.  Must sum to a positive
                   value; will be normalised internally.  None → equal weights.
    model_names  : optional model-name labels, one per score_dict. When
                   provided, each weight is multiplied by the model's
                   adoption multiplier from ``KnowledgeAdaptionAgent``
                   (fresh=1.0, monitor=0.7, retrain=0.4) and the result is
                   re-normalised so the blend mass stays at 1.0. When every
                   model is flagged ``retrain`` (all multipliers collapse to
                   zero) we fall back to the unadjusted weights so the blend
                   never silently empties out.

    Returns
    -------
    dict mapping ticker → weighted-average score, clipped to [-1.0, 1.0].
    Empty dict if no score_dicts are provided.

    Raises
    ------
    ValueError if weights is provided but its length differs from score_dicts,
               if model_names is provided with a mismatched length,
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

    if model_names is not None:
        if len(model_names) != n:
            raise ValueError(
                f"model_names length ({len(model_names)}) must match "
                f"number of score dicts ({n})"
            )
        from agents.knowledge_agent import recommendation_multiplier

        # Per-call cache so we don't invoke the agent more than once per
        # unique model name in a single blend call.
        rec_cache: dict[str, str] = {}

        def _rec(name: str) -> str:
            if name not in rec_cache:
                rec_cache[name] = _get_model_recommendation(name)
            return rec_cache[name]

        adjusted = [w[i] * recommendation_multiplier(_rec(model_names[i])) for i in range(n)]
        adjusted_total = sum(adjusted)
        if adjusted_total > 0:
            w = [a / adjusted_total for a in adjusted]
        # If every model collapsed to zero (all retrain with zero base weight),
        # leave w unchanged so the caller still gets a coherent blend.

    # Union of all tickers
    all_tickers: set[str] = set()
    for sd in score_dicts:
        all_tickers.update(sd.keys())

    blended: dict[str, float] = {}
    for ticker in all_tickers:
        score = sum(w[i] * sd.get(ticker, 0.0) for i, sd in enumerate(score_dicts))
        blended[ticker] = max(-1.0, min(1.0, score))

    return blended
