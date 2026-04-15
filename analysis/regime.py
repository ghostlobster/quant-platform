"""
analysis/regime.py — Market Regime Detector.

Classifies market conditions using SPY vs 200d SMA and VIX into one of four
states: trending_bull, trending_bear, mean_reverting, high_vol.

Optional LLM fusion: set REGIME_LLM_WEIGHT > 0.0 to blend a macro LLM signal
with the quantitative signal. Requires LLM_PROVIDER to be configured.

ENV vars
--------
    REGIME_LLM_WEIGHT   float 0.0–1.0 (default 0.0 = disabled)
"""
from __future__ import annotations

import json
import os

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

REGIME_STATES: list[str] = [
    "trending_bull",
    "trending_bear",
    "mean_reverting",
    "high_vol",
]

# ── Regime metadata ────────────────────────────────────────────────────────────

REGIME_METADATA: dict[str, dict] = {
    "trending_bull": {
        "description": (
            "SPY above 200d SMA with VIX below 20 — sustained uptrend with low fear."
        ),
        "recommended_strategies": [
            "momentum",
            "SMA crossover",
            "trend following",
            "long equities",
        ],
    },
    "trending_bear": {
        "description": (
            "SPY below 200d SMA with VIX below 20 — downtrend with subdued volatility."
        ),
        "recommended_strategies": [
            "short momentum",
            "pairs trading",
            "inverse ETFs",
        ],
    },
    "mean_reverting": {
        "description": (
            "VIX between 20–30 — elevated but not extreme volatility; range-bound conditions."
        ),
        "recommended_strategies": [
            "RSI mean reversion",
            "pairs trading",
            "range strategies",
        ],
    },
    "high_vol": {
        "description": (
            "VIX above 30 — extreme fear; reduce exposure and halve Kelly fraction."
        ),
        "recommended_strategies": [
            "reduce position sizes",
            "cash",
            "long volatility",
        ],
    },
}


# ── Core classification ────────────────────────────────────────────────────────

def detect_regime(spy_prices: pd.Series, vix_level: float) -> str:
    """Classify the current market regime.

    Priority (VIX overrides SPY position when >= 20):
      VIX > 30                          → 'high_vol'
      20 <= VIX <= 30                   → 'mean_reverting'
      VIX < 20 and SPY > 200d SMA      → 'trending_bull'
      VIX < 20 and SPY <= 200d SMA     → 'trending_bear'

    Args:
        spy_prices: Series of SPY closing prices (at least 200 points recommended).
        vix_level:  Current VIX index value.

    Returns:
        One of: 'trending_bull' | 'trending_bear' | 'mean_reverting' | 'high_vol'
    """
    n = len(spy_prices)
    if n < 200:
        logger.warning(
            "detect_regime: only %d price points; 200d SMA will use all available data.", n
        )

    # VIX-based regimes take priority over trend direction
    if vix_level > 30:
        return "high_vol"

    if vix_level >= 20:
        return "mean_reverting"

    # VIX < 20 — determine trend direction relative to 200d SMA
    lookback = min(200, n)
    sma200 = float(spy_prices.iloc[-lookback:].mean())
    current_price = float(spy_prices.iloc[-1])

    logger.debug(
        "detect_regime: price=%.2f sma200=%.2f vix=%.2f", current_price, sma200, vix_level
    )

    if current_price > sma200:
        return "trending_bull"
    return "trending_bear"


def kelly_regime_multiplier(regime: str) -> float:
    """Return Kelly fraction multiplier for the given regime.

    Returns 0.5 for 'high_vol' (halve position sizes), 1.0 for all other regimes.
    """
    return 0.5 if regime == "high_vol" else 1.0


# ── Live data fetch ────────────────────────────────────────────────────────────

def get_live_regime() -> dict:
    """Fetch live SPY and ^VIX data from yfinance and return regime classification.

    Returns:
        dict with keys: regime, spy_price, spy_sma200, vix,
                        description, recommended_strategies
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for get_live_regime()") from exc

    logger.info("get_live_regime: fetching SPY and ^VIX from yfinance")

    spy_hist = yf.download("SPY", period="1y", progress=False, auto_adjust=True)
    if spy_hist.empty:
        raise RuntimeError("Failed to fetch SPY data from yfinance")

    spy_prices: pd.Series = spy_hist["Close"].squeeze()

    vix_hist = yf.download("^VIX", period="5d", progress=False, auto_adjust=True)
    if vix_hist.empty:
        raise RuntimeError("Failed to fetch ^VIX data from yfinance")

    vix_level = float(vix_hist["Close"].iloc[-1])
    regime = detect_regime(spy_prices, vix_level)

    lookback = min(200, len(spy_prices))
    sma200 = float(spy_prices.iloc[-lookback:].mean())
    spy_price = float(spy_prices.iloc[-1])

    meta = REGIME_METADATA[regime]
    logger.info(
        "get_live_regime: regime=%s spy=%.2f sma200=%.2f vix=%.2f",
        regime, spy_price, sma200, vix_level,
    )

    return {
        "regime": regime,
        "spy_price": spy_price,
        "spy_sma200": sma200,
        "vix": vix_level,
        "description": meta["description"],
        "recommended_strategies": list(meta["recommended_strategies"]),
    }


# ── LLM fusion helpers ─────────────────────────────────────────────────────────

def _build_macro_prompt(spy_price: float, spy_sma200: float, vix: float) -> str:
    """Build a structured prompt asking the LLM to classify the market regime."""
    trend = "above" if spy_price > spy_sma200 else "below"
    return (
        "You are a quantitative macro analyst. Classify the current US equity market regime "
        "based on the data below. Respond ONLY with valid JSON — no explanation.\n\n"
        f"SPY price: {spy_price:.2f}\n"
        f"SPY 200-day SMA: {spy_sma200:.2f}  (SPY is {trend} its 200d SMA)\n"
        f"VIX: {vix:.2f}\n\n"
        'Return JSON with exactly two keys: "regime" and "confidence".\n'
        f'"regime" must be one of: {REGIME_STATES}.\n'
        '"confidence" must be a float between 0.0 and 1.0.\n'
        'Example: {"regime": "trending_bull", "confidence": 0.82}'
    )


def _parse_llm_regime(llm_response: str) -> tuple[str, float]:
    """
    Parse the LLM's JSON regime response.

    Returns (regime_label, confidence). Falls back to ('high_vol', 0.5) on
    any parse failure so the system degrades gracefully.
    """
    _FALLBACK = ("high_vol", 0.5)
    try:
        # Strip markdown code fences if present
        text = llm_response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        data = json.loads(text)
        regime = str(data.get("regime", "")).strip()
        if regime not in REGIME_STATES:
            logger.warning("LLM returned unknown regime %r; falling back", regime)
            return _FALLBACK
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        return regime, confidence
    except Exception as exc:
        logger.warning("Failed to parse LLM regime response: %s  raw=%r", exc, llm_response[:200])
        return _FALLBACK


def _blend_regimes(
    quant_regime: str,
    llm_regime: str,
    llm_confidence: float,
    weight: float,
) -> str:
    """
    Blend a quantitative regime signal with an LLM regime signal.

    At weight=0.0 returns quant_regime unchanged.
    At weight=1.0 returns llm_regime (scaled by llm_confidence).
    In between, builds probability vectors over REGIME_STATES and takes argmax.
    """
    if weight <= 0.0:
        return quant_regime
    if weight >= 1.0:
        return llm_regime

    # Quant signal: full confidence in its single prediction
    q_vec = [1.0 if r == quant_regime else 0.0 for r in REGIME_STATES]
    # LLM signal: llm_confidence on the predicted regime, remainder spread uniformly
    remainder = (1.0 - llm_confidence) / (len(REGIME_STATES) - 1)
    l_vec = [
        llm_confidence if r == llm_regime else remainder
        for r in REGIME_STATES
    ]
    # Weighted blend
    blended = [
        (1.0 - weight) * q_val + weight * llm_val
        for q_val, llm_val in zip(q_vec, l_vec)
    ]
    best_idx = blended.index(max(blended))
    return REGIME_STATES[best_idx]


def get_live_regime_with_llm(llm_weight: float | None = None) -> dict:
    """
    Fetch live SPY/VIX data, classify regime, and optionally blend with an LLM
    macro signal.

    Parameters
    ----------
    llm_weight : float or None
        Weight for the LLM signal in [0.0, 1.0].  None reads REGIME_LLM_WEIGHT
        env var (default 0.0 = disabled — identical to get_live_regime()).

    Returns
    -------
    dict with all keys from get_live_regime() plus, when LLM is enabled:
        llm_regime      : str  — regime label returned by LLM
        llm_confidence  : float
        llm_weight      : float
        quant_regime    : str  — original price-based regime before blending
    """
    result = get_live_regime()

    weight = llm_weight if llm_weight is not None else float(
        os.environ.get("REGIME_LLM_WEIGHT", "0.0")
    )

    if weight <= 0.0:
        return result

    try:
        from providers.llm import get_llm
        llm = get_llm()
        prompt = _build_macro_prompt(
            result["spy_price"], result["spy_sma200"], result["vix"]
        )
        raw = llm.complete(prompt)
        llm_regime, llm_conf = _parse_llm_regime(raw)
        quant_regime = result["regime"]
        blended = _blend_regimes(quant_regime, llm_regime, llm_conf, weight)

        result["quant_regime"] = quant_regime
        result["llm_regime"] = llm_regime
        result["llm_confidence"] = llm_conf
        result["llm_weight"] = weight
        result["regime"] = blended
        # Update description/strategies for the blended regime
        meta = REGIME_METADATA[blended]
        result["description"] = meta["description"]
        result["recommended_strategies"] = list(meta["recommended_strategies"])

        logger.info(
            "LLM regime fusion: quant=%s llm=%s(%.2f) weight=%.2f → %s",
            quant_regime, llm_regime, llm_conf, weight, blended,
        )
    except Exception as exc:
        logger.warning("LLM regime fusion failed, using quant regime: %s", exc)

    return result
