"""Market Regime Detector — classifies market conditions using SPY vs 200d SMA and VIX."""
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

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
