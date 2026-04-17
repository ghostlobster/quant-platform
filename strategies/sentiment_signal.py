"""
strategies/sentiment_signal.py — Cross-sectional sentiment alpha signal.

Jansen *ML for Algorithmic Trading* (Ch 14) uses average news/social
sentiment per ticker as a standalone alpha signal that complements
price-based models.  This module wraps the platform's existing
``providers.sentiment`` layer so sentiment scores can be blended into
the ensemble alongside LGBM / Ridge / Bayesian.

The heavy lifting (fetching headlines, running VADER, caching in
SQLite) lives in the adapters — here we just:

  * pull the current per-ticker sentiment from the configured provider,
  * z-score cross-sectionally across the universe,
  * clip to ``[-1, 1]`` so the blender's invariants hold.

Gracefully no-ops (``{ticker: 0.0}``) when the sentiment provider
isn't configured or fails at runtime; the ensemble just sees a neutral
4th source.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 14.6.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


def fetch_sentiment_scores(
    tickers: list[str],
    lookback_hours: int = 24,
    provider_name: Optional[str] = None,
) -> dict[str, float]:
    """Return raw per-ticker sentiment in ``[-1, 1]``.

    Parameters
    ----------
    tickers :
        Universe to score.
    lookback_hours :
        Hours of history to aggregate.  Defaults to a 24-hour window —
        enough to catch overnight news for the next session.
    provider_name :
        Override for the ``SENTIMENT_PROVIDER`` env var.  ``None`` uses
        the configured default.

    Returns
    -------
    ``dict[ticker, float]`` with values in ``[-1, 1]``.  Tickers whose
    score could not be fetched get ``0.0`` (neutral).  Empty dict when
    ``tickers`` is empty.
    """
    if not tickers:
        return {}

    try:
        from providers.sentiment import get_sentiment
        provider = get_sentiment(provider_name)
    except Exception as exc:
        log.info(
            "sentiment_signal: provider unavailable, returning neutral scores",
            error=str(exc),
        )
        return {t: 0.0 for t in tickers}

    scores: dict[str, float] = {}
    for ticker in tickers:
        try:
            raw = provider.ticker_sentiment(ticker, lookback_hours=lookback_hours)
            scores[ticker] = float(np.clip(raw, -1.0, 1.0))
        except Exception as exc:
            log.warning(
                "sentiment_signal: ticker lookup failed",
                ticker=ticker, error=str(exc),
            )
            scores[ticker] = 0.0
    return scores


def sentiment_alpha_scores(
    tickers: list[str],
    lookback_hours: int = 24,
    provider_name: Optional[str] = None,
) -> dict[str, float]:
    """Ranked alpha scores derived from cross-sectional sentiment z-score.

    Flow:

    1. Fetch raw per-ticker sentiment via :func:`fetch_sentiment_scores`.
    2. Z-score across the universe so scores are mean-zero,
       unit-variance — makes them compatible with the other
       model-output ranges the ensemble expects.
    3. Clip to ``[-1, 1]``.

    Returns ``{ticker: score}``.  Universes where every raw score is
    zero or identical collapse to ``{ticker: 0.0}`` so the ensemble
    treats sentiment as neutral rather than spuriously confident.
    """
    raw = fetch_sentiment_scores(tickers, lookback_hours=lookback_hours,
                                  provider_name=provider_name)
    if not raw:
        return {}

    tickers_ordered = list(raw.keys())
    values = np.array([raw[t] for t in tickers_ordered], dtype=float)

    std = float(values.std())
    if std == 0.0:
        return {t: 0.0 for t in tickers_ordered}

    z = (values - values.mean()) / std
    z = np.clip(z, -1.0, 1.0)
    return {t: float(z[i]) for i, t in enumerate(tickers_ordered)}
