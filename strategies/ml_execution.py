"""
strategies/ml_execution.py — Execute ML alpha signals via the broker provider.

Translates an MLSignal.predict() score dict into live broker orders by routing
through providers.broker.get_broker(). Works with any configured broker
(paper/alpaca/ibkr/schwab).

Position sizing
---------------
Orders are sized via a conservative Kelly fraction scaled by:
  * regime multiplier (0.5x in high-vol regimes)
  * |score| (higher-conviction names get larger allocations)

Override the Kelly baseline via env vars ``ML_KELLY_WIN_RATE`` (default 0.55),
``ML_KELLY_AVG_WIN`` (0.03) and ``ML_KELLY_AVG_LOSS`` (0.02).

Usage
-----
    from strategies.ml_execution import execute_ml_signals
    from strategies.ml_signal import MLSignal

    scores = MLSignal().predict(tickers, period="6mo")
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)
"""
from __future__ import annotations

import math
import os
from typing import Optional

from analysis.regime import kelly_regime_multiplier
from data.fetcher import fetch_ohlcv
from providers.broker import BrokerProvider, get_broker
from risk.kelly import kelly_fraction
from utils.logger import get_logger

log = get_logger(__name__)


def _kelly_baseline() -> float:
    """Read Kelly priors from env and return the capped fraction (0–0.25)."""
    win_rate = float(os.environ.get("ML_KELLY_WIN_RATE", "0.55"))
    avg_win = float(os.environ.get("ML_KELLY_AVG_WIN", "0.03"))
    avg_loss = float(os.environ.get("ML_KELLY_AVG_LOSS", "0.02"))
    return kelly_fraction(win_rate, avg_win, avg_loss)


def _current_regime() -> str:
    """Return live regime name, or 'trending_bull' on failure (neutral mult)."""
    try:
        from analysis.regime import get_live_regime
        return str(get_live_regime().get("regime", "trending_bull"))
    except Exception as exc:
        log.warning("ml_execution: regime lookup failed", error=str(exc))
        return "trending_bull"


def _knowledge_gate(regime: str) -> tuple[float, str, bool]:
    """Ask the KnowledgeAdaptionAgent whether the pickled models are fit to trade.

    Returns ``(multiplier, recommendation, skip_all)``.  ``skip_all`` is True
    when the baseline pickle is missing entirely — without a model we should
    not place trades even if a caller passed in synthetic scores.
    """
    try:
        from agents.knowledge_agent import KnowledgeAdaptionAgent, recommendation_multiplier
    except Exception as exc:
        log.debug("ml_execution: knowledge agent import failed", error=str(exc))
        return 1.0, "fresh", False

    # Pull the rolling live IC so the agent's IC-degradation branch can fire
    # when alpha has collapsed even though the pickle is still fresh (#115).
    live_ic: float | None = None
    try:
        from analysis.live_ic import rolling_live_ic

        live_ic = rolling_live_ic("lgbm_alpha")
    except Exception as exc:
        log.debug("ml_execution: rolling_live_ic failed", error=str(exc))

    try:
        sig = KnowledgeAdaptionAgent().run(
            {"regime": regime, "live_ic": live_ic},
        )
    except Exception as exc:
        log.debug("ml_execution: knowledge agent run failed", error=str(exc))
        return 1.0, "fresh", False

    meta = sig.metadata or {}
    rec = str(meta.get("recommendation") or "fresh")
    mult = recommendation_multiplier(rec)
    skip_all = bool(rec == "retrain" and meta.get("baseline_age_days") is None)
    if rec != "fresh":
        log.warning(
            "ml_execution: knowledge gate",
            recommendation=rec,
            multiplier=mult,
            reason=sig.reasoning,
        )
    return mult, rec, skip_all


def _size_order(
    equity: float,
    price: float,
    score: float,
    kelly_base: float,
    regime_mult: float,
) -> int:
    """Compute integer share quantity for a new long position."""
    if equity <= 0 or price <= 0:
        return 0
    target_notional = equity * kelly_base * regime_mult * abs(score)
    qty = int(math.floor(target_notional / price))
    return max(qty, 1)


def _latest_price(ticker: str) -> Optional[float]:
    """Return the most recent close price, or None on failure."""
    try:
        df = fetch_ohlcv(ticker, "5d")
        if df is not None and not df.empty:
            return float(df["Close"].iloc[-1])
    except Exception as exc:
        log.warning("ml_execution: price fetch failed", ticker=ticker, error=str(exc))
    return None


def execute_ml_signals(
    scores: dict[str, float],
    threshold: float = 0.3,
    max_positions: int = 5,
    broker: Optional[BrokerProvider] = None,
) -> list[str]:
    """
    Translate alpha scores into broker orders.

    Longs the top ``max_positions`` tickers with ``score > threshold``; exits
    any held ticker whose score falls below ``-threshold`` or drops out of the
    top-N long candidates. Position sizing uses Kelly × regime × |score|.

    Parameters
    ----------
    scores        : dict mapping ticker → score in [-1, 1]
    threshold     : minimum |score| to act on (default 0.3)
    max_positions : max simultaneous long positions (default 5)
    broker        : optional BrokerProvider (defaults to ``get_broker()``)

    Returns
    -------
    list of action strings, e.g. ``["BUY AAPL x12", "SELL MSFT x8"]``.
    Empty when no scores clear the neutral band.
    """
    if not scores:
        return []

    # Persist scored rows to live_predictions before any filtering so the
    # rolling IC estimator sees the unconditioned distribution — recording
    # only traded tickers would bias the IC toward high-conviction names.
    # Wrapped in try/except so live-IC plumbing never breaks the trading path.
    try:
        from analysis.live_ic import record_predictions

        record_predictions(scores, model_name="lgbm_alpha", horizon_d=5)
    except Exception as exc:
        log.warning("ml_execution: record_predictions failed", error=str(exc))

    broker = broker or get_broker()
    actions: list[str] = []

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    long_candidates = [(t, s) for t, s in sorted_scores if s > threshold][:max_positions]
    bearish_tickers = {t for t, s in sorted_scores if s < -threshold}
    long_tickers = {t for t, _ in long_candidates}

    try:
        positions = broker.get_positions()
        existing: dict[str, float] = {
            str(p["symbol"]): float(p.get("qty", 0.0)) for p in positions
        }
    except Exception as exc:
        log.warning("ml_execution: could not fetch positions", error=str(exc))
        existing = {}

    # Exit bearish or no-longer-favoured positions.
    for ticker, held_qty in list(existing.items()):
        if held_qty <= 0:
            continue
        if ticker not in bearish_tickers and ticker in long_tickers:
            continue
        price = _latest_price(ticker)
        if price is None:
            log.warning("ml_execution: skipping sell — no price", ticker=ticker)
            continue
        try:
            broker.place_order(ticker, held_qty, "sell", order_type="market")
            actions.append(f"SELL {ticker} x{int(held_qty)}")
            log.info(
                "ml_execution: sold",
                ticker=ticker, qty=held_qty, price=price,
                score=scores.get(ticker, 0.0),
            )
        except Exception as exc:
            log.warning("ml_execution: sell failed", ticker=ticker, error=str(exc))

    if not long_candidates:
        return actions

    # Fetch account equity + Kelly/regime sizing inputs once for all buys.
    try:
        account = broker.get_account_info() or {}
        equity = float(account.get("equity") or account.get("cash") or 0.0)
    except Exception as exc:
        log.warning("ml_execution: could not fetch account info", error=str(exc))
        equity = 0.0

    kelly_base = _kelly_baseline()
    regime = _current_regime()
    regime_mult = kelly_regime_multiplier(regime)
    knowledge_mult, knowledge_rec, skip_all = _knowledge_gate(regime)
    if skip_all:
        log.warning(
            "ml_execution: skipping all buys — baseline model missing",
            recommendation=knowledge_rec,
        )
        return actions
    kelly_base *= knowledge_mult

    # Enter new long positions sized by Kelly × regime × |score|.
    for ticker, score in long_candidates:
        if ticker in existing and existing[ticker] > 0:
            continue
        price = _latest_price(ticker)
        if price is None:
            log.warning("ml_execution: skipping buy — no price", ticker=ticker)
            continue
        qty = _size_order(equity, price, score, kelly_base, regime_mult)
        if qty <= 0:
            log.info("ml_execution: skipping buy — zero size", ticker=ticker, equity=equity)
            continue
        try:
            broker.place_order(ticker, qty, "buy", order_type="market")
            actions.append(f"BUY {ticker} x{qty}")
            log.info(
                "ml_execution: bought",
                ticker=ticker, qty=qty, price=price, score=score,
                kelly=kelly_base, regime=regime, regime_mult=regime_mult,
            )
        except Exception as exc:
            log.warning("ml_execution: buy failed", ticker=ticker, error=str(exc))

    return actions
