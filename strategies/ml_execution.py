"""
strategies/ml_execution.py — Execute ML alpha signals via the paper trader.

Translates MLSignal.predict() score dict into paper trading buy/sell orders.
Long the top-N tickers with score > threshold; exit tickers with score < −threshold.

Usage
-----
    from strategies.ml_execution import execute_ml_signals
    from strategies.ml_signal import MLSignal

    scores = MLSignal().predict(tickers, period="6mo")
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)
"""
from __future__ import annotations

from broker.paper_trader import buy, get_portfolio, sell
from data.fetcher import fetch_ohlcv
from utils.logger import get_logger

log = get_logger(__name__)


def execute_ml_signals(
    scores: dict[str, float],
    threshold: float = 0.3,
    max_positions: int = 5,
) -> list[str]:
    """
    Translate alpha scores into paper trading orders.

    Fetches the current market price for each candidate ticker via
    data.fetcher.fetch_ohlcv, then calls broker.paper_trader.buy / sell.

    Parameters
    ----------
    scores        : dict mapping ticker → score in [-1, 1]
    threshold     : minimum score magnitude to act on (default 0.3)
    max_positions : maximum simultaneous long positions (default 5)

    Returns
    -------
    list of action strings, e.g. ["BUY AAPL", "SELL MSFT"]
    Empty list if scores is empty or all scores are within the neutral band.
    """
    if not scores:
        return []

    actions: list[str] = []

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    long_candidates = [(t, s) for t, s in sorted_scores if s > threshold][:max_positions]
    bearish_tickers = {t for t, s in sorted_scores if s < -threshold}
    long_tickers = {t for t, _ in long_candidates}

    # Fetch existing paper positions
    try:
        portfolio_df = get_portfolio()
        existing_positions: set[str] = set(portfolio_df["Ticker"].tolist()) if not portfolio_df.empty else set()
    except Exception as exc:
        log.warning("ml_execution: could not fetch portfolio", error=str(exc))
        existing_positions = set()

    def _latest_price(ticker: str) -> float | None:
        """Return the most recent close price, or None on failure."""
        try:
            df = fetch_ohlcv(ticker, "5d")
            if df is not None and not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as exc:
            log.warning("ml_execution: price fetch failed", ticker=ticker, error=str(exc))
        return None

    # Exit bearish or no-longer-favoured positions
    for ticker in list(existing_positions):
        if ticker in bearish_tickers or ticker not in long_tickers:
            price = _latest_price(ticker)
            if price is None:
                log.warning("ml_execution: skipping sell — no price", ticker=ticker)
                continue
            try:
                portfolio_df = get_portfolio()
                row = portfolio_df[portfolio_df["Ticker"] == ticker]
                shares = float(row["Shares"].iloc[0]) if not row.empty else 0.0
                if shares > 0:
                    sell(ticker, shares, price)
                    actions.append(f"SELL {ticker}")
                    log.info("ml_execution: sold", ticker=ticker, shares=shares, price=price,
                             score=scores.get(ticker, 0.0))
            except Exception as exc:
                log.warning("ml_execution: sell failed", ticker=ticker, error=str(exc))

    # Enter new long positions (1 share each — position sizing handled by Kelly/RL sizer)
    for ticker, score in long_candidates:
        if ticker not in existing_positions:
            price = _latest_price(ticker)
            if price is None:
                log.warning("ml_execution: skipping buy — no price", ticker=ticker)
                continue
            try:
                buy(ticker, 1, price)
                actions.append(f"BUY {ticker}")
                log.info("ml_execution: bought", ticker=ticker, price=price, score=score)
            except Exception as exc:
                log.warning("ml_execution: buy failed", ticker=ticker, error=str(exc))

    return actions
