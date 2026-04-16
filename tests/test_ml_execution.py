"""
tests/test_ml_execution.py — Unit tests for strategies/ml_execution.py.

All paper_trader calls and fetch_ohlcv calls are mocked — no DB or network access.
"""
import os
import sys
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.ml_execution import execute_ml_signals


def _make_portfolio(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["Ticker", "Shares"])
    return pd.DataFrame({"Ticker": tickers, "Shares": [10.0] * len(tickers)})


def _scores(mapping: dict[str, float]) -> dict[str, float]:
    return mapping


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_buys_top_long_candidates(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """High positive scores trigger BUY orders for tickers not already held."""
    mock_get_portfolio.return_value = _make_portfolio([])
    mock_fetch.return_value = pd.DataFrame({"Close": [150.0]})

    scores = {"AAPL": 0.8, "MSFT": 0.6, "GOOG": -0.5}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    assert "BUY AAPL" in actions
    assert "BUY MSFT" in actions
    assert "BUY GOOG" not in actions
    mock_buy.assert_any_call("AAPL", 1, 150.0)
    mock_buy.assert_any_call("MSFT", 1, 150.0)
    mock_sell.assert_not_called()


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_sells_bearish_existing_position(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """Bearish score (< -threshold) for a held position triggers SELL."""
    mock_get_portfolio.side_effect = [
        _make_portfolio(["AAPL"]),   # first call: existing positions
        _make_portfolio(["AAPL"]),   # second call: fetching shares for sell
    ]
    mock_fetch.return_value = pd.DataFrame({"Close": [200.0]})

    scores = {"AAPL": -0.7, "MSFT": 0.9}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    assert "SELL AAPL" in actions
    mock_sell.assert_called_once_with("AAPL", 10.0, 200.0)


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_no_action_within_neutral_band(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """Scores within [-threshold, threshold] produce no orders."""
    mock_get_portfolio.return_value = _make_portfolio([])
    mock_fetch.return_value = pd.DataFrame({"Close": [100.0]})

    scores = {"AAPL": 0.1, "MSFT": -0.2, "GOOG": 0.0}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    assert actions == []
    mock_buy.assert_not_called()
    mock_sell.assert_not_called()


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_max_positions_limits_buys(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """Only top max_positions tickers are bought."""
    mock_get_portfolio.return_value = _make_portfolio([])
    mock_fetch.return_value = pd.DataFrame({"Close": [50.0]})

    scores = {f"T{i}": 0.9 - i * 0.01 for i in range(10)}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=3)

    buy_actions = [a for a in actions if a.startswith("BUY")]
    assert len(buy_actions) == 3


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_empty_scores_returns_empty(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """Empty scores dict returns empty actions list without touching any API."""
    actions = execute_ml_signals({}, threshold=0.3, max_positions=5)

    assert actions == []
    mock_get_portfolio.assert_not_called()
    mock_buy.assert_not_called()
    mock_sell.assert_not_called()


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_no_duplicate_buy_for_existing_position(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """A ticker already in the portfolio is not bought again even with high score."""
    mock_get_portfolio.return_value = _make_portfolio(["AAPL"])
    mock_fetch.return_value = pd.DataFrame({"Close": [180.0]})

    scores = {"AAPL": 0.95}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    # No buy since already held; no sell since not bearish
    assert "BUY AAPL" not in actions
    assert "SELL AAPL" not in actions
    mock_buy.assert_not_called()


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_price_fetch_failure_skips_order(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """If price cannot be fetched, the order is skipped (no crash)."""
    mock_get_portfolio.return_value = _make_portfolio([])
    mock_fetch.return_value = None  # simulates fetch failure

    scores = {"AAPL": 0.8}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    assert actions == []
    mock_buy.assert_not_called()


@patch("strategies.ml_execution.fetch_ohlcv")
@patch("strategies.ml_execution.get_portfolio")
@patch("strategies.ml_execution.buy")
@patch("strategies.ml_execution.sell")
def test_buy_failure_does_not_raise(mock_sell, mock_buy, mock_get_portfolio, mock_fetch):
    """A buy() exception is caught and logged; remaining orders still proceed."""
    mock_get_portfolio.return_value = _make_portfolio([])
    mock_fetch.return_value = pd.DataFrame({"Close": [100.0]})
    mock_buy.side_effect = [ValueError("insufficient cash"), None]

    scores = {"FAIL": 0.9, "OK": 0.8}
    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5)

    # FAIL raises, OK succeeds
    assert "BUY OK" in actions
    assert "BUY FAIL" not in actions
