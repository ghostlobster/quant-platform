import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from strategies.rebalancer import compute_rebalance_trades, rebalance_summary, RebalanceTrade


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _three_ticker_setup():
    """
    3-ticker portfolio with $100 000 total equity.
    Target weights: AAPL 50%, MSFT 30%, GOOG 20%

    Current positions (before rebalance):
      AAPL  $60 000  → overweight by $10 000  → sell
      MSFT  $25 000  → underweight by  $5 000  → buy
      GOOG  $20 000  → exactly at target       → no trade (delta = $0)
    """
    current_positions = {"AAPL": 60_000.0, "MSFT": 25_000.0, "GOOG": 20_000.0}
    target_weights = {"AAPL": 0.50, "MSFT": 0.30, "GOOG": 0.20}
    total_equity = 100_000.0
    current_prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 2500.0}
    return current_positions, target_weights, total_equity, current_prices


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_basic_rebalance_direction_and_count():
    """Overweight ticker becomes a sell; underweight becomes a buy; at-target is excluded."""
    positions, weights, equity, prices = _three_ticker_setup()
    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)

    tickers = {t.ticker: t for t in trades}

    assert "AAPL" in tickers, "overweight AAPL should generate a sell"
    assert tickers["AAPL"].action == "sell"
    assert abs(tickers["AAPL"].delta_value - (-10_000.0)) < 0.01

    assert "MSFT" in tickers, "underweight MSFT should generate a buy"
    assert tickers["MSFT"].action == "buy"
    assert abs(tickers["MSFT"].delta_value - 5_000.0) < 0.01

    # GOOG delta = 0 — should not appear at all
    assert "GOOG" not in tickers


def test_trades_sorted_by_abs_delta_descending():
    """Trades must be ordered largest abs(delta) first."""
    positions, weights, equity, prices = _three_ticker_setup()
    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)

    deltas = [abs(t.delta_value) for t in trades]
    assert deltas == sorted(deltas, reverse=True)


def test_min_trade_value_filter_removes_small_trades():
    """Trades below min_trade_value must be excluded."""
    current_positions = {
        "AAPL": 49_600.0,   # delta = 400 → below threshold
        "MSFT": 24_000.0,   # delta = 6 000 → above threshold
        "GOOG": 20_000.0,   # delta = 0
    }
    target_weights = {"AAPL": 0.50, "MSFT": 0.30, "GOOG": 0.20}
    total_equity = 100_000.0
    current_prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 2500.0}

    trades = compute_rebalance_trades(
        current_positions, target_weights, total_equity, current_prices,
        min_trade_value=500,
    )
    tickers = {t.ticker for t in trades}

    assert "AAPL" not in tickers, "AAPL delta $400 should be filtered out (< $500)"
    assert "MSFT" in tickers, "MSFT delta $6 000 should pass the filter"


def test_trades_below_default_min_excluded():
    """Default min_trade_value=500 — tiny delta is excluded by default."""
    positions = {"AAPL": 49_800.0}
    weights = {"AAPL": 0.50}
    equity = 100_000.0
    prices = {"AAPL": 200.0}

    trades = compute_rebalance_trades(positions, weights, equity, prices)
    assert len(trades) == 0, "delta $200 should be excluded by default min_trade_value=500"


def test_summary_totals_correct():
    """rebalance_summary must correctly aggregate buys, sells, and net cash impact."""
    positions, weights, equity, prices = _three_ticker_setup()
    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)
    summary = rebalance_summary(trades)

    # AAPL: sell $10 000 → total_sells = 10 000
    # MSFT: buy  $5 000  → total_buys  =  5 000
    assert abs(summary["total_buys"] - 5_000.0) < 0.01
    assert abs(summary["total_sells"] - 10_000.0) < 0.01
    assert abs(summary["net_cash_impact"] - 5_000.0) < 0.01
    assert summary["num_trades"] == 2


def test_summary_commission_calculation():
    """Commission = max($1, shares * $0.005) per trade."""
    positions, weights, equity, prices = _three_ticker_setup()
    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)
    summary = rebalance_summary(trades)

    # AAPL: delta=$10 000, price=$200 → 50 shares → 50*0.005 = $0.25 → min $1.00
    # MSFT: delta= $5 000, price=$400 → 12 or 13 shares → ~$0.06 → min $1.00
    # Both trades hit the $1 minimum → total = $2.00
    assert summary["estimated_commission"] == pytest.approx(2.0, abs=0.01)


def test_empty_current_positions_new_portfolio():
    """With no current holdings, all target positions should be buys."""
    current_positions = {}
    target_weights = {"AAPL": 0.60, "MSFT": 0.40}
    total_equity = 50_000.0
    current_prices = {"AAPL": 150.0, "MSFT": 300.0}

    trades = compute_rebalance_trades(
        current_positions, target_weights, total_equity, current_prices,
        min_trade_value=100,
    )
    tickers = {t.ticker: t for t in trades}

    assert len(trades) == 2
    assert all(t.action == "buy" for t in trades)
    assert abs(tickers["AAPL"].delta_value - 30_000.0) < 0.01
    assert abs(tickers["MSFT"].delta_value - 20_000.0) < 0.01


def test_shares_approx_calculation():
    """shares_approx should be round(abs(delta_value) / price)."""
    positions = {"AAPL": 0.0}
    weights = {"AAPL": 1.0}
    equity = 10_000.0
    prices = {"AAPL": 153.0}  # 10 000 / 153 ≈ 65.36 → rounds to 65

    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)
    assert trades[0].shares_approx == round(10_000 / 153)


def test_delta_pct_values():
    """delta_pct must equal delta_value / total_equity * 100."""
    positions, weights, equity, prices = _three_ticker_setup()
    trades = compute_rebalance_trades(positions, weights, equity, prices, min_trade_value=100)

    for t in trades:
        expected = t.delta_value / equity * 100
        assert abs(t.delta_pct - expected) < 1e-9
