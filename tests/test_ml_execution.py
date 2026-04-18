"""
tests/test_ml_execution.py — Unit tests for strategies/ml_execution.py.

Uses an in-memory FakeBroker implementing the BrokerProvider Protocol to
avoid touching SQLite, network, or the real paper_trader. The regime lookup
is patched to avoid yfinance.
"""
import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.ml_execution import execute_ml_signals


class FakeBroker:
    """Minimal in-memory broker mock conforming to BrokerProvider."""

    def __init__(self, positions=None, equity=100_000.0, raise_on="none"):
        self._positions = positions or []
        self._equity = equity
        self._raise_on = raise_on
        self.orders: list[dict] = []

    def get_account_info(self) -> dict:
        if self._raise_on == "account":
            raise RuntimeError("account fetch failed")
        return {"equity": self._equity, "cash": self._equity}

    def get_positions(self) -> list[dict]:
        if self._raise_on == "positions":
            raise RuntimeError("positions fetch failed")
        return list(self._positions)

    def place_order(self, symbol, qty, side, order_type="market", limit_price=None):
        if self._raise_on == f"order:{symbol}":
            raise RuntimeError("simulated order failure")
        self.orders.append({
            "symbol": symbol, "qty": qty, "side": side, "type": order_type,
        })
        return {"order_id": f"test-{len(self.orders)}", "status": "filled",
                "symbol": symbol, "qty": qty, "side": side}

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - unused
        return True

    def get_orders(self, status: str = "open") -> list[dict]:  # pragma: no cover
        return []


def _held(ticker: str, qty: float = 10.0) -> dict:
    return {"symbol": ticker, "qty": qty, "avg_entry_price": 100.0,
            "market_value": qty * 100.0, "unrealized_pl": 0.0}


@pytest.fixture(autouse=True)
def _patch_regime_and_prices():
    """Keep regime lookup deterministic and avoid yfinance in every test.

    The knowledge gate is stubbed to "fresh" so existing sizing tests are not
    perturbed by missing model pickles in the test checkout.  New tests below
    override ``_knowledge_gate`` explicitly where needed.
    """
    with patch("strategies.ml_execution._current_regime", return_value="trending_bull"), \
         patch("strategies.ml_execution.fetch_ohlcv",
               return_value=pd.DataFrame({"Close": [150.0]})), \
         patch("strategies.ml_execution._knowledge_gate",
               return_value=(1.0, "fresh", False)):
        yield


def _buys_for(broker: FakeBroker) -> list[str]:
    return [o["symbol"] for o in broker.orders if o["side"] == "buy"]


def _sells_for(broker: FakeBroker) -> list[str]:
    return [o["symbol"] for o in broker.orders if o["side"] == "sell"]


def test_buys_top_long_candidates():
    """High positive scores trigger BUY orders for tickers not already held."""
    broker = FakeBroker(positions=[], equity=100_000.0)
    scores = {"AAPL": 0.8, "MSFT": 0.6, "GOOG": -0.5}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert "AAPL" in _buys_for(broker)
    assert "MSFT" in _buys_for(broker)
    assert "GOOG" not in _buys_for(broker)
    assert any(a.startswith("BUY AAPL x") for a in actions)
    assert _sells_for(broker) == []


def test_sells_bearish_existing_position():
    """Bearish score (< -threshold) for a held position triggers SELL."""
    broker = FakeBroker(positions=[_held("AAPL", qty=10.0)], equity=100_000.0)
    scores = {"AAPL": -0.7, "MSFT": 0.9}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert "AAPL" in _sells_for(broker)
    assert any(a.startswith("SELL AAPL") for a in actions)
    sell_order = next(o for o in broker.orders if o["side"] == "sell")
    assert sell_order["qty"] == 10.0


def test_no_action_within_neutral_band():
    """Scores within [-threshold, threshold] produce no orders."""
    broker = FakeBroker(positions=[], equity=100_000.0)
    scores = {"AAPL": 0.1, "MSFT": -0.2, "GOOG": 0.0}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert actions == []
    assert broker.orders == []


def test_max_positions_limits_buys():
    """Only top max_positions tickers are bought."""
    broker = FakeBroker(positions=[], equity=100_000.0)
    scores = {f"T{i}": 0.9 - i * 0.01 for i in range(10)}

    execute_ml_signals(scores, threshold=0.3, max_positions=3, broker=broker)

    assert len(_buys_for(broker)) == 3


def test_empty_scores_returns_empty():
    """Empty scores dict returns empty actions list without touching any API."""
    broker = FakeBroker()

    actions = execute_ml_signals({}, threshold=0.3, max_positions=5, broker=broker)

    assert actions == []
    assert broker.orders == []


def test_no_duplicate_buy_for_existing_position():
    """A ticker already in the portfolio is not bought again even with high score."""
    broker = FakeBroker(positions=[_held("AAPL", qty=5.0)], equity=100_000.0)
    scores = {"AAPL": 0.95}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert _buys_for(broker) == []
    assert _sells_for(broker) == []
    assert actions == []


def test_price_fetch_failure_skips_order():
    """If price cannot be fetched, the order is skipped (no crash)."""
    broker = FakeBroker(positions=[], equity=100_000.0)
    scores = {"AAPL": 0.8}

    with patch("strategies.ml_execution.fetch_ohlcv", return_value=None):
        actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert actions == []
    assert broker.orders == []


def test_buy_failure_does_not_raise():
    """A broker.place_order exception is caught; remaining orders still proceed."""
    broker = FakeBroker(positions=[], equity=100_000.0, raise_on="order:FAIL")
    scores = {"FAIL": 0.9, "OK": 0.8}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert "OK" in _buys_for(broker)
    assert "FAIL" not in _buys_for(broker)
    assert any(a.startswith("BUY OK") for a in actions)


def test_kelly_sizing_scales_with_equity_and_score():
    """Larger equity and higher score both produce more shares."""
    small = FakeBroker(positions=[], equity=10_000.0)
    large = FakeBroker(positions=[], equity=1_000_000.0)

    execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=small)
    execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=large)

    small_qty = small.orders[0]["qty"]
    large_qty = large.orders[0]["qty"]
    assert large_qty > small_qty
    assert small_qty >= 1

    low_score = FakeBroker(positions=[], equity=1_000_000.0)
    high_score = FakeBroker(positions=[], equity=1_000_000.0)
    execute_ml_signals({"AAPL": 0.4}, threshold=0.3, max_positions=5, broker=low_score)
    execute_ml_signals({"AAPL": 0.95}, threshold=0.3, max_positions=5, broker=high_score)
    assert high_score.orders[0]["qty"] > low_score.orders[0]["qty"]


def test_regime_multiplier_reduces_size_in_high_vol():
    """high_vol regime halves the Kelly fraction → smaller position sizes."""
    normal = FakeBroker(positions=[], equity=100_000.0)
    stormy = FakeBroker(positions=[], equity=100_000.0)

    with patch("strategies.ml_execution._current_regime", return_value="trending_bull"):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=normal)
    with patch("strategies.ml_execution._current_regime", return_value="high_vol"):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=stormy)

    assert stormy.orders[0]["qty"] < normal.orders[0]["qty"]


def test_positions_fetch_failure_falls_back_to_empty():
    """If get_positions raises, treat the portfolio as empty and still place buys."""
    broker = FakeBroker(positions=[], equity=100_000.0, raise_on="positions")
    scores = {"AAPL": 0.8}

    actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert "AAPL" in _buys_for(broker)
    assert any(a.startswith("BUY AAPL") for a in actions)


# ── Knowledge-adaption gate ───────────────────────────────────────────────

def test_knowledge_gate_skips_all_when_baseline_missing():
    """retrain + baseline missing → no buys even for strong positive scores."""
    broker = FakeBroker(positions=[], equity=100_000.0)
    scores = {"AAPL": 0.9, "MSFT": 0.8}

    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(0.4, "retrain", True)):
        actions = execute_ml_signals(scores, threshold=0.3, max_positions=5, broker=broker)

    assert actions == []
    assert _buys_for(broker) == []


def test_knowledge_gate_reduces_kelly_on_monitor():
    """monitor recommendation (0.7×) produces smaller qty than fresh (1.0×)."""
    fresh = FakeBroker(positions=[], equity=1_000_000.0)
    monitor = FakeBroker(positions=[], equity=1_000_000.0)

    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(1.0, "fresh", False)):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=fresh)
    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(0.7, "monitor", False)):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=monitor)

    assert monitor.orders[0]["qty"] < fresh.orders[0]["qty"]


def test_knowledge_gate_reduces_kelly_on_retrain_without_skip():
    """retrain + pickles present (skip_all=False) still shrinks sizing."""
    fresh = FakeBroker(positions=[], equity=1_000_000.0)
    retrain = FakeBroker(positions=[], equity=1_000_000.0)

    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(1.0, "fresh", False)):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=fresh)
    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(0.4, "retrain", False)):
        execute_ml_signals({"AAPL": 0.9}, threshold=0.3, max_positions=5, broker=retrain)

    assert 0 < retrain.orders[0]["qty"] < fresh.orders[0]["qty"]


def test_knowledge_gate_still_exits_bearish_positions_when_skip():
    """skip_all blocks buys but existing bearish positions should still be exited."""
    broker = FakeBroker(positions=[_held("AAPL", qty=10.0)], equity=100_000.0)

    with patch("strategies.ml_execution._knowledge_gate",
               return_value=(0.4, "retrain", True)):
        actions = execute_ml_signals(
            {"AAPL": -0.7, "MSFT": 0.9}, threshold=0.3, max_positions=5, broker=broker
        )

    # AAPL should be sold (risk-reducing), MSFT must not be bought.
    assert "AAPL" in _sells_for(broker)
    assert _buys_for(broker) == []
    assert any(a.startswith("SELL AAPL") for a in actions)
