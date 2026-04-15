import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock optional transitive deps before anything imports them
sys.modules.setdefault("yfinance", MagicMock())

from broker.execution import ZERO_COST_MODEL, cost_drag, simulate_execution  # noqa: E402


def test_buy_pays_more():
    buy  = simulate_execution(100.0, 100, "buy")
    sell = simulate_execution(100.0, 100, "sell")
    assert buy.net_price > 100.0
    assert sell.net_price < 100.0


def test_zero_cost_model():
    ec = simulate_execution(100.0, 100, "buy", model=ZERO_COST_MODEL)
    assert ec.net_price == 100.0
    assert ec.commission == 0.0
    assert ec.total_cost == 0.0


def test_commission_minimum():
    # 1 share × $0.005 = $0.005, but min is $1
    ec = simulate_execution(100.0, 1, "buy")
    assert ec.commission == 1.0


def test_commission_scales_with_shares():
    ec = simulate_execution(100.0, 1000, "buy")
    assert ec.commission == pytest.approx(5.0)


def test_market_impact_increases_cost():
    # Small ADV means high participation → more slippage
    low_adv  = simulate_execution(100.0, 1000, "buy", avg_daily_volume=2000)
    no_adv   = simulate_execution(100.0, 1000, "buy")
    assert low_adv.slippage_bps >= no_adv.slippage_bps


def test_cost_drag_positive():
    drag = cost_drag(trades_per_year=52)
    assert drag > 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_execution(-1.0, 100, "buy")
    with pytest.raises(ValueError):
        simulate_execution(100.0, 0, "buy")


# ── TWAPAdapter ────────────────────────────────────────────────────────────────

def test_twap_execute_returns_execution_result():
    from unittest.mock import MagicMock, patch

    from adapters.execution_algo.result import ExecutionResult
    from adapters.execution_algo.twap_adapter import TWAPAdapter

    mock_broker = MagicMock()
    # Return fill with the qty that was requested so total_qty sums correctly
    mock_broker.place_order.side_effect = lambda **kw: {
        "symbol": kw["symbol"], "qty": kw["qty"],
        "side": kw["side"], "fill_price": 150.0, "status": "filled",
    }

    adapter = TWAPAdapter(slice_seconds=30)  # 2 slices for 1-min window
    with patch("time.sleep"):
        result = adapter.execute(
            symbol="AAPL",
            total_qty=10.0,
            side="buy",
            broker=mock_broker,
            duration_minutes=1,
            decision_price=149.0,
        )

    assert isinstance(result, ExecutionResult)
    assert result.symbol == "AAPL"
    assert result.algo == "twap"
    assert result.total_qty == pytest.approx(10.0)


def test_twap_slices_into_multiple_orders():
    from unittest.mock import MagicMock, patch

    from adapters.execution_algo.twap_adapter import TWAPAdapter

    mock_broker = MagicMock()
    mock_broker.place_order.return_value = {
        "symbol": "AAPL", "qty": 5.0, "side": "buy",
        "fill_price": 200.0, "status": "filled",
    }

    adapter = TWAPAdapter(slice_seconds=30)  # 2 slices in 1 min
    with patch("time.sleep"):
        adapter.execute(
            symbol="AAPL",
            total_qty=100.0,
            side="buy",
            broker=mock_broker,
            duration_minutes=1,
        )

    # Should have called place_order 2 times (60s / 30s = 2 slices)
    assert mock_broker.place_order.call_count == 2


def test_twap_single_slice_no_sleep():
    """Duration <= slice_seconds → 1 slice → no sleep."""
    from unittest.mock import MagicMock, patch

    from adapters.execution_algo.twap_adapter import TWAPAdapter

    mock_broker = MagicMock()
    mock_broker.place_order.return_value = {
        "symbol": "MSFT", "qty": 50.0, "side": "sell",
        "fill_price": 300.0, "status": "filled",
    }

    adapter = TWAPAdapter(slice_seconds=120)  # 1 slice for 1-min window
    with patch("time.sleep") as mock_sleep:
        adapter.execute(
            symbol="MSFT",
            total_qty=50.0,
            side="sell",
            broker=mock_broker,
            duration_minutes=1,
        )

    mock_sleep.assert_not_called()


# ── VWAPAdapter ────────────────────────────────────────────────────────────────

def test_vwap_execute_returns_execution_result():
    from unittest.mock import patch

    import pandas as pd

    import data.fetcher  # noqa: F401 — ensures data.fetcher is in sys.modules
    from adapters.execution_algo.result import ExecutionResult
    from adapters.execution_algo.vwap_adapter import VWAPAdapter

    mock_broker = MagicMock()
    mock_broker.place_order.side_effect = lambda **kw: {
        "symbol": kw["symbol"], "qty": kw["qty"],
        "side": kw["side"], "fill_price": 152.0, "status": "filled",
    }

    mock_df = pd.DataFrame({"Volume": [1_000_000.0] * 5, "Close": [150.0] * 5})

    adapter = VWAPAdapter(lookback_days=5, slice_seconds=30)
    with patch("data.fetcher.fetch_ohlcv", return_value=mock_df), \
         patch("time.sleep"):
        result = adapter.execute(
            symbol="AAPL",
            total_qty=10.0,
            side="buy",
            broker=mock_broker,
            duration_minutes=1,
            decision_price=151.0,
        )

    assert isinstance(result, ExecutionResult)
    assert result.symbol == "AAPL"
    assert result.algo == "vwap"


def test_vwap_falls_back_to_uniform_weights_on_empty_df():
    from unittest.mock import patch

    import pandas as pd

    import data.fetcher  # noqa: F401 — ensures data.fetcher is in sys.modules
    from adapters.execution_algo.vwap_adapter import VWAPAdapter

    mock_broker = MagicMock()
    mock_broker.place_order.side_effect = lambda **kw: {
        "symbol": kw["symbol"], "qty": kw["qty"],
        "side": kw["side"], "fill_price": 50.0, "status": "filled",
    }

    empty_df = pd.DataFrame()  # empty → fallback to uniform weights

    adapter = VWAPAdapter(lookback_days=5, slice_seconds=30)
    with patch("data.fetcher.fetch_ohlcv", return_value=empty_df), \
         patch("time.sleep"):
        result = adapter.execute(
            symbol="XYZ",
            total_qty=5.0,
            side="buy",
            broker=mock_broker,
            duration_minutes=1,
        )

    assert result.symbol == "XYZ"
