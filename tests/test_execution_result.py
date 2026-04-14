"""Tests for ExecutionResult dataclass and execution algo typing (Issue #24)."""
from __future__ import annotations

import pytest


def test_from_fills_empty():
    from adapters.execution_algo.result import ExecutionResult
    result = ExecutionResult.from_fills([], "AAPL", "buy", "market", decision_price=150.0)
    assert result.symbol == "AAPL"
    assert result.side == "buy"
    assert result.total_qty == 0.0
    assert result.avg_fill_price == 0.0
    assert result.slippage_bps == 0.0


def test_from_fills_single():
    from adapters.execution_algo.result import ExecutionResult
    fills = [{"price": 151.0, "qty": 10.0}]
    result = ExecutionResult.from_fills(fills, "AAPL", "buy", "market", decision_price=150.0)
    assert result.total_qty == pytest.approx(10.0)
    assert result.avg_fill_price == pytest.approx(151.0)
    # slippage = (151 - 150) / 150 * 10000 ≈ 66.67 bps
    assert result.slippage_bps == pytest.approx(66.6667, rel=1e-3)


def test_from_fills_multiple():
    from adapters.execution_algo.result import ExecutionResult
    fills = [
        {"price": 100.0, "qty": 5.0},
        {"price": 102.0, "qty": 5.0},
    ]
    result = ExecutionResult.from_fills(fills, "SPY", "buy", "twap", decision_price=100.0)
    assert result.total_qty == pytest.approx(10.0)
    assert result.avg_fill_price == pytest.approx(101.0)
    # slippage = (101 - 100) / 100 * 10000 = 100 bps
    assert result.slippage_bps == pytest.approx(100.0)


def test_market_adapter_returns_execution_result():
    from unittest.mock import MagicMock
    from adapters.execution_algo.market_adapter import MarketAlgoAdapter
    from adapters.execution_algo.result import ExecutionResult

    mock_broker = MagicMock()
    mock_broker.place_order.return_value = {"price": 200.0, "qty": 5.0}

    adapter = MarketAlgoAdapter()
    result = adapter.execute("MSFT", 5.0, "buy", mock_broker, decision_price=200.0)

    assert isinstance(result, ExecutionResult)
    assert result.symbol == "MSFT"
    assert result.algo == "market"
    assert result.total_qty == pytest.approx(5.0)


def test_execution_result_algo_field():
    from adapters.execution_algo.result import ExecutionResult
    result = ExecutionResult.from_fills(
        [{"price": 50.0, "qty": 2.0}], "JPM", "sell", "vwap"
    )
    assert result.algo == "vwap"
    assert result.side == "sell"


def test_zero_decision_price_no_slippage():
    from adapters.execution_algo.result import ExecutionResult
    fills = [{"price": 300.0, "qty": 1.0}]
    result = ExecutionResult.from_fills(fills, "NVDA", "buy", "market", decision_price=0.0)
    assert result.slippage_bps == 0.0
