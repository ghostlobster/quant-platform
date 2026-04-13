import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from broker.execution import ZERO_COST_MODEL, cost_drag, simulate_execution


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
