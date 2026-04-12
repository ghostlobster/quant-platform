"""Slippage and commission modelling for realistic backtest cost simulation."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionCost:
    gross_price: float       # price before slippage
    net_price: float         # price after slippage
    commission: float        # dollar commission
    slippage_bps: float      # slippage in basis points
    total_cost: float        # commission + slippage dollar cost
    cost_pct: float          # total cost as % of trade value


@dataclass
class ExecutionModel:
    """
    Configurable execution cost model.
    commission_per_share: $ per share (e.g. $0.005 for IB)
    min_commission: minimum per order (e.g. $1.00)
    slippage_bps: basis points of slippage (1 bps = 0.01%)
    market_impact_bps: additional impact for large orders
    """
    commission_per_share: float = 0.005
    min_commission: float = 1.00
    slippage_bps: float = 5.0       # 5 bps typical for liquid stocks
    market_impact_bps: float = 2.0  # additional for large orders


DEFAULT_MODEL = ExecutionModel()
ZERO_COST_MODEL = ExecutionModel(
    commission_per_share=0.0, min_commission=0.0,
    slippage_bps=0.0, market_impact_bps=0.0
)


def simulate_execution(
    price: float,
    shares: float,
    side: str,             # 'buy' or 'sell'
    avg_daily_volume: Optional[float] = None,
    model: ExecutionModel = DEFAULT_MODEL,
) -> ExecutionCost:
    """
    Simulate realistic execution cost for a trade.

    Slippage direction: buys pay more, sells receive less.
    Market impact scales with order size relative to ADV.
    """
    if price <= 0 or shares <= 0:
        raise ValueError("price and shares must be positive")

    # Base slippage
    total_bps = model.slippage_bps

    # Market impact: larger orders relative to ADV get more slippage
    if avg_daily_volume and avg_daily_volume > 0:
        participation_rate = shares / avg_daily_volume
        impact_bps = model.market_impact_bps * (participation_rate ** 0.5) * 10000
        total_bps += min(impact_bps, 50)  # cap at 50 bps

    slippage_pct = total_bps / 10000
    direction = 1 if side == "buy" else -1
    net_price = price * (1 + direction * slippage_pct)

    commission = max(shares * model.commission_per_share, model.min_commission)
    trade_value = net_price * shares
    total_cost = commission + abs(net_price - price) * shares
    cost_pct = total_cost / trade_value if trade_value > 0 else 0.0

    return ExecutionCost(
        gross_price=price,
        net_price=net_price,
        commission=commission,
        slippage_bps=total_bps,
        total_cost=total_cost,
        cost_pct=cost_pct,
    )


def apply_costs_to_returns(
    trade_returns: list,
    avg_trade_value: float,
    shares_per_trade: float = 100,
    model: ExecutionModel = DEFAULT_MODEL,
) -> list:
    """Apply round-trip execution costs to a list of trade returns."""
    adjusted = []
    for ret in trade_returns:
        # Round-trip cost: buy cost + sell cost
        buy_cost = simulate_execution(avg_trade_value, shares_per_trade, "buy", model=model)
        sell_cost = simulate_execution(avg_trade_value, shares_per_trade, "sell", model=model)
        round_trip_cost_pct = buy_cost.cost_pct + sell_cost.cost_pct
        adjusted.append(ret - round_trip_cost_pct)
    return adjusted


def cost_drag(model: ExecutionModel = DEFAULT_MODEL,
              trades_per_year: int = 52) -> float:
    """Estimate annualised return drag from execution costs (assuming $10k avg trade)."""
    avg_price = 100.0
    shares = 100
    buy = simulate_execution(avg_price, shares, "buy", model=model)
    sell = simulate_execution(avg_price, shares, "sell", model=model)
    round_trip_pct = buy.cost_pct + sell.cost_pct
    return round_trip_pct * trades_per_year
