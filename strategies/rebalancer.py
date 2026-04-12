"""Portfolio rebalancer — computes trades needed to reach target weights."""
from dataclasses import dataclass


@dataclass
class RebalanceTrade:
    ticker: str
    action: str          # 'buy' | 'sell'
    current_value: float
    target_value: float
    delta_value: float   # target - current (positive = buy, negative = sell)
    delta_pct: float     # delta / total_equity * 100
    shares_approx: int   # abs(delta_value / current_price) rounded


def compute_rebalance_trades(
    current_positions: dict,      # {ticker: current_market_value}
    target_weights: dict,         # {ticker: weight} from get_max_sharpe_portfolio()
    total_equity: float,
    current_prices: dict,         # {ticker: current_price}
    min_trade_value: float = 500,
) -> list[RebalanceTrade]:
    """
    Compute the trades needed to move from current_positions to target_weights.

    Tickers in target_weights but missing from current_positions are treated as
    $0 current holdings (new buys).  Tickers in current_positions that are NOT
    in target_weights are treated as full sells (target weight = 0).

    Returns a list of RebalanceTrade sorted by abs(delta_value) descending,
    with trades whose abs(delta_value) < min_trade_value excluded.
    """
    all_tickers = set(target_weights) | set(current_positions)
    trades: list[RebalanceTrade] = []

    for ticker in all_tickers:
        current_value = current_positions.get(ticker, 0.0)
        weight = target_weights.get(ticker, 0.0)
        target_value = weight * total_equity
        delta_value = target_value - current_value

        if abs(delta_value) < min_trade_value:
            continue

        price = current_prices.get(ticker)
        if price is None or price <= 0:
            # Can't compute shares without a price — skip this ticker
            continue

        shares_approx = round(abs(delta_value) / price)
        delta_pct = delta_value / total_equity * 100

        trades.append(RebalanceTrade(
            ticker=ticker,
            action="buy" if delta_value > 0 else "sell",
            current_value=current_value,
            target_value=target_value,
            delta_value=delta_value,
            delta_pct=delta_pct,
            shares_approx=shares_approx,
        ))

    trades.sort(key=lambda t: abs(t.delta_value), reverse=True)
    return trades


def rebalance_summary(trades: list[RebalanceTrade]) -> dict:
    """
    Aggregate statistics for a list of RebalanceTrade.

    Returns:
        total_buys          – sum of delta_value for buy trades
        total_sells         – sum of abs(delta_value) for sell trades
        net_cash_impact     – total_sells - total_buys  (positive = net cash freed)
        num_trades          – number of trades
        estimated_commission – $0.005/share per trade, minimum $1 per trade
    """
    total_buys = sum(t.delta_value for t in trades if t.action == "buy")
    total_sells = sum(abs(t.delta_value) for t in trades if t.action == "sell")
    net_cash_impact = total_sells - total_buys

    commission = sum(
        max(1.0, t.shares_approx * 0.005)
        for t in trades
    )

    return {
        "total_buys": total_buys,
        "total_sells": total_sells,
        "net_cash_impact": net_cash_impact,
        "num_trades": len(trades),
        "estimated_commission": commission,
    }
