"""Kelly Criterion position sizing."""
import numpy as np
from typing import Optional


def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                   max_fraction: float = 0.25) -> float:
    """Full Kelly fraction, capped at max_fraction (default 25%) for safety.

    Args:
        win_rate: fraction of trades that are winners (0-1)
        avg_win:  average gain on winning trades (positive, e.g. 0.05 = 5%)
        avg_loss: average loss on losing trades (positive magnitude, e.g. 0.03 = 3%)
        max_fraction: safety cap — never exceed this allocation

    Returns:
        Recommended position size as fraction of capital (0-1)
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    loss_rate = 1 - win_rate
    odds = avg_win / avg_loss  # b in classic Kelly formula
    kelly = (odds * win_rate - loss_rate) / odds
    # Half-Kelly for conservatism, then cap
    half_kelly = kelly / 2
    return float(max(0.0, min(half_kelly, max_fraction)))


def kelly_from_backtest(total_return: float, trade_count: int,
                        win_rate: float, max_fraction: float = 0.25) -> float:
    """Estimate Kelly fraction from backtest summary stats."""
    if trade_count == 0 or win_rate <= 0:
        return 0.0
    avg_return_per_trade = total_return / trade_count if trade_count > 0 else 0.0
    win_trades = win_rate * trade_count
    loss_trades = (1 - win_rate) * trade_count
    if win_trades == 0 or loss_trades == 0:
        return 0.0
    # Approximate avg_win and avg_loss from total_return
    assumed_win = abs(avg_return_per_trade) * 1.5
    assumed_loss = abs(avg_return_per_trade) * 0.8
    return kelly_fraction(win_rate, assumed_win, assumed_loss, max_fraction)
