"""Momentum and trend-following strategies."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class MomentumSignal:
    ticker: str
    date: pd.Timestamp
    signal: str          # 'buy', 'sell', 'hold'
    strength: float      # 0.0 - 1.0
    reason: str


def compute_momentum_score(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    Composite momentum score combining:
    - Price momentum (N-day return)
    - Volume momentum (volume vs 20d avg)
    - RSI momentum (distance from 50)
    Returns a Series of scores normalised to [-1, 1].
    """
    close = df["Close"]
    volume = df["Volume"]

    # Price momentum: N-day return
    price_mom = close.pct_change(lookback)

    # Volume momentum: current vol vs rolling avg
    vol_avg = volume.rolling(lookback).mean()
    vol_mom = (volume / vol_avg - 1).clip(-1, 1)

    # RSI momentum: (RSI - 50) / 50 → -1 to 1
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_mom = ((rsi - 50) / 50).clip(-1, 1)

    # Weighted composite
    score = (0.5 * price_mom.clip(-0.5, 0.5) / 0.5 +
             0.25 * vol_mom +
             0.25 * rsi_mom)
    return score.clip(-1, 1)


def momentum_signals(df: pd.DataFrame, ticker: str = "TICKER",
                     lookback: int = 20,
                     buy_threshold: float = 0.3,
                     sell_threshold: float = -0.3) -> List[MomentumSignal]:
    """
    Generate buy/sell signals from momentum score threshold crossings.
    Only generates a new signal when score crosses the threshold (avoids noise).
    """
    score = compute_momentum_score(df, lookback)
    signals = []
    in_position = False

    for i in range(lookback + 14, len(df)):
        s = score.iloc[i]
        if pd.isna(s):
            continue
        date = df.index[i]

        if not in_position and s >= buy_threshold:
            signals.append(MomentumSignal(
                ticker=ticker, date=date, signal='buy',
                strength=float(min(s, 1.0)),
                reason=f"Momentum score {s:.2f} crossed buy threshold {buy_threshold}"
            ))
            in_position = True
        elif in_position and s <= sell_threshold:
            signals.append(MomentumSignal(
                ticker=ticker, date=date, signal='sell',
                strength=float(abs(min(s, -1.0))),
                reason=f"Momentum score {s:.2f} crossed sell threshold {sell_threshold}"
            ))
            in_position = False

    return signals


def momentum_backtest(df: pd.DataFrame, lookback: int = 20,
                      buy_threshold: float = 0.3,
                      sell_threshold: float = -0.3) -> dict:
    """
    Simple momentum strategy backtest. Returns summary dict.
    """
    signals = momentum_signals(df, lookback=lookback,
                               buy_threshold=buy_threshold,
                               sell_threshold=sell_threshold)
    if len(signals) < 2:
        return {"total_return": 0.0, "num_trades": 0, "win_rate": 0.0}

    close = df["Close"]
    trades = []
    entry_price = None

    for sig in signals:
        price = close.asof(sig.date) if hasattr(close, 'asof') else close[close.index <= sig.date].iloc[-1]
        if sig.signal == 'buy':
            entry_price = float(price)
        elif sig.signal == 'sell' and entry_price is not None:
            ret = (float(price) - entry_price) / entry_price
            trades.append(ret)
            entry_price = None

    if not trades:
        return {"total_return": 0.0, "num_trades": 0, "win_rate": 0.0}

    total_return = float(np.prod([1 + r for r in trades]) - 1)
    win_rate = float(np.mean([1 if r > 0 else 0 for r in trades]))
    return {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_trade": float(np.mean(trades)),
        "best_trade": float(max(trades)),
        "worst_trade": float(min(trades)),
    }
