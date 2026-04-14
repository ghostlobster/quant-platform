"""
backtester/engine.py — Event-driven backtesting engine.

Supported strategies
--------------------
  sma_crossover   : Buy when SMA20 crosses above SMA50; sell when it crosses below.
  rsi_mean_revert : Buy when RSI(14) < 30; sell when RSI(14) > 70.

Output
------
  BacktestResult  : dataclass holding metrics + equity curve DataFrame + trade log.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── Data types ────────────────────────────────────────────────────────────────

StrategyName = Literal["sma_crossover", "rsi_mean_revert"]


@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    ret_pct: float          # (exit - entry) / entry * 100


@dataclass
class BacktestResult:
    strategy: str
    ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp

    # ── Core metrics ──────────────────────────────────────────────────────────
    total_return_pct: float
    buy_hold_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    num_trades: int
    avg_trade_pct: float = 0.0

    # ── Extended ratios (R-05) ────────────────────────────────────────────────
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # ── ATR stop-loss counter (R-06) ──────────────────────────────────────────
    stop_losses_triggered: int = 0

    # ── Detail ────────────────────────────────────────────────────────────────
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)  # Date, Equity, BuyHold


# ── Indicator helpers (standalone — no import of strategies.indicators) ───────

def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Signal generators ─────────────────────────────────────────────────────────

def _signals_sma_crossover(close: pd.Series) -> pd.Series:
    """
    +1 = long signal (SMA20 > SMA50), -1 = out (SMA20 <= SMA50), 0 = insufficient data.
    """
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sig = pd.Series(0, index=close.index, dtype=int)
    valid = sma20.notna() & sma50.notna()
    sig[valid & (sma20 > sma50)] = 1
    sig[valid & (sma20 <= sma50)] = -1
    return sig


def _signals_rsi_mean_revert(close: pd.Series) -> pd.Series:
    """
    +1 = buy trigger (RSI crossed below 30 on previous bar),
    -1 = sell trigger (RSI crossed above 70 on previous bar),
     0 = hold.
    State-based: once in a trade, stay until the exit signal fires.
    """
    rsi = _rsi(close)
    # Use position-based state machine: build raw entry/exit triggers
    triggers = pd.Series(0, index=close.index, dtype=int)
    triggers[rsi < 30] = 1   # oversold → want to be long
    triggers[rsi > 70] = -1  # overbought → want to be flat
    return triggers


# ── Core backtest loop ────────────────────────────────────────────────────────

def _run(
    df: pd.DataFrame,
    signals: pd.Series,
    strategy: StrategyName,
    atr: pd.Series | None = None,
    atr_multiplier: float = 2.0,
) -> tuple[list[Trade], pd.DataFrame, int]:
    """
    Simulate trades from a signal series.

    For SMA crossover signals are continuous (+1 = long, -1 = flat).
    For RSI, signals are trigger-based (+1 = enter long, -1 = exit long).

    If *atr* is provided, an ATR-based trailing stop-loss is applied: exit if
    price falls below entry_price - atr_multiplier * atr_at_entry.

    Returns (trades, equity_df, stop_losses_triggered).
    """
    close = df["Close"]
    n = len(close)
    trades: list[Trade] = []
    stop_losses_triggered = 0

    in_position = False
    entry_price = 0.0
    stop_price = 0.0
    entry_date: pd.Timestamp = close.index[0]
    equity = 1.0
    equity_curve = []

    for i in range(n):
        date = close.index[i]
        price = float(close.iloc[i])
        sig = int(signals.iloc[i])

        # ATR stop-loss check — exit if in position and price breaches stop
        if in_position and atr is not None and price <= stop_price:
            ret = (price - entry_price) / entry_price
            equity *= (1 + ret)
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=price,
                ret_pct=ret * 100,
            ))
            in_position = False
            stop_losses_triggered += 1

        if strategy == "sma_crossover":
            # Continuous: enter when sig turns +1, exit when it turns -1/0
            if not in_position and sig == 1:
                in_position = True
                entry_price = price
                entry_date = date
                atr_val = float(atr.iloc[i]) if atr is not None and not np.isnan(atr.iloc[i]) else 0.0
                stop_price = entry_price - atr_multiplier * atr_val
            elif in_position and sig != 1:
                ret = (price - entry_price) / entry_price
                equity *= (1 + ret)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=price,
                    ret_pct=ret * 100,
                ))
                in_position = False

        else:  # rsi_mean_revert — trigger-based
            if not in_position and sig == 1:
                in_position = True
                entry_price = price
                entry_date = date
                atr_val = float(atr.iloc[i]) if atr is not None and not np.isnan(atr.iloc[i]) else 0.0
                stop_price = entry_price - atr_multiplier * atr_val
            elif in_position and sig == -1:
                ret = (price - entry_price) / entry_price
                equity *= (1 + ret)
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=price,
                    ret_pct=ret * 100,
                ))
                in_position = False

        # Mark-to-market equity for the curve
        if in_position:
            mtm = equity * (price / entry_price)
        else:
            mtm = equity
        equity_curve.append({"Date": date, "Equity": mtm})

    # Close any open position at the last bar
    if in_position:
        price = float(close.iloc[-1])
        ret = (price - entry_price) / entry_price
        equity *= (1 + ret)
        trades.append(Trade(
            entry_date=entry_date,
            exit_date=close.index[-1],
            entry_price=entry_price,
            exit_price=price,
            ret_pct=ret * 100,
        ))
        equity_curve[-1]["Equity"] = equity

    equity_df = pd.DataFrame(equity_curve).set_index("Date")
    # Normalise to 100 starting value for readability
    equity_df["Equity"] = equity_df["Equity"] / equity_df["Equity"].iloc[0] * 100

    # Buy-and-hold curve
    bh = close / float(close.iloc[0]) * 100
    equity_df["BuyHold"] = bh.values

    return trades, equity_df, stop_losses_triggered


# ── Metric calculations ───────────────────────────────────────────────────────

def _sharpe(equity_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio from the equity curve (daily returns)."""
    daily_ret = equity_df["Equity"].pct_change().dropna()
    if daily_ret.std() == 0 or len(daily_ret) < 2:
        return 0.0
    excess = daily_ret - risk_free_rate / 252
    return float(excess.mean() / excess.std() * math.sqrt(252))


def _max_drawdown(equity_df: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown as a negative percentage."""
    eq = equity_df["Equity"]
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    return float(drawdown.min() * 100)


def _sortino(equity_df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio — only downside deviation in denominator."""
    daily_ret = equity_df["Equity"].pct_change().dropna()
    if len(daily_ret) < 2:
        return 0.0
    excess = daily_ret - risk_free_rate / 252
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    downside_std = float(downside.std() * math.sqrt(252))
    ann_return = float(excess.mean() * 252)
    return round(ann_return / downside_std, 3)


def _calmar(ann_return_pct: float, max_dd_pct: float) -> float:
    """Calmar ratio = annualised return / |max drawdown|. Returns 0 if drawdown is zero."""
    if max_dd_pct == 0.0:
        return 0.0
    return round(ann_return_pct / abs(max_dd_pct), 3)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range over *period* bars."""
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _metrics(trades: list[Trade], equity_df: pd.DataFrame) -> dict:
    if not trades:
        return {
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate_pct": 0.0,
            "num_trades": 0,
            "avg_trade_pct": 0.0,
        }

    final_equity = float(equity_df["Equity"].iloc[-1])
    total_ret = (final_equity / 100 - 1) * 100
    max_dd = round(_max_drawdown(equity_df), 2)

    # Annualised return for Calmar (approximate from total return and bar count)
    n_years = max(len(equity_df) / 252, 1e-6)
    ann_ret = ((final_equity / 100) ** (1 / n_years) - 1) * 100

    wins = [t for t in trades if t.ret_pct > 0]
    win_rate = len(wins) / len(trades) * 100
    avg_trade = sum(t.ret_pct for t in trades) / len(trades)

    return {
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio": round(_sharpe(equity_df), 3),
        "sortino_ratio": _sortino(equity_df),
        "calmar_ratio": _calmar(ann_ret, max_dd),
        "max_drawdown_pct": max_dd,
        "win_rate_pct": round(win_rate, 1),
        "num_trades": len(trades),
        "avg_trade_pct": round(avg_trade, 2),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    strategy: StrategyName,
    ticker: str = "UNKNOWN",
    start_date: str | None = None,
    end_date: str | None = None,
) -> BacktestResult:
    """
    Run a backtest on *df* (OHLCV DataFrame) using the chosen strategy.

    Parameters
    ----------
    df         : OHLCV DataFrame from data.fetcher.fetch_ohlcv
    strategy   : "sma_crossover" or "rsi_mean_revert"
    ticker     : symbol string (used in result labelling)
    start_date : optional ISO date string to slice the data
    end_date   : optional ISO date string to slice the data

    Returns
    -------
    BacktestResult
    """
    # Slice to requested date range
    data = df.copy()
    if start_date:
        data = data[data.index >= pd.Timestamp(start_date)]
    if end_date:
        data = data[data.index <= pd.Timestamp(end_date)]

    if len(data) < 60:
        raise ValueError(
            f"Not enough data for backtesting ({len(data)} rows). "
            "Select a longer date range (at least 60 trading days)."
        )

    close = data["Close"]

    if strategy == "sma_crossover":
        signals = _signals_sma_crossover(close)
    elif strategy == "rsi_mean_revert":
        signals = _signals_rsi_mean_revert(close)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    atr_series = _atr(data)
    trades, equity_df, stop_losses = _run(data, signals, strategy, atr=atr_series)
    m = _metrics(trades, equity_df)

    # Buy-and-hold return
    bh_ret = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100

    return BacktestResult(
        strategy=strategy,
        ticker=ticker.upper(),
        start_date=data.index[0],
        end_date=data.index[-1],
        total_return_pct=m["total_return_pct"],
        buy_hold_return_pct=round(bh_ret, 2),
        sharpe_ratio=m["sharpe_ratio"],
        sortino_ratio=m["sortino_ratio"],
        calmar_ratio=m["calmar_ratio"],
        max_drawdown_pct=m["max_drawdown_pct"],
        win_rate_pct=m["win_rate_pct"],
        num_trades=m["num_trades"],
        avg_trade_pct=m["avg_trade_pct"],
        stop_losses_triggered=stop_losses,
        trades=trades,
        equity_curve=equity_df,
    )


# ── Chart builder ─────────────────────────────────────────────────────────────

def build_equity_chart(result: BacktestResult) -> go.Figure:
    """
    Return a Plotly figure comparing the strategy equity curve to buy-and-hold.
    """
    strategy_label = {
        "sma_crossover": "SMA Crossover",
        "rsi_mean_revert": "RSI Mean Reversion",
    }.get(result.strategy, result.strategy)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve["Equity"],
        mode="lines",
        name=strategy_label,
        line=dict(color="#26a69a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve["BuyHold"],
        mode="lines",
        name="Buy & Hold",
        line=dict(color="#7986cb", width=1.5, dash="dash"),
    ))

    # Shade drawdown area
    eq = result.equity_curve["Equity"]
    roll_max = eq.cummax()
    fig.add_trace(go.Scatter(
        x=list(result.equity_curve.index) + list(result.equity_curve.index[::-1]),
        y=list(roll_max) + list(eq[::-1]),
        fill="toself",
        fillcolor="rgba(239,83,80,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Drawdown",
        showlegend=True,
        hoverinfo="skip",
    ))

    fig.update_layout(
        title=f"{result.ticker} — {strategy_label} vs Buy & Hold",
        template="plotly_dark",
        height=380,
        yaxis_title="Portfolio Value (100 = start)",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=50, b=40),
        hovermode="x unified",
    )
    return fig


def build_trade_log_df(result: BacktestResult) -> pd.DataFrame:
    """Return the trade log as a tidy DataFrame for display."""
    if not result.trades:
        return pd.DataFrame(columns=["Entry Date", "Exit Date", "Entry Price", "Exit Price", "Return (%)"])
    rows = [
        {
            "Entry Date":  t.entry_date.date(),
            "Exit Date":   t.exit_date.date(),
            "Entry Price": round(t.entry_price, 2),
            "Exit Price":  round(t.exit_price, 2),
            "Return (%)":  round(t.ret_pct, 2),
        }
        for t in result.trades
    ]
    return pd.DataFrame(rows)
