"""
screener/screener.py — Multi-factor stock screener.

Fetches ~3 months of daily data for a predefined 30-ticker universe,
computes five filter-ready metrics, and returns a ranked DataFrame.

Filters supported
-----------------
  rsi_min / rsi_max       : RSI(14) bounds  (e.g. RSI < 30 = oversold)
  change_days             : look-back window for % price change  (default 5)
  change_min / change_max : % price-change bounds over that window
  vol_spike_min           : current-day volume / 20-day avg volume  (e.g. 1.5)
  price_min / price_max   : last-close price bounds
  trend                   : "above_sma50" | "below_sma50" | None (any)

Signal labels (assigned to the *unfiltered* DataFrame for display)
-----------------
  Oversold      RSI < 30
  Overbought    RSI > 70
  Trending Up   price > SMA50 AND 5-day change > +3 %
  Trending Down price < SMA50 AND 5-day change < -3 %
  Neutral       everything else
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from data.indicators import compute_rsi

logger = logging.getLogger(__name__)

# ── Ticker universe ───────────────────────────────────────────────────────────

UNIVERSE: list[dict] = [
    # Technology
    {"ticker": "AAPL",  "sector": "Technology",   "name": "Apple"},
    {"ticker": "MSFT",  "sector": "Technology",   "name": "Microsoft"},
    {"ticker": "NVDA",  "sector": "Technology",   "name": "NVIDIA"},
    {"ticker": "GOOGL", "sector": "Technology",   "name": "Alphabet"},
    {"ticker": "META",  "sector": "Technology",   "name": "Meta Platforms"},
    {"ticker": "AMD",   "sector": "Technology",   "name": "Advanced Micro Devices"},
    {"ticker": "TSLA",  "sector": "Technology",   "name": "Tesla"},
    # Finance
    {"ticker": "JPM",   "sector": "Finance",      "name": "JPMorgan Chase"},
    {"ticker": "BAC",   "sector": "Finance",      "name": "Bank of America"},
    {"ticker": "GS",    "sector": "Finance",      "name": "Goldman Sachs"},
    {"ticker": "MS",    "sector": "Finance",      "name": "Morgan Stanley"},
    {"ticker": "V",     "sector": "Finance",      "name": "Visa"},
    {"ticker": "MA",    "sector": "Finance",      "name": "Mastercard"},
    # Healthcare
    {"ticker": "JNJ",   "sector": "Healthcare",   "name": "Johnson & Johnson"},
    {"ticker": "UNH",   "sector": "Healthcare",   "name": "UnitedHealth Group"},
    {"ticker": "PFE",   "sector": "Healthcare",   "name": "Pfizer"},
    {"ticker": "ABBV",  "sector": "Healthcare",   "name": "AbbVie"},
    {"ticker": "MRK",   "sector": "Healthcare",   "name": "Merck"},
    {"ticker": "LLY",   "sector": "Healthcare",   "name": "Eli Lilly"},
    # Energy
    {"ticker": "XOM",   "sector": "Energy",       "name": "ExxonMobil"},
    {"ticker": "CVX",   "sector": "Energy",       "name": "Chevron"},
    {"ticker": "COP",   "sector": "Energy",       "name": "ConocoPhillips"},
    {"ticker": "SLB",   "sector": "Energy",       "name": "SLB (Schlumberger)"},
    {"ticker": "EOG",   "sector": "Energy",       "name": "EOG Resources"},
    # Consumer
    {"ticker": "WMT",   "sector": "Consumer",     "name": "Walmart"},
    {"ticker": "COST",  "sector": "Consumer",     "name": "Costco"},
    {"ticker": "MCD",   "sector": "Consumer",     "name": "McDonald's"},
    {"ticker": "NKE",   "sector": "Consumer",     "name": "Nike"},
    {"ticker": "HD",    "sector": "Consumer",     "name": "Home Depot"},
    # Industrials
    {"ticker": "CAT",   "sector": "Industrials",  "name": "Caterpillar"},
    {"ticker": "BA",    "sector": "Industrials",  "name": "Boeing"},
    {"ticker": "GE",    "sector": "Industrials",  "name": "GE Aerospace"},
]

_SECTOR_META: dict[str, dict] = {item["ticker"]: item for item in UNIVERSE}
TICKERS: list[str] = [item["ticker"] for item in UNIVERSE]


# ── Indicator helpers ─────────────────────────────────────────────────────────


def _compute_sma(close: pd.Series, window: int = 50) -> Optional[float]:
    """Return latest SMA(window), or None if insufficient data."""
    if len(close) < window:
        return None
    return float(close.rolling(window).mean().dropna().iloc[-1])


def _compute_metrics(ticker: str, df: pd.DataFrame, change_days: int) -> dict:
    """Derive all screener metrics for one ticker from its OHLCV DataFrame."""
    close  = df["Close"]
    volume = df["Volume"]

    last_price  = float(close.iloc[-1])
    last_volume = float(volume.iloc[-1])

    # % price change over change_days
    if len(close) > change_days:
        prev_close  = float(close.iloc[-(change_days + 1)])
        price_change = (last_price - prev_close) / prev_close * 100
    else:
        price_change = float("nan")

    # Volume ratio vs 20-day average
    vol_avg = float(volume.rolling(20).mean().dropna().iloc[-1]) if len(volume) >= 20 else None
    vol_ratio = round(last_volume / vol_avg, 2) if (vol_avg and vol_avg > 0) else float("nan")

    # RSI (14)
    rsi = compute_rsi(close)

    # SMA 50
    sma50 = _compute_sma(close, 50)
    above_sma50 = (last_price > sma50) if sma50 is not None else None

    # Signal label
    if rsi is not None:
        if rsi < 30:
            signal = "Oversold"
        elif rsi > 70:
            signal = "Overbought"
        elif above_sma50 is True and not np.isnan(price_change) and price_change > 3:
            signal = "Trending Up"
        elif above_sma50 is False and not np.isnan(price_change) and price_change < -3:
            signal = "Trending Down"
        else:
            signal = "Neutral"
    else:
        signal = "N/A"

    return {
        "Ticker":        ticker,
        "Sector":        _SECTOR_META.get(ticker, {}).get("sector", "—"),
        "Name":          _SECTOR_META.get(ticker, {}).get("name", ticker),
        "Last Price":    round(last_price, 2),
        "RSI":           round(rsi, 1) if rsi is not None else None,
        f"Change {change_days}d (%)": round(price_change, 2) if not np.isnan(price_change) else None,
        "Vol Ratio":     vol_ratio if not np.isnan(vol_ratio) else None,
        "Above SMA50":   above_sma50,
        "Signal":        signal,
    }


# ── Batch fetch (single yfinance call — thread-safe) ─────────────────────────

def _fetch_batch(symbols: list[str], period: str = "3mo") -> dict[str, pd.DataFrame]:
    """
    Download all *symbols* in one yfinance call and split by ticker.
    Returns a dict {ticker: DataFrame}.  Skips tickers with < 20 rows.
    """
    if not symbols:
        return {}
    try:
        raw = yf.download(
            symbols, period=period, progress=False,
            auto_adjust=True, group_by="ticker",
        )
    except Exception as exc:
        logger.warning("Batch download failed: %s", exc)
        return {}

    if raw.empty:
        return {}

    data_map: dict[str, pd.DataFrame] = {}

    if len(symbols) == 1:
        # Single-ticker download has flat columns (no MultiIndex)
        sym = symbols[0]
        df = raw.dropna(subset=["Close"])
        if len(df) >= 20:
            data_map[sym] = df
    else:
        for sym in symbols:
            try:
                if sym not in raw.columns.get_level_values(0):
                    continue
                df = raw[sym].copy()
                df = df.dropna(subset=["Close"])
                if len(df) >= 20:
                    data_map[sym] = df
            except Exception as exc:
                logger.warning("Skipping %s: %s", sym, exc)

    return data_map


# ── Public API ────────────────────────────────────────────────────────────────

def run_screen(
    tickers: Optional[list[str]] = None,
    change_days: int = 5,
    rsi_min: Optional[float] = None,
    rsi_max: Optional[float] = None,
    change_min: Optional[float] = None,
    change_max: Optional[float] = None,
    vol_spike_min: Optional[float] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    trend: Optional[str] = None,
    period: str = "3mo",
) -> pd.DataFrame:
    """
    Fetch data for *tickers* (default: full UNIVERSE) via a single batch
    yfinance call, compute metrics, apply filters, and return a sorted DataFrame.

    Parameters
    ----------
    tickers       : list of symbols, or None to use full UNIVERSE
    change_days   : look-back window for % price change  (1–20)
    rsi_min/max   : RSI filter bounds
    change_min/max: % change bounds
    vol_spike_min : minimum volume ratio  (e.g. 1.5 = 1.5× avg)
    price_min/max : last-price bounds
    trend         : "above_sma50" | "below_sma50" | None
    period        : yfinance period string passed to download

    Returns
    -------
    pd.DataFrame sorted by Signal priority then RSI.
    Empty DataFrame if no tickers pass the filters.
    """
    symbols = tickers or TICKERS
    data_map = _fetch_batch(symbols, period)

    if not data_map:
        return pd.DataFrame()

    # ── Compute metrics ───────────────────────────────────────────────────────
    rows = []
    for sym, df in data_map.items():
        try:
            row = _compute_metrics(sym, df, change_days)
            rows.append(row)
        except Exception as exc:
            logger.warning("Metrics failed for %s: %s", sym, exc)

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    change_col = f"Change {change_days}d (%)"

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = pd.Series(True, index=result.index)

    if rsi_min is not None:
        mask &= result["RSI"].fillna(50) >= rsi_min
    if rsi_max is not None:
        mask &= result["RSI"].fillna(50) <= rsi_max

    if change_col in result.columns:
        if change_min is not None:
            mask &= result[change_col].fillna(0) >= change_min
        if change_max is not None:
            mask &= result[change_col].fillna(0) <= change_max

    if vol_spike_min is not None:
        mask &= result["Vol Ratio"].fillna(0) >= vol_spike_min

    if price_min is not None:
        mask &= result["Last Price"] >= price_min
    if price_max is not None:
        mask &= result["Last Price"] <= price_max

    if trend == "above_sma50":
        mask &= result["Above SMA50"] == True   # noqa: E712
    elif trend == "below_sma50":
        mask &= result["Above SMA50"] == False  # noqa: E712

    result = result[mask].copy()

    # ── Sort: oversold first, then overbought, then by RSI ascending ─────────
    signal_order = {"Oversold": 0, "Trending Up": 1, "Neutral": 2, "Trending Down": 3, "Overbought": 4, "N/A": 5}
    result["_sort"] = result["Signal"].map(signal_order).fillna(5)
    result = result.sort_values(["_sort", "RSI"]).drop(columns=["_sort"])
    result = result.reset_index(drop=True)

    return result
