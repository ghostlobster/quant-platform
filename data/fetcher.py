"""
data/fetcher.py — yfinance wrapper with SQLite-backed cache.

Cache strategy
--------------
  - Intraday periods  (≤5d)  : TTL = 1 hour  (3 600 s)
  - Short periods     (≤1mo) : TTL = 4 hours (14 400 s)
  - Historical periods (>1mo): TTL = 24 hours (86 400 s)

The cache stores the DataFrame as JSON so no binary serialisation is needed.
Call `fetch_ohlcv()` everywhere instead of calling yfinance directly.
"""
import io
import time
from typing import Optional

import pandas as pd
import yfinance as yf

from data.db import get_connection, init_db
from utils.logger import get_logger

logger = get_logger(__name__)

# TTL buckets (seconds)
_TTL = {
    "intraday":   3_600,    # ≤ 5d
    "short":     14_400,    # ≤ 1mo
    "historical": 86_400,   # everything longer
}

# Periods classified as intraday / short / historical
_INTRADAY_PERIODS  = {"1d", "2d", "5d"}
_SHORT_PERIODS     = {"1mo", "3mo"}


def _ttl_for(period: str) -> int:
    """Return the appropriate TTL in seconds for a given period string."""
    if period in _INTRADAY_PERIODS:
        return _TTL["intraday"]
    if period in _SHORT_PERIODS:
        return _TTL["short"]
    return _TTL["historical"]


def _cache_read(ticker: str, period: str) -> Optional[pd.DataFrame]:
    """
    Return cached DataFrame if it exists and is still fresh, else None.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT data_json, fetched_at FROM price_cache WHERE ticker=? AND period=?",
            (ticker.upper(), period),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    age = time.time() - row["fetched_at"]
    ttl = _ttl_for(period)
    if age > ttl:
        logger.debug("Cache expired for %s/%s (age=%.0fs, ttl=%ds)", ticker, period, age, ttl)
        return None

    logger.debug("Cache HIT for %s/%s (age=%.0fs)", ticker, period, age)
    # Wrap in StringIO — newer pandas requires a file-like object, not a raw string
    df = pd.read_json(io.StringIO(row["data_json"]), orient="split")
    df.index = pd.to_datetime(df.index)
    return df


def _cache_write(ticker: str, period: str, df: pd.DataFrame) -> None:
    """Persist a DataFrame into the price_cache table (upsert)."""
    data_json = df.to_json(orient="split", date_format="iso")
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO price_cache (ticker, period, fetched_at, data_json)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, period) DO UPDATE SET
                    fetched_at = excluded.fetched_at,
                    data_json  = excluded.data_json
                """,
                (ticker.upper(), period, time.time(), data_json),
            )
    finally:
        conn.close()
    logger.debug("Cache WRITE for %s/%s (%d rows)", ticker, period, len(df))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance returns a MultiIndex (Price, Ticker) for the columns.
    Flatten to simple column names and drop NaN-close rows.
    """
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.droplevel(1)
    df = df.dropna(subset=["Close"])
    return df


def fetch_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Return a clean OHLCV DataFrame for *ticker* over *period*.

    Checks SQLite cache first; downloads from Yahoo Finance only on a miss
    or when the cached data has expired.

    Parameters
    ----------
    ticker : str   e.g. "AAPL", "SPY"
    period : str   yfinance period string: 1d 5d 1mo 3mo 6mo 1y 2y 5y

    Returns
    -------
    pd.DataFrame with columns: Open, High, Low, Close, Volume
    DatetimeIndex as the index (UTC dates).

    Raises
    ------
    ValueError if yfinance returns no data for the ticker/period.
    """
    init_db()  # no-op if tables already exist
    ticker = ticker.upper().strip()

    # --- Try cache first ---
    cached = _cache_read(ticker, period)
    if cached is not None:
        return cached

    # --- Network fetch ---
    logger.info("Fetching %s/%s from Yahoo Finance…", ticker, period)
    raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(f"No data returned by yfinance for '{ticker}' (period={period})")

    df = _flatten_columns(raw)

    # --- Persist to cache ---
    _cache_write(ticker, period, df)

    return df


def fetch_latest_price(ticker: str) -> dict:
    """
    Return a small dict with the latest price and 1-day change for *ticker*.
    Uses the 5d period so the cache refresh is frequent (1-hour TTL).

    Returns
    -------
    dict with keys: ticker, price, prev_close, change, pct_change, error
    """
    result = {
        "ticker": ticker.upper(),
        "price": None,
        "prev_close": None,
        "change": None,
        "pct_change": None,
        "error": None,
    }
    try:
        df = fetch_ohlcv(ticker, period="5d")
        if len(df) < 2:
            result["error"] = "insufficient data"
            return result
        result["price"]      = round(float(df["Close"].iloc[-1]), 2)
        result["prev_close"] = round(float(df["Close"].iloc[-2]), 2)
        result["change"]     = round(result["price"] - result["prev_close"], 2)
        result["pct_change"] = round(
            (result["change"] / result["prev_close"]) * 100, 2
        )
    except Exception as exc:
        result["error"] = str(exc)
    return result
