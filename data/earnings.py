"""Earnings calendar — fetch upcoming earnings dates via yfinance."""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_earnings_dates(ticker: str, lookahead_days: int = 90) -> Optional[pd.DataFrame]:
    """
    Fetch upcoming earnings dates for a ticker using yfinance.
    Returns a DataFrame with columns: [Date, EPS Estimate, Reported EPS, Surprise(%)]
    or None if unavailable.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.earnings_dates
        if cal is None or cal.empty:
            return None
        cal = cal.reset_index()
        cal.columns = [str(c) for c in cal.columns]
        # Filter to upcoming dates within lookahead window
        now = pd.Timestamp.now(tz='UTC')
        future = now + pd.Timedelta(days=lookahead_days)
        date_col = cal.columns[0]
        cal[date_col] = pd.to_datetime(cal[date_col], utc=True)
        upcoming = cal[(cal[date_col] >= now) & (cal[date_col] <= future)]
        return upcoming.reset_index(drop=True) if not upcoming.empty else cal.head(5)
    except Exception as e:
        logger.warning(f"Could not fetch earnings for {ticker}: {e}")
        return None


def get_next_earnings_date(ticker: str) -> Optional[str]:
    """Return the next earnings date as a string, or None."""
    df = get_earnings_dates(ticker, lookahead_days=180)
    if df is None or df.empty:
        return None
    try:
        date_col = df.columns[0]
        return str(df[date_col].iloc[0].date())
    except Exception:
        return None
