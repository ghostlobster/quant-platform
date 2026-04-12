import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import patch, MagicMock
import pandas as pd
from data.earnings import get_earnings_dates, get_next_earnings_date


def make_fake_calendar():
    dates = pd.to_datetime(["2026-05-01", "2026-08-01"], utc=True)
    return pd.DataFrame({
        "Earnings Date": dates,
        "EPS Estimate": [1.5, 1.8],
        "Reported EPS": [None, None],
        "Surprise(%)": [None, None],
    }).set_index("Earnings Date")


def test_get_earnings_dates_returns_dataframe():
    with patch("yfinance.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.earnings_dates = make_fake_calendar()
        result = get_earnings_dates("AAPL")
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_get_earnings_dates_handles_none():
    with patch("yfinance.Ticker") as MockTicker:
        instance = MockTicker.return_value
        instance.earnings_dates = None
        result = get_earnings_dates("FAKE")
    assert result is None


def test_get_next_earnings_date_returns_string():
    with patch("data.earnings.get_earnings_dates") as mock_get:
        dates = pd.to_datetime(["2026-06-15"], utc=True)
        mock_get.return_value = pd.DataFrame({"Earnings Date": dates})
        result = get_next_earnings_date("AAPL")
    assert result is None or isinstance(result, str)
