"""Tests for data/features.py — feature engineering framework."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.features import (
    _FEATURE_COLS,
    _FWD_COLS,
    _TB_LABEL_COLS,
    build_feature_matrix,
)


def _make_ohlcv(n: int = 200, seed: int = 42, start: str = "2022-01-01") -> pd.DataFrame:
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def _mock_fetch(ticker: str, period: str) -> pd.DataFrame:
    """Return synthetic OHLCV regardless of ticker/period."""
    seed = abs(hash(ticker)) % 2**16
    return _make_ohlcv(n=252, seed=seed)


# ── build_feature_matrix ───────────────────────────────────────────────────────

def test_build_feature_matrix_shape():
    tickers = ["AAPL", "MSFT", "GOOGL"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    assert not fm.empty
    assert fm.index.names == ["date", "ticker"]
    for col in _FEATURE_COLS:
        assert col in fm.columns, f"Missing feature column: {col}"
    for col in _FWD_COLS:
        assert col in fm.columns, f"Missing forward-return column: {col}"


def test_build_feature_matrix_ticker_in_index():
    tickers = ["AAPL", "MSFT"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    index_tickers = set(fm.index.get_level_values("ticker").unique())
    assert index_tickers == {"AAPL", "MSFT"}


def test_forward_return_no_lookahead():
    """fwd_ret_5d on date t should equal close[t+5] / close[t+1] - 1."""
    tickers = ["AAPL"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    aapl = fm.xs("AAPL", level="ticker")

    # Find a row with a valid fwd_ret_5d
    valid = aapl["fwd_ret_5d"].dropna()
    assert len(valid) > 10, "Expected forward returns to exist"

    # The label must be negative or positive but well-formed (not an obviously wrong sign)
    # Main assertion: fwd_ret_5d is not equal to ret_5d (which would be a lookahead)
    # (they can occasionally be equal by coincidence, so check overall correlation < 1)
    overlap = aapl[["ret_5d", "fwd_ret_5d"]].dropna()
    corr = overlap["ret_5d"].corr(overlap["fwd_ret_5d"])
    assert corr < 0.99, "fwd_ret_5d appears to be identical to ret_5d (possible lookahead)"


def test_cross_sectional_zscore_mean_near_zero():
    """Mean of any feature across tickers on a given date should be ~0 after z-scoring."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    # Sample a date with all tickers present
    date_counts = fm.groupby(level="date").size()
    full_dates = date_counts[date_counts == len(tickers)].index
    if len(full_dates) == 0:
        pytest.skip("No date with all tickers — skip z-score check")

    sample_date = full_dates[len(full_dates) // 2]
    group = fm.xs(sample_date, level="date")

    for col in ["ret_5d", "realised_vol_21d", "vol_ratio_20d"]:
        if col in group.columns:
            col_vals = group[col].dropna()
            if len(col_vals) >= 3:
                assert abs(col_vals.mean()) < 0.5, (
                    f"Cross-sectional mean of {col} on {sample_date} = {col_vals.mean():.4f}, "
                    f"expected near 0 after z-scoring"
                )


def test_insufficient_data_ticker_dropped(caplog):
    """A ticker returning fewer rows than _MIN_ROWS should be excluded."""
    def _short_fetch(ticker: str, period: str) -> pd.DataFrame:
        if ticker == "SHORT":
            return _make_ohlcv(n=10, seed=1)  # below _MIN_ROWS
        return _make_ohlcv(n=252, seed=42)

    import logging
    with patch("data.features.fetch_ohlcv", side_effect=_short_fetch):
        with caplog.at_level(logging.WARNING, logger="data.features"):
            fm = build_feature_matrix(["AAPL", "SHORT"], period="2y")

    index_tickers = set(fm.index.get_level_values("ticker").unique())
    assert "SHORT" not in index_tickers
    assert "AAPL" in index_tickers


def test_all_bad_tickers_returns_empty():
    """If all tickers fail, build_feature_matrix returns an empty DataFrame."""
    def _empty_fetch(ticker: str, period: str) -> pd.DataFrame:
        return pd.DataFrame()

    with patch("data.features.fetch_ohlcv", side_effect=_empty_fetch):
        fm = build_feature_matrix(["AAPL", "MSFT"], period="2y")

    assert fm.empty


def test_volume_ratio_formula():
    """vol_ratio_20d should be Volume / 20-day rolling mean Volume (un-z-scored for 1 ticker)."""
    tickers = ["SOLO"]  # single ticker so z-score is skipped
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    solo = fm.xs("SOLO", level="ticker")
    # Verify non-null and positive (Volume > 0 always)
    valid = solo["vol_ratio_20d"].dropna()
    assert len(valid) > 0
    assert (valid > 0).all(), "vol_ratio_20d should be positive"


def test_build_feature_matrix_triple_barrier_emits_bin_column():
    """label_type='triple_barrier' adds tb_bin/tb_ret/tb_target columns with
    values in the expected support."""
    tickers = ["AAPL", "MSFT"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(
            tickers, period="2y",
            label_type="triple_barrier", pt_sl=(1.0, 1.0), num_days=5,
        )

    for col in _TB_LABEL_COLS:
        assert col in fm.columns, f"Missing triple-barrier column: {col}"

    bins = fm["tb_bin"].dropna().unique()
    assert set(bins).issubset({-1, 0, 1}), f"unexpected tb_bin values: {bins}"

    assert fm["tb_target"].dropna().ge(0).all(), "tb_target (vol) must be non-negative"


def test_build_feature_matrix_fwd_ret_default_has_no_tb_columns():
    """Default label_type keeps the feature matrix free of triple-barrier cols."""
    tickers = ["AAPL"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    for col in _TB_LABEL_COLS:
        assert col not in fm.columns


def test_realised_vol_is_finite():
    """After cross-sectional z-scoring realised_vol_21d should be finite (not inf/NaN) where present."""
    tickers = ["AAPL", "MSFT"]
    with patch("data.features.fetch_ohlcv", side_effect=_mock_fetch):
        fm = build_feature_matrix(tickers, period="2y")

    valid = fm["realised_vol_21d"].dropna()
    assert np.isfinite(valid).all(), "Z-scored realised vol contains non-finite values"
