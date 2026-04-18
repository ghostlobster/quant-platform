"""
data/features.py — Financial feature engineering for ML alpha models.

Builds a MultiIndex (date, ticker) feature matrix from OHLCV data suitable
for supervised learning.  All data is fetched via fetch_ohlcv() to benefit
from the SQLite cache layer.

Feature columns produced
------------------------
Input features (cross-sectionally z-scored across tickers per date):
    ret_1d, ret_5d, ret_10d, ret_21d      — lookback price returns
    skew_21d                               — 21-day rolling skewness of daily returns
    kurt_21d                               — 21-day rolling excess kurtosis
    autocorr_1                             — lag-1 autocorrelation over 21-day window
    realised_vol_21d                       — annualised realised volatility (21d)
    vol_ratio_20d                          — Volume / 20-day rolling mean Volume
    vol_zscore_20d                         — (Volume − mean) / std over 20-day window
    vpin_50d                               — Volume-Synchronized PIN via BVC over 50 bars
    kyle_lambda_21d                        — Rolling Kyle's λ (price-impact coefficient)

Forward return targets (NOT z-scored; NaN for last n rows of each ticker):
    fwd_ret_1d, fwd_ret_5d, fwd_ret_10d, fwd_ret_21d

Triple-barrier targets (emitted when label_type="triple_barrier"):
    tb_bin      — label in {-1, 0, +1} (first barrier touched)
    tb_ret      — realised return from entry to first-touch
    tb_target   — volatility used to scale the barriers
    tb_t1       — timestamp of first barrier touched (needed for AFML
                  Ch 4 sample-uniqueness weighting)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data.fetcher import fetch_ohlcv
from utils.logger import get_logger

log = get_logger(__name__)

# Minimum rows required to compute any features for a ticker
_MIN_ROWS = 50

_FEATURE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_21d",
    "skew_21d", "kurt_21d", "autocorr_1", "realised_vol_21d",
    "vol_ratio_20d", "vol_zscore_20d",
    "vpin_50d", "kyle_lambda_21d",
]
_FWD_COLS = ["fwd_ret_1d", "fwd_ret_5d", "fwd_ret_10d", "fwd_ret_21d"]
_TB_LABEL_COLS = ["tb_bin", "tb_ret", "tb_target", "tb_t1"]


def _single_ticker_features(
    ticker: str,
    period: str,
    label_type: str = "fwd_ret",
    pt_sl: tuple[float, float] = (1.0, 1.0),
    num_days: int = 5,
) -> pd.DataFrame | None:
    """
    Compute raw (un-z-scored) features for one ticker.

    Returns a date-indexed DataFrame, or None if insufficient data.
    """
    df = fetch_ohlcv(ticker, period)
    if df is None or df.empty or len(df) < _MIN_ROWS:
        log.warning("features: skipping ticker — insufficient data", ticker=ticker, rows=len(df) if df is not None else 0)
        return None

    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    daily_ret = close.pct_change()

    out = pd.DataFrame(index=df.index)

    # ── Lookback return features ──────────────────────────────────────────────
    for n in (1, 5, 10, 21):
        out[f"ret_{n}d"] = close.pct_change(n)

    # ── Rolling statistics on daily returns ───────────────────────────────────
    roll21 = daily_ret.rolling(21)
    out["skew_21d"] = roll21.skew()
    # pandas rolling().kurt() returns excess kurtosis
    out["kurt_21d"] = roll21.kurt()
    # Vectorised lag-1 autocorrelation: corr(r_t, r_{t-1}) over 21-day window
    out["autocorr_1"] = daily_ret.rolling(21).corr(daily_ret.shift(1))
    out["realised_vol_21d"] = roll21.std() * np.sqrt(252)

    # ── Volume features ───────────────────────────────────────────────────────
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["vol_ratio_20d"] = volume / vol_mean
    out["vol_zscore_20d"] = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # ── Microstructural features (AFML Ch 19) ─────────────────────────────────
    from analysis.microstructure import kyle_lambda, vpin

    out["vpin_50d"] = vpin(close, volume, window=50)
    out["kyle_lambda_21d"] = kyle_lambda(close, volume, window=21)

    # ── Forward return labels (shift(-n) so label sits on the entry date) ─────
    for n in (1, 5, 10, 21):
        out[f"fwd_ret_{n}d"] = close.pct_change(n).shift(-n)

    # ── Triple-barrier labels (López de Prado Ch 3) ───────────────────────────
    if label_type == "triple_barrier":
        from analysis.triple_barrier import triple_barrier_labels

        tb = triple_barrier_labels(
            close,
            events=close.index,
            pt_sl=pt_sl,
            num_days=num_days,
        )
        if not tb.empty:
            out["tb_bin"] = tb["bin"].reindex(out.index)
            out["tb_ret"] = tb["ret"].reindex(out.index)
            out["tb_target"] = tb["target"].reindex(out.index)
            out["tb_t1"] = tb["t1"].reindex(out.index)

    return out


def build_feature_matrix(
    tickers: list[str],
    period: str = "2y",
    label_type: str = "fwd_ret",
    pt_sl: tuple[float, float] = (1.0, 1.0),
    num_days: int = 5,
) -> pd.DataFrame:
    """
    Build a MultiIndex (date, ticker) feature matrix.

    Parameters
    ----------
    tickers    : list of ticker symbols to include
    period     : yfinance-style period string passed to fetch_ohlcv
    label_type : "fwd_ret" (default) keeps the existing forward-return
                 targets.  "triple_barrier" additionally emits tb_bin,
                 tb_ret, tb_target columns from
                 analysis.triple_barrier.triple_barrier_labels.
    pt_sl      : (profit-take, stop-loss) multipliers in daily-vol units;
                 used only when label_type="triple_barrier".
    num_days   : vertical-barrier horizon in days; triple-barrier only.

    Returns
    -------
    pd.DataFrame with MultiIndex(date, ticker).
    Input features are cross-sectionally z-scored across tickers per date.
    Forward return and triple-barrier columns are NOT z-scored.
    """
    frames: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        feat = _single_ticker_features(
            ticker, period,
            label_type=label_type, pt_sl=pt_sl, num_days=num_days,
        )
        if feat is not None and not feat.empty:
            frames[ticker] = feat

    if not frames:
        log.warning("build_feature_matrix: no usable tickers", requested=len(tickers))
        return pd.DataFrame()

    # Stack into MultiIndex (date, ticker) — concat along ticker axis
    combined = pd.concat(frames, names=["ticker", "date"])
    # Swap so that (date, ticker) is the natural order
    combined = combined.swaplevel().sort_index()

    # ── Cross-sectional z-score on input features only ────────────────────────
    feature_cols_present = [c for c in _FEATURE_COLS if c in combined.columns]

    if len(frames) > 1:
        # Vectorised approach avoids pandas version issues with transform + function
        for col in feature_cols_present:
            col_mean = combined.groupby(level="date")[col].transform("mean")
            col_std = combined.groupby(level="date")[col].transform("std")
            # Where std == 0 (all values identical on that date), leave as 0
            combined[col] = ((combined[col] - col_mean) / col_std.replace(0, np.nan)).fillna(0)
    # If only 1 ticker, z-score is undefined — leave features unscaled

    log.info(
        "build_feature_matrix: complete",
        tickers=len(frames),
        rows=len(combined),
        period=period,
    )
    return combined
