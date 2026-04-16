"""
strategies/indicators.py — Technical indicator calculations.

All functions accept a clean OHLCV DataFrame (output of data/fetcher.fetch_ohlcv)
and return a new DataFrame with the indicator columns appended.

Uses the `ta` library (https://github.com/bukosabino/ta) which wraps pandas
and requires no compiled extensions.
"""
import pandas as pd

try:
    import ta
    _TA_AVAILABLE = True
except ImportError:  # pragma: no cover
    ta = None  # type: ignore[assignment]
    _TA_AVAILABLE = False

# ── Individual indicator functions ────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index.
    Adds column: RSI_14
    Range 0–100; overbought > 70, oversold < 30.
    """
    df = df.copy()
    df[f"RSI_{window}"] = ta.momentum.RSIIndicator(
        close=df["Close"], window=window
    ).rsi()
    return df


def add_macd(
    df: pd.DataFrame,
    window_slow: int = 26,
    window_fast: int = 12,
    window_sign: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.
    Adds columns: MACD_line, MACD_signal, MACD_hist
    """
    df = df.copy()
    indicator = ta.trend.MACD(
        close=df["Close"],
        window_slow=window_slow,
        window_fast=window_fast,
        window_sign=window_sign,
    )
    df["MACD_line"]   = indicator.macd()
    df["MACD_signal"] = indicator.macd_signal()
    df["MACD_hist"]   = indicator.macd_diff()   # histogram = line − signal
    return df


def add_bollinger_bands(
    df: pd.DataFrame, window: int = 20, std: int = 2
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Adds columns: BB_upper, BB_mid (SMA), BB_lower, BB_pct_b
      - BB_pct_b: position of price within the band (0 = lower, 1 = upper)
    """
    df = df.copy()
    indicator = ta.volatility.BollingerBands(
        close=df["Close"], window=window, window_dev=std
    )
    df["BB_upper"]  = indicator.bollinger_hband()
    df["BB_mid"]    = indicator.bollinger_mavg()
    df["BB_lower"]  = indicator.bollinger_lband()
    df["BB_pct_b"]  = indicator.bollinger_pband()
    return df


def add_ema(df: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """
    Exponential Moving Averages.
    Adds columns: EMA_<window> for each window in *windows*.
    Default windows: [20, 50]
    """
    if windows is None:
        windows = [20, 50]
    df = df.copy()
    for w in windows:
        df[f"EMA_{w}"] = ta.trend.EMAIndicator(
            close=df["Close"], window=w
        ).ema_indicator()
    return df


def add_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Simple Moving Average of Volume.
    Adds column: Vol_SMA_<window>
    """
    df = df.copy()
    df[f"Vol_SMA_{window}"] = df["Volume"].rolling(window=window).mean()
    return df


# ── Convenience: apply all indicators at once ─────────────────────────────────

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply RSI, MACD, Bollinger Bands, EMA 20/50, and Volume SMA to *df*.
    Returns a new DataFrame with all indicator columns added.
    Rows without enough history will contain NaN (normal for rolling indicators).
    """
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema(df)
    df = add_volume_sma(df)
    return df


# ── Signal interpretation ─────────────────────────────────────────────────────

def _latest(df: pd.DataFrame, col: str):
    """Return the most recent non-NaN value for *col*, or None."""
    series = df[col].dropna()
    return float(series.iloc[-1]) if not series.empty else None


def generate_signals(df: pd.DataFrame) -> list[dict]:
    """
    Derive simple human-readable signals from the indicator values.

    Returns a list of dicts, each with:
      - indicator : str   e.g. "RSI"
      - value     : float current value
      - signal    : str   short label e.g. "Oversold"
      - detail    : str   full description
      - icon      : str   emoji
      - bullish   : bool | None   (None = neutral)
    """
    signals = []

    # ── RSI ──────────────────────────────────────────────────────────────────
    if "RSI_14" in df.columns:
        rsi = _latest(df, "RSI_14")
        if rsi is not None:
            if rsi < 30:
                signals.append({
                    "indicator": "RSI (14)",
                    "value": round(rsi, 1),
                    "signal": "Oversold",
                    "detail": f"RSI {rsi:.1f} — below 30, potential reversal up",
                    "icon": "⚠️",
                    "bullish": True,
                })
            elif rsi > 70:
                signals.append({
                    "indicator": "RSI (14)",
                    "value": round(rsi, 1),
                    "signal": "Overbought",
                    "detail": f"RSI {rsi:.1f} — above 70, potential reversal down",
                    "icon": "⚠️",
                    "bullish": False,
                })
            else:
                signals.append({
                    "indicator": "RSI (14)",
                    "value": round(rsi, 1),
                    "signal": "Neutral",
                    "detail": f"RSI {rsi:.1f} — within normal range (30–70)",
                    "icon": "➖",
                    "bullish": None,
                })

    # ── MACD ─────────────────────────────────────────────────────────────────
    if "MACD_line" in df.columns and "MACD_signal" in df.columns:
        macd_line   = _latest(df, "MACD_line")
        macd_signal = _latest(df, "MACD_signal")

        if macd_line is not None and macd_signal is not None:
            # Detect crossover: previous histogram vs current
            hist_series = df["MACD_hist"].dropna()
            prev_hist = float(hist_series.iloc[-2]) if len(hist_series) >= 2 else 0.0
            curr_hist = float(hist_series.iloc[-1])

            if prev_hist < 0 and curr_hist >= 0:
                label, icon, bullish = "Bullish crossover", "✅", True
                detail = f"MACD crossed above signal line ({macd_line:.3f} > {macd_signal:.3f})"
            elif prev_hist > 0 and curr_hist <= 0:
                label, icon, bullish = "Bearish crossover", "🔴", False
                detail = f"MACD crossed below signal line ({macd_line:.3f} < {macd_signal:.3f})"
            elif curr_hist > 0:
                label, icon, bullish = "Bullish momentum", "✅", True
                detail = f"MACD above signal line (hist={curr_hist:.3f})"
            else:
                label, icon, bullish = "Bearish momentum", "🔴", False
                detail = f"MACD below signal line (hist={curr_hist:.3f})"

            signals.append({
                "indicator": "MACD (12/26/9)",
                "value": round(macd_line, 4),
                "signal": label,
                "detail": detail,
                "icon": icon,
                "bullish": bullish,
            })

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    if "BB_pct_b" in df.columns:
        pct_b = _latest(df, "BB_pct_b")
        upper = _latest(df, "BB_upper")
        lower = _latest(df, "BB_lower")
        close = float(df["Close"].iloc[-1])

        if pct_b is not None:
            if pct_b > 1.0:
                label, icon, bullish = "Above upper band", "⚠️", False
                detail = f"Price ${close:.2f} above BB upper ${upper:.2f}"
            elif pct_b < 0.0:
                label, icon, bullish = "Below lower band", "⚠️", True
                detail = f"Price ${close:.2f} below BB lower ${lower:.2f}"
            elif pct_b > 0.8:
                label, icon, bullish = "Near upper band", "🔶", False
                detail = f"Price near upper band (pct_b={pct_b:.2f})"
            elif pct_b < 0.2:
                label, icon, bullish = "Near lower band", "🔶", True
                detail = f"Price near lower band (pct_b={pct_b:.2f})"
            else:
                label, icon, bullish = "Mid-band", "➖", None
                detail = f"Price within bands (pct_b={pct_b:.2f})"

            signals.append({
                "indicator": "Bollinger Bands (20,2)",
                "value": round(pct_b, 2),
                "signal": label,
                "detail": detail,
                "icon": icon,
                "bullish": bullish,
            })

    # ── EMA trend ─────────────────────────────────────────────────────────────
    if "EMA_20" in df.columns and "EMA_50" in df.columns:
        ema20 = _latest(df, "EMA_20")
        ema50 = _latest(df, "EMA_50")
        if ema20 is not None and ema50 is not None:
            if ema20 > ema50:
                signals.append({
                    "indicator": "EMA 20/50",
                    "value": round(ema20 - ema50, 2),
                    "signal": "Uptrend",
                    "detail": f"EMA20 ${ema20:.2f} > EMA50 ${ema50:.2f}",
                    "icon": "✅",
                    "bullish": True,
                })
            else:
                signals.append({
                    "indicator": "EMA 20/50",
                    "value": round(ema20 - ema50, 2),
                    "signal": "Downtrend",
                    "detail": f"EMA20 ${ema20:.2f} < EMA50 ${ema50:.2f}",
                    "icon": "🔴",
                    "bullish": False,
                })

    return signals
