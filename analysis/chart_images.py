"""
analysis/chart_images.py — turn OHLCV windows into 2-D image tensors.

Companion to :mod:`strategies.cnn_signal` (Jansen Ch 18).  Two
transforms are provided:

  * :func:`to_gramian_angular_field` — GAF / GASF image of a 1-D price
    series (López de Prado-friendly visualisation, used heavily by
    Jansen Ch 18.4 as a CNN input).
  * :func:`ohlc_to_pixels` — direct rasterisation of OHLC bars onto a
    fixed-size grid; cheaper than GAF but preserves price ranges.

Both return ``(H, W)`` ``np.float32`` arrays in ``[0, 1]`` (or
``[-1, 1]`` for the signed GAF) so callers can stack them into the
``(n_samples, 1, H, W)`` tensor a Conv2d head expects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def to_gramian_angular_field(series: pd.Series | np.ndarray, window: int) -> np.ndarray:
    """Gramian Angular Summation Field (GASF) of the trailing ``window`` values.

    Steps (Jansen Ch 18.4.1):

    1. Take the last ``window`` observations.
    2. Min-max scale into ``[-1, 1]``.
    3. Encode each scaled value as the angle ``φ = arccos(x)``.
    4. Build the ``window × window`` matrix ``cos(φ_i + φ_j)``.

    Returns
    -------
    np.ndarray
        ``(window, window)`` float32 image in ``[-1, 1]``. All-NaN
        windows yield a zero matrix; constant windows yield a matrix of
        ones (degenerate but well-defined).
    """
    arr = np.asarray(pd.Series(series).to_numpy(dtype=float))[-int(window):]
    if arr.size < window:
        return np.zeros((window, window), dtype=np.float32)
    if not np.isfinite(arr).any():
        return np.zeros((window, window), dtype=np.float32)

    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi == lo:
        return np.ones((window, window), dtype=np.float32)

    scaled = 2.0 * (arr - lo) / (hi - lo) - 1.0
    scaled = np.clip(scaled, -1.0, 1.0)
    phi = np.arccos(scaled)
    gasf = np.cos(phi[:, None] + phi[None, :])
    return gasf.astype(np.float32)


def ohlc_to_pixels(df: pd.DataFrame, window: int, height: int = 32) -> np.ndarray:
    """Rasterise the trailing ``window`` OHLC bars onto a ``(height, window)`` grid.

    Each column is one bar; pixel rows are filled between the bar's
    Low and High (the candle wick) with a darker stripe between Open
    and Close (the body). Output is float32 in ``[0, 1]``.
    """
    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(df.columns) or len(df) < window:
        return np.zeros((height, window), dtype=np.float32)

    sub = df[list(needed)].iloc[-int(window):].astype(float)
    lo = float(sub["Low"].min())
    hi = float(sub["High"].max())
    if hi == lo:
        return np.zeros((height, window), dtype=np.float32)

    def _row(value: float) -> int:
        # Higher prices live near the top of the image (row 0 is top).
        frac = (value - lo) / (hi - lo)
        return int(np.clip(round((1.0 - frac) * (height - 1)), 0, height - 1))

    img = np.zeros((height, window), dtype=np.float32)
    for col, (_, row) in enumerate(sub.iterrows()):
        wick_top = _row(float(row["High"]))
        wick_bot = _row(float(row["Low"]))
        body_top = _row(max(float(row["Open"]), float(row["Close"])))
        body_bot = _row(min(float(row["Open"]), float(row["Close"])))
        img[wick_top : wick_bot + 1, col] = 0.5      # wick
        img[body_top : body_bot + 1, col] = 1.0      # body (darker)
    return img
