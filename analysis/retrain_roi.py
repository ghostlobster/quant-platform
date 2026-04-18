"""
analysis/retrain_roi.py — Retrain return-on-investment tracker.

Reads the per-model ``test_ic_delta`` history from the ``model_metadata``
table and exposes two helpers:

- ``retrain_roi(model_name, n=6)``     — last N deltas + linear-regression
                                          slope.
- ``is_ic_plateau(model_name, n=3)``   — True when the slope over the
                                          last N retrains is non-positive
                                          (i.e., retraining is not
                                          improving IC any more).

The ``KnowledgeAdaptionAgent`` consults ``is_ic_plateau`` to demote a
would-be ``fresh`` verdict to ``monitor`` when retraining is no longer
paying off — staleness is not the bottleneck, the feature set or
hyperparameters probably are (#122).
"""
from __future__ import annotations

import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)


def retrain_roi(model_name: str, n: int = 6) -> tuple[pd.Series, float]:
    """Return the last ``n`` ``test_ic_delta`` values for ``model_name``
    (chronological order) together with the slope of a simple linear fit.

    The slope is computed over an integer index 0..len-1 so it is roughly
    "delta-per-retrain". When fewer than 2 non-null deltas are available
    the slope is ``0.0``. Any DB error returns an empty series and a zero
    slope so callers never have to defensively catch.
    """
    try:
        from data.db import get_connection

        conn = get_connection()
    except Exception as exc:
        log.debug("retrain_roi: could not open DB: %s", exc)
        return pd.Series(dtype=float), 0.0

    try:
        rows = conn.execute(
            "SELECT trained_at, test_ic_delta FROM model_metadata "
            "WHERE model_name = ? AND test_ic_delta IS NOT NULL "
            "ORDER BY trained_at DESC LIMIT ?",
            (model_name, int(n)),
        ).fetchall()
    except Exception as exc:
        log.debug("retrain_roi: query failed: %s", exc)
        rows = []
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        return pd.Series(dtype=float), 0.0

    # rows came out DESC for LIMIT; reverse to chronological
    ordered = list(reversed(rows))
    series = pd.Series(
        [float(r["test_ic_delta"] if hasattr(r, "keys") else r[1]) for r in ordered],
        index=[
            float(r["trained_at"] if hasattr(r, "keys") else r[0]) for r in ordered
        ],
        dtype=float,
        name="test_ic_delta",
    )

    if len(series) < 2:
        return series, 0.0

    # Integer index for the slope so the scale is deltas-per-retrain.
    x = pd.Series(range(len(series)), dtype=float)
    y = series.reset_index(drop=True)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return series, 0.0
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denom)
    return series, slope


def is_ic_plateau(model_name: str, n: int = 3) -> bool:
    """Return True when the last ``n`` retrains show a non-positive slope.

    Non-positive rather than "negative" — a flat series (slope == 0) is
    already evidence that retraining is not earning its keep. Requires at
    least ``n`` non-null deltas to fire; otherwise returns False so the
    detector is silent until we have enough history.
    """
    series, slope = retrain_roi(model_name, n=n)
    if len(series) < n:
        return False
    return slope <= 0.0
