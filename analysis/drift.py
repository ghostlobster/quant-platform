"""
analysis/drift.py — Covariate-shift detection for alpha-model features.

Provides three pure helpers (no DB, no filesystem, no global state) that
feed `KnowledgeAdaptionAgent`'s drift rung:

1. :func:`summarize_features` — compact per-column fingerprint
   (mean, std, quantiles, sample count) persisted at training time in
   ``model_feature_stats`` by :func:`strategies.ml_signal.MLSignal._write_feature_stats`.

2. :func:`feature_psi` — Population Stability Index per feature,
   computed by reconstructing bin edges from the stored fingerprint
   (quantile anchors + ±3σ tails) and scoring the live distribution
   mass against the expected training mass.

3. :func:`kolmogorov_smirnov` — thin wrapper around
   :func:`scipy.stats.ks_2samp` so callers that still have both raw
   samples can run the classical test.

4. :func:`aggregate_drift` — collapse a per-feature PSI dict into the
   verdict tier the knowledge agent consumes (`none / monitor /
   retrain`) plus the list of drifted feature names.

The PSI computation uses the stored fingerprint, not the raw training
sample, so it is an approximation — accurate enough to surface the
shifts the agent cares about (shifts that would move the verdict by a
tier) but not precise enough for an academic PSI calibration. Tests
validate the approximation error stays below 10 % of the exact PSI on
synthetic Gaussian shifts.

References
----------
- López de Prado, *Advances in Financial Machine Learning*, Ch 17
  (structural breaks & covariate drift).
- Jansen, *Machine Learning for Algorithmic Trading*, Ch 17
  (monitoring shift in production features).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_SAMPLES = 30
_DEFAULT_N_BINS = 10
_EPS = 1e-6

_DEFAULT_PSI_MONITOR = 0.10
_DEFAULT_PSI_RETRAIN = 0.25


# ── Feature fingerprint ──────────────────────────────────────────────────────

def summarize_features(
    frame: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, dict[str, float]]:
    """Return ``{feature_name: {mean, std, q10, q50, q90, n_samples}}``.

    Columns that are missing from ``frame`` or have fewer than
    ``_MIN_SAMPLES`` non-NaN rows are silently dropped. Constant columns
    (``std == 0``) are kept — the live comparator will either match
    exactly (PSI ≈ 0) or produce a large bin-edge-collapse PSI that
    still surfaces the shift.
    """
    stats: dict[str, dict[str, float]] = {}
    for col in feature_cols:
        if col not in frame.columns:
            continue
        series = frame[col].dropna()
        if len(series) < _MIN_SAMPLES:
            continue
        arr = np.asarray(series.to_numpy(dtype=float))
        q10, q50, q90 = np.quantile(arr, [0.10, 0.50, 0.90])
        stats[col] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "q10": float(q10),
            "q50": float(q50),
            "q90": float(q90),
            "n_samples": int(len(arr)),
        }
    return stats


# ── PSI ──────────────────────────────────────────────────────────────────────

def _bin_edges_from_fingerprint(
    stats: dict[str, float],
    n_bins: int = _DEFAULT_N_BINS,
) -> np.ndarray:
    """Reconstruct ``n_bins`` ordered bin edges from a fingerprint.

    Uses the quantile anchors (q10, q50, q90) plus ±3σ tails as explicit
    cut points, then fills the remaining edges with linear interpolation
    between them. The first and last edges are ``-inf`` / ``+inf`` so
    live samples beyond the training tails still land in a bin. Returns
    an array of length ``n_bins + 1``.
    """
    mean = stats["mean"]
    std = max(stats["std"], _EPS)
    q10, q50, q90 = stats["q10"], stats["q50"], stats["q90"]

    anchors = [
        mean - 3.0 * std,
        q10,
        (q10 + q50) / 2.0,
        q50,
        (q50 + q90) / 2.0,
        q90,
        mean + 3.0 * std,
    ]
    anchors = sorted(set(float(a) for a in anchors if not math.isnan(a)))

    if len(anchors) < 2:
        # Degenerate fingerprint (all anchors collapsed) — fall back to
        # ±3σ around the mean.
        anchors = [mean - 3.0 * std, mean + 3.0 * std]

    # Interpolate to exactly n_bins - 1 interior edges (so len(edges) == n_bins + 1
    # after adding ±inf sentinels).
    interior = np.interp(
        np.linspace(0.0, 1.0, n_bins - 1),
        np.linspace(0.0, 1.0, len(anchors)),
        anchors,
    )
    edges = np.concatenate([[-np.inf], interior, [np.inf]])
    # Monotonicity guard (interp can emit dupes when anchors bunch).
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges


def _training_mass_for_edges(
    stats: dict[str, float],
    edges: np.ndarray,
) -> np.ndarray:
    """Expected bin mass under the training fingerprint's Gaussian-ish model.

    Uses the (mean, std) normal approximation — the fingerprint does not
    carry enough information to reproduce the raw training mass
    exactly. Good enough for drift detection: the ratio of live vs
    expected mass is what PSI cares about, and the approximation error
    largely cancels when the live sample is itself near the training
    distribution.
    """
    mean = stats["mean"]
    std = max(stats["std"], _EPS)
    from math import erf, sqrt

    def _cdf(x: float) -> float:
        if math.isinf(x):
            return 0.0 if x < 0 else 1.0
        return 0.5 * (1.0 + erf((x - mean) / (std * sqrt(2.0))))

    cdfs = np.array([_cdf(float(e)) for e in edges])
    mass = np.diff(cdfs)
    # Floor tiny negatives from floating-point noise.
    mass = np.clip(mass, _EPS, None)
    mass = mass / mass.sum()
    return mass


def feature_psi(
    training_stats: dict[str, dict[str, float]],
    live_frame: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_bins: int = _DEFAULT_N_BINS,
) -> dict[str, float]:
    """Return ``{feature_name: psi}``.

    Features with fewer than ``_MIN_SAMPLES`` live observations map to
    ``float('nan')``. Features absent from the training fingerprint are
    skipped.
    """
    cols = feature_cols or list(training_stats.keys())
    scores: dict[str, float] = {}
    for col in cols:
        stats = training_stats.get(col)
        if stats is None:
            continue
        if col not in live_frame.columns:
            scores[col] = float("nan")
            continue
        live = live_frame[col].dropna().to_numpy(dtype=float)
        if live.size < _MIN_SAMPLES:
            scores[col] = float("nan")
            continue

        edges = _bin_edges_from_fingerprint(stats, n_bins=n_bins)
        expected = _training_mass_for_edges(stats, edges)

        live_counts, _ = np.histogram(live, bins=edges)
        observed = live_counts.astype(float) / max(live.size, 1)
        observed = np.clip(observed, _EPS, None)
        observed = observed / observed.sum()

        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        scores[col] = psi
    return scores


# ── KS test ──────────────────────────────────────────────────────────────────

def kolmogorov_smirnov(
    training_col: np.ndarray,
    live_col: np.ndarray,
) -> tuple[float, float]:
    """Thin wrapper around ``scipy.stats.ks_2samp``.

    Drops NaNs from both inputs. Returns ``(statistic, pvalue)``. When
    either side has fewer than 2 non-NaN samples, returns ``(nan, nan)``.
    """
    from scipy.stats import ks_2samp

    train = np.asarray(training_col, dtype=float)
    live = np.asarray(live_col, dtype=float)
    train = train[~np.isnan(train)]
    live = live[~np.isnan(live)]
    if train.size < 2 or live.size < 2:
        return (float("nan"), float("nan"))
    res = ks_2samp(train, live)
    return (float(res.statistic), float(res.pvalue))


# ── Verdict aggregator ───────────────────────────────────────────────────────

def aggregate_drift(
    psi_scores: dict[str, float],
    *,
    psi_monitor: float = _DEFAULT_PSI_MONITOR,
    psi_retrain: float = _DEFAULT_PSI_RETRAIN,
) -> dict[str, Any]:
    """Collapse a per-feature PSI dict into the agent-facing verdict.

    Returns ``{"level": "none" | "monitor" | "retrain",
               "max_psi": float | None,
               "drifted_features": list[str]}``.

    * ``retrain`` — any single feature with PSI ≥ ``psi_retrain``.
    * ``monitor`` — any single feature with PSI ≥ ``psi_monitor`` but
      none at the retrain threshold.
    * ``none`` — every PSI below ``psi_monitor`` (NaNs ignored).

    ``drifted_features`` lists every feature that crossed
    ``psi_monitor``, sorted by descending PSI so operators see the
    worst offenders first.
    """
    if not psi_scores:
        return {"level": "none", "max_psi": None, "drifted_features": []}

    valid = {
        k: v for k, v in psi_scores.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
    if not valid:
        return {"level": "none", "max_psi": None, "drifted_features": []}

    max_psi = max(valid.values())
    drifted = sorted(
        [k for k, v in valid.items() if v >= psi_monitor],
        key=lambda k: valid[k],
        reverse=True,
    )
    if max_psi >= psi_retrain:
        level = "retrain"
    elif max_psi >= psi_monitor:
        level = "monitor"
    else:
        level = "none"
    return {
        "level": level,
        "max_psi": float(max_psi),
        "drifted_features": drifted,
    }
