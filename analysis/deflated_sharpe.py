"""
analysis/deflated_sharpe.py — AFML Ch 11 backtest-overfitting guards.

After testing ``N`` strategy variants the best Sharpe ratio looks
remarkable only because selection bias is absorbed into the estimator.
López de Prado's Deflated Sharpe Ratio (DSR, Eq 11.6) adjusts for
selection, sample size, skewness, and kurtosis and returns the
*probability* that the observed Sharpe is genuinely above zero.

The Probability of Backtest Overfitting (PBO, Eq 11.9) is a separate
diagnostic that estimates — from a matrix of in-sample vs out-of-sample
performance across many trials — how often the trial that looks best
in-sample underperforms out-of-sample.

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Ch 11.4-11.5.
"""
from __future__ import annotations

from math import log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm

# Euler-Mascheroni constant, per AFML Eq 11.5.
_EULER_MASCHERONI = 0.577215664901532
_MAX_EXPECTED_SHARPE_APPROX = 250.0   # numerical guard for extreme n_trials


def _expected_maximum_sharpe(n_trials: int, variance: float = 1.0) -> float:
    """AFML Eq 11.5: expected maximum of N i.i.d. Sharpe estimates under
    a null of zero true Sharpe.

    Uses the closed-form approximation from López de Prado's reference
    code (based on the expected value of the maximum order statistic of
    ``N`` standard normals).
    """
    if n_trials <= 1:
        return 0.0
    sd = sqrt(variance)
    # Two-term approximation from AFML Code Snippet 20.4.
    n = max(2, int(n_trials))
    a = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
    b = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * (a + b))


def deflated_sharpe(
    sharpe: float,
    n_trials: int,
    skew: float,
    kurt: float,
    num_obs: int,
    sharpe_variance: float = 1.0,
) -> float:
    """Probability that the true Sharpe exceeds zero given a selection bias.

    Implements AFML Eq 11.6 — the Deflated Sharpe Ratio.  Returns a
    *probability* in ``[0, 1]``, not a ratio; treat anything below
    ``0.05`` as "likely backtest overfitting".

    Parameters
    ----------
    sharpe :
        Observed (annualised or per-period; consistency matters only
        inside this call) Sharpe ratio of the best trial.
    n_trials :
        Total number of strategy variants tried (hyperparameter grid,
        combinatorial CV paths, etc.).
    skew :
        Skewness of the returns of the best trial.
    kurt :
        *Excess* kurtosis (``kurt(normal) == 0``).
    num_obs :
        Number of return observations used to estimate ``sharpe``.
    sharpe_variance :
        Variance across trial Sharpes.  Defaults to ``1``, which is
        the standard null-hypothesis convention.

    Returns
    -------
    ``float`` — ``P(true_sharpe > 0 | observed)``.  Returns ``0.0``
    when the formula is not evaluable (e.g. ``num_obs < 2``).
    """
    if num_obs < 2:
        return 0.0

    # Expected maximum of n_trials independent Sharpe estimates under H0.
    if n_trials > _MAX_EXPECTED_SHARPE_APPROX ** 2:
        expected_max = sqrt(sharpe_variance) * sqrt(2 * log(n_trials))
    else:
        expected_max = _expected_maximum_sharpe(n_trials, sharpe_variance)

    # AFML Eq 11.6 denominator uses *non-excess* kurtosis γ_4 = kurt + 3,
    # so the term becomes (γ_4 - 1) / 4 = (kurt_excess + 2) / 4.
    numerator = (sharpe - expected_max) * sqrt(num_obs - 1)
    denom_sq = 1.0 - skew * sharpe + ((kurt + 2.0) / 4.0) * sharpe * sharpe
    denominator = sqrt(max(1e-12, denom_sq))
    z = numerator / denominator
    return float(norm.cdf(z))


def _rank_logit(ranks: np.ndarray, n_trials: int) -> np.ndarray:
    """Map integer ranks in ``[1, n_trials]`` to a logit-transformed
    relative performance.

    AFML Eq 11.9 defines PBO off the logit of the OOS rank of the
    in-sample best trial.  Ranks are converted to quantiles
    ``(rank - 0.5) / n_trials`` first — that's the standard
    continuity-corrected empirical CDF value.
    """
    q = (np.asarray(ranks, dtype=float) - 0.5) / max(1, n_trials)
    q = np.clip(q, 1e-9, 1.0 - 1e-9)
    return np.log(q / (1.0 - q))


def probability_backtest_overfitting(
    is_oos_matrix: pd.DataFrame,
) -> float:
    """Combinatorially-symmetric Probability of Backtest Overfitting.

    AFML Eq 11.9.  Given an ``N_splits × N_trials`` DataFrame where
    each row is one CV split's ranking of every trial's performance,
    we compute, for each split, the OOS rank of the IS-best trial.
    PBO is the fraction of splits where that OOS rank is below median.

    Input convention: each row of ``is_oos_matrix`` represents the
    **out-of-sample** performance of every trial on one CV split.
    The "in-sample" ranking is taken as the mean performance across
    the other splits — equivalent to AFML's combinatorially-symmetric
    leave-one-out procedure.

    Returns
    -------
    ``float`` ∈ ``[0, 1]``.  A PBO above ``0.5`` means the apparent
    winner rarely wins out-of-sample → backtest overfitting likely.
    """
    if is_oos_matrix is None or is_oos_matrix.empty:
        return float("nan")

    mat = is_oos_matrix.to_numpy(dtype=float, copy=True)
    n_splits, n_trials = mat.shape
    if n_splits < 2 or n_trials < 2:
        return float("nan")

    drops = 0
    total = 0
    for i in range(n_splits):
        is_mask = np.ones(n_splits, dtype=bool)
        is_mask[i] = False
        is_means = mat[is_mask].mean(axis=0)       # IS performance per trial
        best_trial = int(np.argmax(is_means))
        oos_perf = mat[i]
        # OOS rank of the best-IS trial (1 = worst, n_trials = best).
        oos_rank = (oos_perf < oos_perf[best_trial]).sum() + 1
        logit = _rank_logit(np.array([oos_rank]), n_trials=n_trials)[0]
        if logit < 0:
            drops += 1
        total += 1

    return drops / total if total > 0 else float("nan")


def deflated_sharpe_warning(dsr_probability: float, threshold: float = 0.05) -> str:
    """Short human-readable interpretation of a DSR probability."""
    if np.isnan(dsr_probability):
        return "insufficient data for DSR"
    if dsr_probability < threshold:
        return (
            f"DSR P(Sharpe>0)={dsr_probability:.3f} — below {threshold:.2f}; "
            "result may be spurious (backtest overfitting suspected)"
        )
    return f"DSR P(Sharpe>0)={dsr_probability:.3f} — passes overfitting guard"
