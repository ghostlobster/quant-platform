"""Value at Risk calculations — Historical and Parametric."""

import numpy as np
import pandas as pd


def _norm_ppf(p: float) -> float:
    """Rational approximation of the normal quantile function (Beasley-Springer-Moro)."""
    try:
        from scipy import stats
        return float(stats.norm.ppf(p))
    except ImportError:
        pass
    # Abramowitz & Stegun approximation, good to ~4.5e-4
    a = [2.515517, 0.802853, 0.010328]
    b = [1.432788, 0.189269, 0.001308]
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"p must be in (0,1), got {p}")
    # For p < 0.5 the quantile is negative; fold to lower tail, apply sign after.
    sign = 1.0
    q = p
    if q > 0.5:
        q = 1.0 - q
    else:
        sign = -1.0
    t = np.sqrt(-2.0 * np.log(q))
    num = a[0] + a[1] * t + a[2] * t**2
    den = 1.0 + b[0] * t + b[1] * t**2 + b[2] * t**3
    return sign * (t - num / den)


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR: percentile of observed return distribution.
    Returns a positive number representing potential loss (e.g. 0.032 = 3.2% loss).
    """
    if returns.empty or len(returns) < 30:
        return 0.0
    cutoff = np.percentile(returns.dropna(), (1 - confidence) * 100)
    return float(-cutoff)  # flip sign so positive = loss


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR using mean and std of returns.
    Returns a positive number representing potential loss.
    """
    if returns.empty or len(returns) < 30:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    z = _norm_ppf(1 - confidence)
    return float(-(mu + z * sigma))


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """CVaR / Expected Shortfall — average loss beyond VaR threshold."""
    if returns.empty or len(returns) < 30:
        return 0.0
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    return float(-tail.mean()) if not tail.empty else var


def portfolio_var(weights: np.ndarray, cov_matrix: np.ndarray, confidence: float = 0.95) -> float:
    """Parametric portfolio VaR using covariance matrix."""
    portfolio_variance = weights @ cov_matrix @ weights
    portfolio_std = np.sqrt(portfolio_variance)
    z = _norm_ppf(1 - confidence)
    return float(-z * portfolio_std)
