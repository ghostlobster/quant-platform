"""
analysis/garch.py — Jansen Ch 9 GARCH volatility forecasting.

Realised volatility (``analysis.risk_metrics``) backs out of past
returns; it's a lagging indicator.  GARCH (Generalized Autoregressive
Conditional Heteroskedasticity) forecasts the *next-period* conditional
volatility and is a much better input for risk-scaled position sizing.
The executed strategy layer can divide a target volatility by the GARCH
forecast to shrink Kelly bets when conditional vol is elevated.

Optional deps
-------------
    arch >= 7.0.0  (``pip install arch>=7.0.0``)

When ``arch`` isn't installed, :func:`fit_garch` raises a clear
``RuntimeError`` and :func:`forecast_next_sigma` returns ``None`` so
calling code can fall back to realised-vol without special-casing.

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading*, Ch 9.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from arch import arch_model  # type: ignore[import]
    _ARCH_AVAILABLE = True
except ImportError:
    arch_model = None  # type: ignore[assignment]
    _ARCH_AVAILABLE = False


@dataclass
class GarchForecast:
    """Per-run output of :func:`fit_garch`.

    Attributes
    ----------
    fit :
        The raw ``arch.univariate.GARCHResults`` (or ``None`` if the
        fit failed / ``arch`` isn't installed).  Exposed so callers can
        inspect the fit object directly.
    sigma_next :
        One-step-ahead conditional volatility in the same units as
        ``returns``.  ``None`` when the fit failed.
    converged :
        ``True`` if the optimiser flagged convergence, else ``False``.
    """

    fit: object = None
    sigma_next: Optional[float] = None
    converged: bool = False


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: str = "Zero",
    vol: str = "GARCH",
    dist: str = "normal",
    rescale: bool = True,
) -> GarchForecast:
    """Fit a GARCH(p, q) model and return a one-step volatility forecast.

    Parameters
    ----------
    returns :
        ``pd.Series`` of *return* observations (typically daily log
        returns).  Must contain ``>= 30`` points to produce a stable
        fit.
    p, q :
        GARCH lag orders.  ``p`` is the ARCH order (squared residuals)
        and ``q`` is the GARCH order (lagged variances).  Default
        GARCH(1,1).
    mean, vol, dist :
        Forwarded to :func:`arch.arch_model`.  Defaults match AFML /
        Jansen recipes.
    rescale :
        Let ``arch`` rescale the input to unit variance internally —
        prevents numerical issues when returns are in decimal form
        (e.g. ``0.01`` per bar).

    Returns
    -------
    :class:`GarchForecast`.  When the fit fails or ``arch`` isn't
    installed, ``sigma_next`` is ``None`` and ``converged`` is
    ``False``.

    Raises
    ------
    RuntimeError if ``arch`` is not installed.
    """
    if not _ARCH_AVAILABLE:
        raise RuntimeError(
            "arch is not installed.  Run: pip install arch>=7.0.0"
        )
    if returns is None or returns.empty or returns.isna().all():
        return GarchForecast()
    clean = returns.dropna()
    if len(clean) < 30:
        return GarchForecast()

    try:
        model = arch_model(
            clean, mean=mean, vol=vol, p=p, q=q, dist=dist, rescale=rescale,
        )
        fit = model.fit(disp="off", show_warning=False)
    except Exception:
        return GarchForecast()

    # One-step-ahead variance forecast, convert to sigma.
    try:
        forecast = fit.forecast(horizon=1, reindex=False)
        var_next = float(forecast.variance.values[-1, 0])
    except Exception:
        return GarchForecast(fit=fit, converged=False)

    # ``arch`` rescales internally when rescale=True; undo the scale so
    # sigma_next lives in the same units as the input returns.
    scale = getattr(fit, "scale", 1.0) or 1.0
    sigma_next = float(np.sqrt(max(0.0, var_next))) / float(scale)
    return GarchForecast(fit=fit, sigma_next=sigma_next, converged=True)


def forecast_next_sigma(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> Optional[float]:
    """Thin wrapper returning just the ``sigma_next`` float.

    Designed for callers that don't need the fit object — e.g. the
    execution layer that scales Kelly by ``sigma_target / sigma_next``.
    Returns ``None`` on any failure or when ``arch`` isn't installed.
    """
    if not _ARCH_AVAILABLE:
        return None
    try:
        result = fit_garch(returns, p=p, q=q)
    except RuntimeError:
        return None
    return result.sigma_next
