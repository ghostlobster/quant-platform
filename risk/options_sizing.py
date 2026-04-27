"""
risk/options_sizing.py — Greeks-aware sizing for multi-leg options.

Two helpers:

    delta_neutral_qty(legs, market) -> int
        Pick the contract count so the portfolio delta exposure stays
        within ``max_abs_delta_shares`` (default 100 shares-equivalent).

    cap_by_max_vega(legs, market, max_vega_dollars) -> int
        Pick the contract count so the absolute portfolio vega (in $ per
        1 vol point) stays within the cap.

Both helpers reuse :func:`analysis.greeks.compute_greeks` so the math is
the project's canonical Black-Scholes path. Each leg in the input list
must come from :mod:`strategies.options_legs` and the ``market`` mapping
must provide ``S`` (underlying spot) and an ``r`` (risk-free rate)
keyed by ``leg.underlying``, plus per-leg ``T`` (years to expiry) and
``sigma`` (implied vol).
"""
from __future__ import annotations

from dataclasses import dataclass

from analysis.greeks import compute_greeks
from strategies.options_legs import OptionLeg


@dataclass(frozen=True)
class LegMarket:
    """Per-leg market inputs for Black-Scholes."""

    S: float           # underlying spot
    T: float           # years to expiry
    sigma: float       # implied vol (annualised)
    r: float = 0.04    # risk-free rate
    contract_price: float | None = None  # observed mid for IV solve / pricing


def _portfolio_greeks(legs: list[OptionLeg], market: dict[int, LegMarket]) -> dict:
    """Sum delta (in shares) and vega (in $/vol-point) over signed legs."""
    total_delta_shares = 0.0
    total_vega = 0.0
    for idx, leg in enumerate(legs):
        m = market[idx]
        g = compute_greeks(
            S=m.S, K=leg.strike, T=m.T, r=m.r,
            sigma=m.sigma, option_type=leg.option_type,
            contract_price=m.contract_price,
        )
        # delta * signed_qty * 100 = exposure in shares-equivalent
        total_delta_shares += g.delta * leg.signed_qty * 100.0
        total_vega += g.vega * leg.signed_qty
    return {"delta_shares": total_delta_shares, "vega": total_vega}


def delta_neutral_qty(
    legs: list[OptionLeg],
    market: dict[int, LegMarket],
    *,
    max_abs_delta_shares: float = 100.0,
) -> int:
    """Largest integer multiplier ``m`` so |portfolio delta × m| ≤ cap.

    Returns ``0`` when even a single contract per leg breaches the cap.
    Useful for premium-harvest structures (short straddle, iron condor)
    that should sit close to delta-flat.
    """
    if max_abs_delta_shares <= 0:
        raise ValueError(
            f"max_abs_delta_shares must be > 0, got {max_abs_delta_shares}",
        )
    base = _portfolio_greeks(legs, market)
    delta_one = abs(base["delta_shares"])
    if delta_one == 0:
        # Already delta-neutral at qty=1 — caller can scale up freely.
        return max(1, max(int(leg.qty) for leg in legs))
    multiplier = max_abs_delta_shares / delta_one
    return int(multiplier)  # truncate toward zero


def cap_by_max_vega(
    legs: list[OptionLeg],
    market: dict[int, LegMarket],
    *,
    max_vega_dollars: float,
) -> int:
    """Largest integer multiplier ``m`` so |portfolio vega × m| ≤ cap.

    Vega is in dollars per 1.0 IV point (matches
    :func:`analysis.greeks.compute_greeks` convention). Returns ``0``
    when one contract per leg already exceeds the cap.
    """
    if max_vega_dollars <= 0:
        raise ValueError(
            f"max_vega_dollars must be > 0, got {max_vega_dollars}",
        )
    base = _portfolio_greeks(legs, market)
    vega_one = abs(base["vega"])
    if vega_one == 0:
        return max(1, max(int(leg.qty) for leg in legs))
    multiplier = max_vega_dollars / vega_one
    return int(multiplier)


def scale_legs(legs: list[OptionLeg], multiplier: int) -> list[OptionLeg]:
    """Return a fresh leg list with ``leg.qty *= multiplier``.

    Convenience wrapper for the sizer outputs. Multiplier ``0`` returns
    an empty list — caller decides whether to abort or shrink the
    construct.
    """
    if multiplier <= 0:
        return []
    return [
        OptionLeg(
            underlying=leg.underlying, expiry=leg.expiry,
            strike=leg.strike, option_type=leg.option_type,
            side=leg.side, qty=leg.qty * multiplier,
            option_symbol=leg.option_symbol,
        )
        for leg in legs
    ]
