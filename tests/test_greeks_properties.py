"""Hypothesis property tests for ``analysis/greeks.py`` (closes #203).

The existing ``tests/test_greeks.py`` covers handpicked example values.
Example tests catch what the author thought of; they miss the inputs
the author *didn't* think of (deep-OTM strikes, tiny times to expiry,
negative rates, very high vols).

This file asserts the closed-form Black-Scholes invariants — properties
that must hold for every valid input — and lets ``hypothesis`` shrink
counter-examples when one is found:

  * **Put-call parity**: ``call − put == S − K · exp(−rT)``
  * **Delta bounds**: ``0 ≤ call_delta ≤ 1`` and ``−1 ≤ put_delta ≤ 0``
  * **Put-call delta**: ``call_delta − put_delta == 1``
  * **Gamma symmetry**: ``call_gamma == put_gamma``
  * **Vega symmetry**: ``call_vega == put_vega``
  * **Vega non-negative**: ``vega ≥ 0`` always

A deliberately-broken sign flip in any of the closed-form derivations
will be reported as a minimised failing case (e.g. ``S=100, K=100,
T=1, r=0, sigma=0.2``) — much more debuggable than a 5e-3 mismatch on
a hand-picked value.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from analysis.greeks import black_scholes_price, compute_greeks

# ── Input strategies ─────────────────────────────────────────────────────────
# Constrained to financially-meaningful ranges; ``hypothesis`` will explore
# the corners (deep-OTM / deep-ITM / very short / very long expiry).
spot = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
strike = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
expiry = st.floats(min_value=1e-3, max_value=5.0, allow_nan=False, allow_infinity=False)
rate = st.floats(min_value=-0.05, max_value=0.20, allow_nan=False, allow_infinity=False)
vol = st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False)


SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ── Put-call parity ──────────────────────────────────────────────────────────

@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_put_call_parity(S: float, K: float, T: float, r: float, sigma: float) -> None:
    """``call − put == S − K · exp(−rT)`` for any (S, K, T, r, σ).

    The Black-Scholes parity is independent of σ — it follows from the
    no-arbitrage replication argument alone. A failing case here points
    at a sign mistake in either the call or put pricing branch.
    """
    call = black_scholes_price(S, K, T, r, sigma, "call")
    put = black_scholes_price(S, K, T, r, sigma, "put")
    parity = S - K * math.exp(-r * T)
    # Tolerance: numerical CDF approximation is good to ~7.5e-8 per
    # call; the parity error is dominated by 2 CDF + 2 exp evaluations
    # at large S, so we allow 1e-4 relative.
    assert call - put == pytest.approx(parity, rel=1e-4, abs=1e-4)


# ── Delta bounds ──────────────────────────────────────────────────────────────

@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_call_delta_in_unit_interval(
    S: float, K: float, T: float, r: float, sigma: float
) -> None:
    """``call_delta = N(d1) ∈ [0, 1]`` for every (S, K, T, r, σ)."""
    g = compute_greeks(S, K, T, r, sigma, "call")
    assert 0.0 <= g.delta <= 1.0


@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_put_delta_in_negative_unit_interval(
    S: float, K: float, T: float, r: float, sigma: float
) -> None:
    """``put_delta = N(d1) − 1 ∈ [−1, 0]``."""
    g = compute_greeks(S, K, T, r, sigma, "put")
    assert -1.0 <= g.delta <= 0.0


@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_put_call_delta_difference_is_one(
    S: float, K: float, T: float, r: float, sigma: float
) -> None:
    """``call_delta − put_delta == 1`` (corollary of put-call parity)."""
    call = compute_greeks(S, K, T, r, sigma, "call")
    put = compute_greeks(S, K, T, r, sigma, "put")
    assert call.delta - put.delta == pytest.approx(1.0, abs=1e-9)


# ── Symmetry of second-order Greeks ──────────────────────────────────────────

@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_gamma_is_call_put_symmetric(
    S: float, K: float, T: float, r: float, sigma: float
) -> None:
    """Gamma is identical for call and put under Black-Scholes."""
    call = compute_greeks(S, K, T, r, sigma, "call")
    put = compute_greeks(S, K, T, r, sigma, "put")
    assert call.gamma == pytest.approx(put.gamma, abs=1e-12)


@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol)
@SETTINGS
def test_vega_is_call_put_symmetric(
    S: float, K: float, T: float, r: float, sigma: float
) -> None:
    """Vega is identical for call and put under Black-Scholes."""
    call = compute_greeks(S, K, T, r, sigma, "call")
    put = compute_greeks(S, K, T, r, sigma, "put")
    assert call.vega == pytest.approx(put.vega, abs=1e-12)


@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol, opt=st.sampled_from(["call", "put"]))
@SETTINGS
def test_vega_non_negative(
    S: float, K: float, T: float, r: float, sigma: float, opt: str
) -> None:
    """``vega = S · N'(d1) · √T ≥ 0`` always (long options gain from a
    rise in IV; ``S``, ``N'(d1)`` and ``√T`` are all non-negative)."""
    g = compute_greeks(S, K, T, r, sigma, opt)
    assert g.vega >= 0.0


@given(S=spot, K=strike, T=expiry, r=rate, sigma=vol, opt=st.sampled_from(["call", "put"]))
@SETTINGS
def test_gamma_non_negative(
    S: float, K: float, T: float, r: float, sigma: float, opt: str
) -> None:
    """Gamma is non-negative for long options (a long call/put is convex
    in the underlying)."""
    g = compute_greeks(S, K, T, r, sigma, opt)
    assert g.gamma >= 0.0
