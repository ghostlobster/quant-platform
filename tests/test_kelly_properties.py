"""Hypothesis property tests for ``risk/kelly.py`` (closes #203).

Asserts the invariants ``kelly_fraction`` and ``kelly_from_backtest``
must hold for every valid input — in particular the bounds, monotonicity,
and the zero-edge corner. Counter-examples are auto-shrunk by hypothesis.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from risk.kelly import kelly_fraction, kelly_from_backtest

# ── Strategies ──────────────────────────────────────────────────────────────

# Win rate strictly in (0, 1) — endpoints are early-return-zero in the impl.
win_rate = st.floats(
    min_value=1e-6, max_value=1 - 1e-6, allow_nan=False, allow_infinity=False
)
# Wins / losses in plausible per-trade ranges (0.1 % to 50 %).
edge = st.floats(min_value=1e-4, max_value=0.5, allow_nan=False, allow_infinity=False)
# Cap fractions allowed by the function.
cap = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)


SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ── kelly_fraction bounds ───────────────────────────────────────────────────

@given(p=win_rate, w=edge, ell=edge, c=cap)
@SETTINGS
def test_kelly_fraction_bounded_by_zero_and_cap(
    p: float, w: float, ell: float, c: float
) -> None:
    """``0 ≤ kelly ≤ max_fraction`` for every (p, w, l, cap)."""
    k = kelly_fraction(p, w, ell, max_fraction=c)
    assert 0.0 <= k <= c


@given(p=win_rate, w=edge, ell=edge, c=cap)
@SETTINGS
def test_kelly_fraction_is_float(p: float, w: float, ell: float, c: float) -> None:
    """Always returns a Python ``float`` — keeps downstream JSON-serialisable."""
    k = kelly_fraction(p, w, ell, max_fraction=c)
    assert isinstance(k, float)


# ── Boundary / domain-error inputs ──────────────────────────────────────────

@given(w=edge, ell=edge, c=cap)
@SETTINGS
def test_zero_or_unit_win_rate_returns_zero(w: float, ell: float, c: float) -> None:
    """Endpoints win_rate ∈ {0, 1} are guarded — no bet."""
    assert kelly_fraction(0.0, w, ell, max_fraction=c) == 0.0
    assert kelly_fraction(1.0, w, ell, max_fraction=c) == 0.0


@given(p=win_rate, w=edge, c=cap)
@SETTINGS
def test_zero_avg_loss_returns_zero(p: float, w: float, c: float) -> None:
    """``avg_loss == 0`` is the divide-by-zero guard."""
    assert kelly_fraction(p, w, 0.0, max_fraction=c) == 0.0


# ── Monotonicity ────────────────────────────────────────────────────────────

@given(p=win_rate, ell=edge)
@SETTINGS
def test_kelly_non_decreasing_in_win_rate(p: float, ell: float) -> None:
    """Holding payoff fixed, increasing the win rate must not decrease
    the recommended fraction."""
    w = 0.05  # fixed
    delta = min(0.001, (1 - p) / 2, p / 2)
    k_lo = kelly_fraction(p - delta, w, ell, max_fraction=1.0)
    k_hi = kelly_fraction(p + delta, w, ell, max_fraction=1.0)
    # Tolerate the cap (both clamped to 1.0 → equal).
    assert k_hi >= k_lo - 1e-12


@given(w_lo=edge, w_hi=edge, p=win_rate, ell=edge)
@SETTINGS
def test_kelly_non_decreasing_in_avg_win(
    w_lo: float, w_hi: float, p: float, ell: float
) -> None:
    """Holding (p, l) fixed, larger avg_win → at least as much Kelly."""
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo
    k_lo = kelly_fraction(p, w_lo, ell, max_fraction=1.0)
    k_hi = kelly_fraction(p, w_hi, ell, max_fraction=1.0)
    assert k_hi >= k_lo - 1e-12


# ── Zero-edge / negative-edge corners ───────────────────────────────────────

def test_zero_edge_returns_zero() -> None:
    """50/50 with even payoff → no bet (full Kelly = 0, clamped)."""
    assert kelly_fraction(0.5, 0.05, 0.05) == 0.0


def test_negative_edge_returns_zero() -> None:
    """``avg_win < avg_loss`` and ``win_rate ≤ 0.5`` ⇒ negative edge ⇒
    Kelly is negative ⇒ clamped to 0."""
    assert kelly_fraction(0.4, 0.03, 0.05) == 0.0
    assert kelly_fraction(0.45, 0.04, 0.06) == 0.0


def test_strong_edge_clamps_to_cap() -> None:
    """A 90 % win rate with 10:1 payoff should hit the cap."""
    k = kelly_fraction(0.9, 1.0, 0.1, max_fraction=0.25)
    assert k == 0.25


# ── kelly_from_backtest ─────────────────────────────────────────────────────

@given(p=win_rate, ret=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False),
       n=st.integers(min_value=1, max_value=10_000), c=cap)
@SETTINGS
def test_kelly_from_backtest_bounded(p: float, ret: float, n: int, c: float) -> None:
    k = kelly_from_backtest(ret, n, p, max_fraction=c)
    assert 0.0 <= k <= c


def test_kelly_from_backtest_zero_trades_returns_zero() -> None:
    """Zero trades → no info → no bet."""
    assert kelly_from_backtest(0.5, 0, 0.6) == 0.0


def test_kelly_from_backtest_zero_win_rate_returns_zero() -> None:
    assert kelly_from_backtest(0.5, 100, 0.0) == 0.0


def test_kelly_from_backtest_unit_win_rate_returns_zero() -> None:
    """``win_rate == 1`` ⇒ all wins ⇒ ``loss_trades == 0`` early-return."""
    assert kelly_from_backtest(0.5, 100, 1.0) == 0.0
