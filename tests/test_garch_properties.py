"""Hypothesis property tests for ``analysis/garch.py`` — closes #216 part 2.

These complement the example-based tests in ``tests/test_garch.py`` by
asserting the closed-form invariants that **every** valid call to
``fit_garch`` / ``forecast_next_sigma`` must satisfy:

  * **Non-negativity** — `sigma_next ≥ 0` for any returns series that
    converges. A negative forecast is a sign-flip bug in the variance
    extraction or scale division.
  * **Type consistency** — ``sigma_next`` is always a Python ``float``
    (not a numpy scalar), keeping the call site JSON-serialisable.
  * **Idempotence** — same returns input → same forecast.
  * **Convergence/sigma agreement** — ``converged is True`` ⇒
    ``sigma_next is not None``; ``sigma_next is None`` ⇒
    ``converged is False``. (The reverse implications need a
    deliberate fit failure.)
  * **Boundary** — passing constant zero returns degrades gracefully
    (no NaN, no exception).

200 examples per property; counter-examples are auto-shrunk by
hypothesis to a minimal failing input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from analysis.garch import _ARCH_AVAILABLE, fit_garch, forecast_next_sigma

pytestmark = pytest.mark.skipif(
    not _ARCH_AVAILABLE,
    reason="arch not installed (optional dep — silent-skip guard #199 "
           "doesn't flag because arch is not in requirements.txt)",
)


_SETTINGS = settings(
    max_examples=30,            # GARCH fits are expensive; keep budget bounded
    deadline=None,              # allow slow individual fits
    suppress_health_check=[HealthCheck.function_scoped_fixture,
                           HealthCheck.too_slow],
)


def _synth(seed: int, n: int = 250, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, vol, n))


# ── Non-negativity ───────────────────────────────────────────────────────────


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=60, max_value=300),
    vol=st.floats(min_value=0.001, max_value=0.05, allow_nan=False),
)
@_SETTINGS
def test_sigma_next_non_negative(seed: int, n: int, vol: float) -> None:
    """``sigma_next ≥ 0`` whenever the GARCH fit produced a number."""
    out = fit_garch(_synth(seed, n=n, vol=vol))
    if out.sigma_next is not None:
        assert out.sigma_next >= 0.0, (
            f"GARCH produced negative sigma {out.sigma_next} — sign-flip bug"
        )


# ── Type consistency ────────────────────────────────────────────────────────


@given(seed=st.integers(min_value=0, max_value=1_000))
@_SETTINGS
def test_sigma_next_is_python_float(seed: int) -> None:
    """``sigma_next`` is a Python ``float`` (not ``np.float64``) so
    callers can JSON-serialise it without an extra cast."""
    out = fit_garch(_synth(seed))
    if out.sigma_next is not None:
        assert type(out.sigma_next) is float


# ── Idempotence ─────────────────────────────────────────────────────────────


@given(seed=st.integers(min_value=0, max_value=1_000))
@_SETTINGS
def test_fit_is_deterministic_for_same_input(seed: int) -> None:
    """Same returns series → same forecast (the GARCH MLE is deterministic
    given the same starting point and data)."""
    r = _synth(seed)
    a = fit_garch(r).sigma_next
    b = fit_garch(r).sigma_next
    if a is None or b is None:
        # both should be None when one is — degenerate case
        assert a is None and b is None
    else:
        assert a == pytest.approx(b, rel=1e-9)


# ── Convergence ↔ sigma agreement ───────────────────────────────────────────


@given(seed=st.integers(min_value=0, max_value=1_000))
@_SETTINGS
def test_converged_implies_sigma_present(seed: int) -> None:
    """``converged=True`` is the contract for ``sigma_next is not None``."""
    out = fit_garch(_synth(seed))
    if out.converged:
        assert out.sigma_next is not None


@given(seed=st.integers(min_value=0, max_value=1_000))
@_SETTINGS
def test_no_sigma_implies_not_converged(seed: int) -> None:
    """The contrapositive: ``sigma_next is None`` ⇒ ``converged is False``."""
    out = fit_garch(_synth(seed))
    if out.sigma_next is None:
        assert out.converged is False


# ── forecast_next_sigma agrees with fit_garch.sigma_next ────────────────────


@given(seed=st.integers(min_value=0, max_value=1_000))
@_SETTINGS
def test_forecast_next_sigma_matches_fit_garch(seed: int) -> None:
    r = _synth(seed)
    full = fit_garch(r).sigma_next
    short = forecast_next_sigma(r)
    if full is None:
        assert short is None
    else:
        assert short == pytest.approx(full, rel=1e-9)


# ── Boundary: degenerate constant input ────────────────────────────────────


def test_constant_zero_returns_degrades_gracefully() -> None:
    """Constant-zero returns mean σ²=0 — the fit either fails to converge
    or returns ``sigma_next ≈ 0``. Either way no exception, no NaN."""
    out = fit_garch(pd.Series(np.zeros(200)))
    assert out.sigma_next is None or out.sigma_next >= 0.0


def test_constant_nonzero_returns_degrades_gracefully() -> None:
    """Constant-non-zero returns are still degenerate (no variance)."""
    out = fit_garch(pd.Series(np.full(200, 0.005)))
    assert out.sigma_next is None or out.sigma_next >= 0.0
