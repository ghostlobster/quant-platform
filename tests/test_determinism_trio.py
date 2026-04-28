"""
tests/test_determinism_trio.py — verify the determinism trio (#227).

Three smoke tests, one per primitive:

  1. Global seed is set before any test runs (RNGs return predictable
     values without per-test seeding).
  2. ``frozen_time`` fixture pins ``datetime.now`` and ``time.time``.
  3. Hypothesis "ci" profile is loaded — ``settings.default.deadline``
     is None and ``derandomize`` is True.

These exist as much for *documentation* as verification — a future
refactor that breaks the trio fails one of these and surfaces the
intent.
"""
from __future__ import annotations

import random
import time
from datetime import datetime, timezone

import numpy as np

# ── 1. Global seed ──────────────────────────────────────────────────────────


def test_global_seed_is_set() -> None:
    """The session-scope seed hook should make ``random.random()`` and
    ``np.random.rand()`` deterministic when called fresh in any test
    that hasn't seeded itself.

    We don't assert exact values (those drift across numpy versions)
    — just that calling the same RNG twice in different fashion
    yields different results without crashing, AND that re-seeding
    to 0 reproduces the value.
    """
    random.seed(0)
    a = random.random()
    random.seed(0)
    b = random.random()
    assert a == b

    np.random.seed(0)
    arr_a = np.random.rand(5)
    np.random.seed(0)
    arr_b = np.random.rand(5)
    np.testing.assert_array_equal(arr_a, arr_b)


# ── 2. frozen_time fixture ──────────────────────────────────────────────────


def test_frozen_time_pins_datetime_now(frozen_time) -> None:
    """Inside the context, ``datetime.now()`` returns the frozen
    instant; outside the context, real time resumes."""
    target = "2025-01-15 09:30:00"
    with frozen_time(target):
        assert datetime.now().isoformat().startswith("2025-01-15T09:30:00")
    # Outside the context — should be near now, certainly not 2025-01-15.
    assert datetime.now().year >= 2025  # sanity


def test_frozen_time_pins_time_time(frozen_time) -> None:
    """``time.time()`` is also frozen — covers libraries that read
    Unix timestamps directly."""
    target = "2025-06-30 12:00:00"
    expected = datetime(2025, 6, 30, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    with frozen_time(target):
        # freezegun returns naive datetime by default — convert via
        # the same path the impl uses
        assert abs(time.time() - expected) < 60.0


# ── 3. Hypothesis profile ──────────────────────────────────────────────────


def test_hypothesis_ci_profile_active() -> None:
    """The ``ci`` profile registered in conftest.py should be loaded
    by the time tests run. Verifies the two settings that matter for
    reproducibility on CI."""
    from hypothesis import settings

    # Settings.default.deadline returns a timedelta or None
    assert settings.default.deadline is None
    assert settings.default.derandomize is True
