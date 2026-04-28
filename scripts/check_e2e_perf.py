"""
scripts/check_e2e_perf.py — per-test + total e2e wall-clock gate.

Closes part of #221 (e2e harness upgrade).

Reads the JUnit XML produced by ``pytest --junitxml=...`` and fails the
job when:

  * any single test exceeds ``E2E_MAX_TEST_SECONDS`` (default 3.0 s)
  * the suite total exceeds ``E2E_MAX_TOTAL_SECONDS`` (default 30 s)

Why a separate gate from the global ``timeout = 60`` in ``pytest.ini``:
the 60-s timeout catches a hung test before it eats the 15-min CI
budget. This gate catches the *milder* regression — a chain that
silently grew from 200 ms to 5 s. Per-test enforcement makes the
contributor own the perf cost they ship.

Usage::

    python scripts/check_e2e_perf.py test-results-e2e.xml

Exit codes:
    0 — every test under budget AND total under budget
    1 — at least one test or the total breaches a budget
    2 — usage / parse error
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_DEFAULT_PER_TEST_SECONDS = 3.0
_DEFAULT_TOTAL_SECONDS = 30.0


def _max_seconds(env_name: str, default: float) -> float:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _scan_junit(xml_path: Path) -> tuple[list[tuple[str, str, float]], float]:
    """Return ``(rows, total)`` where rows is ``(classname, name, time)``."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows: list[tuple[str, str, float]] = []
    for case in root.iter("testcase"):
        try:
            t = float(case.attrib.get("time", "0") or "0")
        except ValueError:
            t = 0.0
        # Skip cases that didn't actually run (skip / error reported).
        if case.find("skipped") is not None or case.find("error") is not None:
            continue
        rows.append(
            (case.attrib.get("classname", ""), case.attrib.get("name", ""), t)
        )
    # Suite total — pytest reports the wall clock on the testsuite element.
    total = 0.0
    for ts in root.iter("testsuite"):
        try:
            total = max(total, float(ts.attrib.get("time", "0") or "0"))
        except ValueError:
            pass
    if total == 0.0:
        # Fallback: sum the per-test times if the suite element didn't
        # carry the wall-clock attribute.
        total = sum(t for _, _, t in rows)
    return rows, total


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "usage: check_e2e_perf.py <junit-xml>",
            file=sys.stderr,
        )
        return 2

    xml_path = Path(args[0])
    per_test = _max_seconds("E2E_MAX_TEST_SECONDS", _DEFAULT_PER_TEST_SECONDS)
    total_max = _max_seconds("E2E_MAX_TOTAL_SECONDS", _DEFAULT_TOTAL_SECONDS)

    try:
        rows, total = _scan_junit(xml_path)
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"::error::cannot read {xml_path}: {exc}", file=sys.stderr)
        return 2

    if not rows:
        print(
            f"e2e perf gate: no testcases in {xml_path} — nothing to check"
        )
        return 0

    print(
        f"e2e perf gate (≤ {per_test:.1f}s/test, ≤ {total_max:.1f}s total):"
    )
    slowest = sorted(rows, key=lambda r: r[2], reverse=True)
    over_per_test = [(c, n, t) for c, n, t in rows if t > per_test]
    print(f"  ran {len(rows)} test(s); total wall-clock {total:.2f}s")
    print("  slowest 5:")
    for cls, name, t in slowest[:5]:
        marker = "FAIL" if t > per_test else "OK"
        print(f"    {t:5.2f}s  {marker}  {cls}::{name}")

    failed = False
    if over_per_test:
        failed = True
        print(
            f"::error::{len(over_per_test)} e2e test(s) exceed the "
            f"{per_test:.1f}s per-test budget. Slow chains compound — "
            "tighten or split them.",
        )
        for cls, name, t in over_per_test:
            print(f"  {cls}::{name}  {t:.2f}s")
    if total > total_max:
        failed = True
        print(
            f"::error::e2e suite total {total:.2f}s exceeds the "
            f"{total_max:.1f}s budget. The slowest tests above are "
            "the place to start.",
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
