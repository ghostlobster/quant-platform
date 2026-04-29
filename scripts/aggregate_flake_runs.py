"""
scripts/aggregate_flake_runs.py — flake-rate aggregator for the
weekly flake-detection workflow (#228).

Reads N junit XMLs produced by N independent pytest runs of the same
suite and reports each test's pass/fail breakdown. Tests that go red
sometimes — but not always — are flaky; tests that always pass and
always fail aren't (those are stable, just possibly broken).

Usage::

    python scripts/aggregate_flake_runs.py run-1.xml run-2.xml ...

Output:

  * Stdout — human-readable summary (count + flake rate per test).
  * ``flake-report.md`` — markdown table sorted by failure count
    descending. Suitable for a GitHub issue body or a CI artefact.
  * Exit code 0 always — flakes are information, not a failure (the
    workflow already runs as ``continue-on-error: true``).

Why a separate script (instead of pytest-rerunfailures or
pytest-repeat): both plugins re-run failing tests within a single
pytest process, which obscures whether the failure is reproducible
across fresh interpreter invocations. We want **independent** runs
to detect flake from RNG drift, BLAS thread races, file-system
ordering, etc. The bash loop in the workflow gives us that for free;
this script just aggregates the resulting XMLs.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def _scan_junit(xml_path: Path) -> dict[str, str]:
    """Return ``{test_id: status}`` for every testcase in ``xml_path``.

    ``status`` is one of: ``"pass"``, ``"fail"``, ``"error"``,
    ``"skip"``. ``test_id`` is ``"classname::name"`` so it matches the
    pytest -v output format that operators see in workflow logs.
    """
    tree = ET.parse(xml_path)
    out: dict[str, str] = {}
    for case in tree.iter("testcase"):
        cls = case.attrib.get("classname", "")
        name = case.attrib.get("name", "")
        test_id = f"{cls}::{name}" if cls else name
        if case.find("failure") is not None:
            out[test_id] = "fail"
        elif case.find("error") is not None:
            out[test_id] = "error"
        elif case.find("skipped") is not None:
            out[test_id] = "skip"
        else:
            out[test_id] = "pass"
    return out


def _aggregate(xml_paths: list[Path]) -> dict[str, list[str]]:
    """Merge per-run statuses into ``{test_id: [status_run_1, ...]}``.

    Tests that appear in only some runs (collection-only changes) get
    a ``"missing"`` entry for the runs they're absent from so the
    flake-rate denominator stays consistent.
    """
    per_run: list[dict[str, str]] = [_scan_junit(p) for p in xml_paths]
    all_ids: set[str] = set()
    for r in per_run:
        all_ids.update(r.keys())
    merged: dict[str, list[str]] = defaultdict(list)
    for r in per_run:
        for tid in all_ids:
            merged[tid].append(r.get(tid, "missing"))
    return dict(merged)


def _flake_rate(statuses: list[str]) -> tuple[int, int, float]:
    """Return ``(fails, runs_excluding_skips_and_missing, flake_rate)``.

    A test that's ``"skip"`` or ``"missing"`` in some runs has those
    runs excluded from the denominator (it's not flake — the test
    just didn't run).
    """
    runs = [s for s in statuses if s not in ("skip", "missing")]
    fails = sum(1 for s in runs if s in ("fail", "error"))
    total = len(runs)
    rate = (fails / total) if total else 0.0
    return fails, total, rate


def _format_report(merged: dict[str, list[str]], n_runs: int) -> str:
    """Build a markdown report sorted by fail count descending."""
    rows: list[tuple[str, int, int, float, list[str]]] = []
    for tid, statuses in merged.items():
        fails, total, rate = _flake_rate(statuses)
        if 0 < fails < total:
            rows.append((tid, fails, total, rate, statuses))
    rows.sort(key=lambda r: (-r[1], r[0]))

    lines = [f"# Flake report ({n_runs} runs)\n"]
    if not rows:
        lines.append("✅ No flakes detected — every test that ran "
                     "passed in every run.")
        return "\n".join(lines)

    lines.append(
        f"⚠️  {len(rows)} flaky test(s) detected. Flake rate is "
        "fails / (runs − skips − missing); 1.0 means always-fail "
        "(broken, not flaky).\n"
    )
    lines.append("| Test | Fails | Runs | Rate | Per-run |")
    lines.append("|---|---:|---:|---:|---|")
    for tid, fails, total, rate, statuses in rows:
        per_run = " ".join("✅" if s == "pass" else "❌" if s in
                           ("fail", "error") else "⚪" for s in statuses)
        lines.append(
            f"| `{tid}` | {fails} | {total} | {rate:.0%} | {per_run} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "usage: aggregate_flake_runs.py <junit-xml> [...]",
            file=sys.stderr,
        )
        return 2

    xml_paths = [Path(a) for a in args]
    missing = [p for p in xml_paths if not p.exists()]
    if missing:
        print(
            f"::warning::skipping {len(missing)} missing XML(s): "
            + ", ".join(str(p) for p in missing),
            file=sys.stderr,
        )
        xml_paths = [p for p in xml_paths if p.exists()]

    if not xml_paths:
        print("::error::no junit XMLs to aggregate", file=sys.stderr)
        return 2

    merged = _aggregate(xml_paths)
    report = _format_report(merged, n_runs=len(xml_paths))
    print(report)

    out_path = Path("flake-report.md")
    out_path.write_text(report)
    print(f"\nReport written to {out_path.resolve()}", file=sys.stderr)

    # Exit 0 always — flake reporting is advisory, not a failure
    # signal. The workflow file decides whether to escalate.
    return 0


if __name__ == "__main__":
    sys.exit(main())
