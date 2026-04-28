"""
scripts/check_changed_module_coverage.py — pre-PR "excellent test" gate.

Closes #215 / harness-rule extension.

Line + branch coverage on the *whole repo* tells you the project is
healthy on average. It does not tell you that the **module the current
PR touched** is well-tested. This script closes that gap: given a
coverage XML and a base ref, it asserts that every Python source file
the diff added or modified has at least
``CHANGED_MODULE_MIN_PCT`` (default 85 %) combined coverage.

Why a separate gate from the global 76 % floor:

  * The global floor passes if you add an uncovered 200-line module —
    the average barely moves. Per-module enforcement makes the
    contributor own the new lines.
  * Test-quality drift is local: a regression always shows up first
    in the file the PR touched. Catching it there is faster than
    waiting for the global average to slip.

Usage::

    python scripts/check_changed_module_coverage.py coverage.xml [BASE_REF]

``BASE_REF`` defaults to ``origin/main``. Diff is computed via
``git diff --name-only --diff-filter=AM <BASE>...HEAD``.

Files implicitly excluded (mirror ``.coveragerc:omit``):
  * ``tests/*``, ``pages/*``, ``app.py``, ``venv/*``, ``.venv/*``
  * generated ``__init__.py`` files (zero-statement)
  * ``scripts/*`` themselves (run-once helpers, not under coverage)

Override the floor with ``CHANGED_MODULE_MIN_PCT=90`` for stricter PRs.

Exit codes:
    0 — every changed source file meets the floor (or no source files changed)
    1 — at least one changed source file is below the floor
    2 — usage / parse error
"""
from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_OMIT_PREFIXES = (
    "tests/",
    "pages/",
    "venv/",
    ".venv/",
    "scripts/",
)
_OMIT_FILES = {"app.py"}
_DEFAULT_BASE = "origin/main"


def _changed_python_files(base: str) -> list[str]:
    """Return Python source files added or modified between BASE and HEAD."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"::error::git diff failed: {exc}", file=sys.stderr)
        return []
    files: list[str] = []
    for raw in out.splitlines():
        f = raw.strip()
        if not f.endswith(".py"):
            continue
        if f in _OMIT_FILES:
            continue
        if any(f.startswith(p) for p in _OMIT_PREFIXES):
            continue
        if f.endswith("__init__.py"):
            # zero-stmt files always 100 %; surface noise, skip.
            continue
        files.append(f)
    return sorted(files)


def _coverage_pct(rate: str | None) -> float:
    if rate is None:
        return 0.0
    try:
        return float(rate) * 100.0
    except ValueError:
        return 0.0


def _combined_coverage(filename: str, root: ET.Element) -> float | None:
    """Combine line + branch rate the way coverage.py reports the headline.

    coverage.py exposes per-class ``line-rate`` and ``branch-rate`` in the
    cobertura XML. With ``branch = True`` the headline percentage is the
    average of the two weighted by their counts; we approximate the same
    by averaging the rates with their relative size, falling back to
    line-rate when branch counts are zero.
    """
    for cls in root.iter("class"):
        if cls.attrib.get("filename", "").replace("\\", "/") != filename:
            continue
        line_rate = _coverage_pct(cls.attrib.get("line-rate"))
        branch_rate = cls.attrib.get("branch-rate")
        if branch_rate is None:
            return line_rate
        # Weight by the actual counts when present; otherwise even average.
        try:
            n_lines = sum(1 for _ in cls.iter("line"))
            n_branches = sum(
                int(line.attrib.get("condition-coverage", "0/0").split("/")[-1] or 0)
                for line in cls.iter("line")
                if line.attrib.get("branch") == "true"
            )
        except (ValueError, AttributeError):
            return (line_rate + _coverage_pct(branch_rate)) / 2.0
        total = n_lines + n_branches
        if total == 0:
            return line_rate
        return (
            line_rate * n_lines + _coverage_pct(branch_rate) * n_branches
        ) / total
    return None


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "usage: check_changed_module_coverage.py <coverage-xml> [BASE_REF]",
            file=sys.stderr,
        )
        return 2

    xml_path = Path(args[0])
    base = args[1] if len(args) > 1 else _DEFAULT_BASE
    floor = float(os.environ.get("CHANGED_MODULE_MIN_PCT", "85"))

    try:
        root = ET.parse(xml_path).getroot()
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"::error::cannot read {xml_path}: {exc}", file=sys.stderr)
        return 2

    changed = _changed_python_files(base)
    if not changed:
        print(
            f"changed-module coverage gate: no source files changed vs {base} — "
            "nothing to check"
        )
        return 0

    print(
        f"changed-module coverage gate (≥ {floor:.0f}% combined; vs {base}):"
    )
    failures: list[tuple[str, float]] = []
    misses: list[str] = []
    for path in changed:
        pct = _combined_coverage(path, root)
        if pct is None:
            misses.append(path)
            print(f"  {path:<48s} MISSING from coverage XML")
            continue
        marker = "OK" if pct >= floor else "FAIL"
        print(f"  {path:<48s} {pct:5.1f}%   {marker}")
        if pct < floor:
            failures.append((path, pct))

    if misses:
        print(
            "::error::Changed source file is missing from the coverage XML — "
            "are these files exercised by any test?\n  " + "\n  ".join(misses),
        )
    if failures:
        print(
            "::error::Changed-module coverage gate not met. Each PR must "
            f"bring its modules to ≥ {floor:.0f}%; if a deliberate "
            "exception is needed, document it in the PR body and bump "
            "the floor via CHANGED_MODULE_MIN_PCT.\n  "
            + "\n  ".join(f"{p}: {a:.1f}%" for p, a in failures),
        )
    return 1 if (failures or misses) else 0


if __name__ == "__main__":
    sys.exit(main())
