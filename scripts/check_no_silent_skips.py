"""
scripts/check_no_silent_skips.py — fail if a test was skipped because a
package pinned in ``requirements.txt`` failed to import.

Closes #199.

The local 78% coverage hides ~150 statements that are silently skipped
when an optional dep can't import — e.g. ``ta`` wheel-build failures
mute every test in ``tests/test_strategies_indicators.py`` and
``strategies/indicators.py`` shows 0 % line coverage with no failing
test. CI installs the deps so the published number is higher, but a
future regression — a removed pin, a broken wheel, a Python-version
bump that drops a dep — would silently lower coverage with no failing
test.

This checker reads the JUnit XML produced by ``pytest --junitxml=...``,
extracts every ``<skipped/>`` reason, and fails if any reason names a
package that's pinned in ``requirements.txt``.

Usage::

    python scripts/check_no_silent_skips.py test-results-unit.xml

Exit codes:
    0 — no silent skips against required deps
    1 — at least one skip references a required package
    2 — usage / parse error
"""
from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Map distribution name (as it appears in ``requirements.txt``) to the
# *import* name(s) it provides. We only need entries for packages whose
# distribution name differs from their import name. Lower-case throughout.
_DIST_TO_IMPORTS: dict[str, set[str]] = {
    "scikit-learn":    {"sklearn"},
    "python-dateutil": {"dateutil"},
    "python-dotenv":   {"dotenv"},
    "gitpython":       {"git"},
    "pillow":          {"pil"},
    "pyyaml":          {"yaml"},
    "beautifulsoup4":  {"bs4"},
    "ib-insync":       {"ib_insync"},
    "curl-cffi":       {"curl_cffi"},
    "streamlit-autorefresh": {"streamlit_autorefresh"},
}

# Patterns that extract an import name from a pytest skip message.
# - importorskip default:   "could not import 'X': ..."
# - generic skipif message: "X not installed"
# - bare ImportError text:  "No module named 'X'"
_PATTERNS = [
    re.compile(r"could not import ['\"]([\w.\-]+)['\"]"),
    re.compile(r"^([\w.\-]+) not installed", re.IGNORECASE),
    re.compile(r"No module named ['\"]([\w.\-]+)['\"]"),
]

# Strip extras and version specifiers to recover the bare distribution name.
# Examples:  "ccxt>=4.0.0" -> "ccxt"
#            "uvicorn[standard]==0.30.0" -> "uvicorn"
_REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.\-]+)")


def _required_imports(requirements_path: Path) -> set[str]:
    """Return the set of import names that are pinned in requirements.txt.

    Each distribution contributes its lower-cased name *plus* any aliases
    declared in :data:`_DIST_TO_IMPORTS`.
    """
    imports: set[str] = set()
    if not requirements_path.exists():
        return imports
    for raw in requirements_path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = _REQ_LINE_RE.match(line)
        if not match:
            continue
        dist = match.group(1).lower()
        imports.add(dist)
        imports.update(_DIST_TO_IMPORTS.get(dist, set()))
    return imports


def _extract_import_name(message: str) -> str | None:
    """Return the import name a skip message blames, or ``None``."""
    for pat in _PATTERNS:
        m = pat.search(message)
        if m:
            return m.group(1).lower()
    return None


def _scan_junit(xml_path: Path) -> list[tuple[str, str, str]]:
    """Return list of (classname, testname, message) for every skipped test.

    Module-level ``pytest.importorskip`` reports the reason in the
    element *text*, not the ``message`` attribute (which is just
    ``"collection skipped"``); we concatenate both so a single regex
    pass catches every variant.
    """
    tree = ET.parse(xml_path)
    skipped: list[tuple[str, str, str]] = []
    for case in tree.iter("testcase"):
        for skip in case.findall("skipped"):
            attr_msg = skip.attrib.get("message", "") or ""
            text_msg = skip.text or ""
            combined = (attr_msg + " | " + text_msg).strip(" |")
            skipped.append(
                (
                    case.attrib.get("classname", ""),
                    case.attrib.get("name", ""),
                    combined,
                )
            )
    return skipped


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(
            "usage: check_no_silent_skips.py <junit-xml> [requirements.txt]",
            file=sys.stderr,
        )
        return 2

    xml_path = Path(args[0])
    req_path = Path(args[1]) if len(args) > 1 else Path("requirements.txt")

    try:
        skipped = _scan_junit(xml_path)
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"::error::cannot read {xml_path}: {exc}", file=sys.stderr)
        return 2

    required = _required_imports(req_path)
    offenders: list[tuple[str, str, str, str]] = []
    for classname, testname, message in skipped:
        pkg = _extract_import_name(message)
        if pkg and pkg in required:
            offenders.append((classname, testname, pkg, message))

    print(f"silent-skip check: scanned {len(skipped)} skip(s) in {xml_path}")
    if not offenders:
        print("  no skips reference packages pinned in requirements.txt")
        return 0

    print(
        f"::error::{len(offenders)} test(s) silently skipped because a "
        "required package failed to import:",
    )
    for classname, testname, pkg, message in offenders:
        print(f"  {classname}::{testname}")
        print(f"    package: {pkg}  (pinned in {req_path})")
        print(f"    reason : {message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
