"""
scripts/check_e2e_coverage.py — enforce per-module coverage floor on the
e2e job.

Reads the cobertura XML emitted by ``pytest --cov-report=xml:...`` and
checks that every cross-module file the e2e suite is meant to exercise
shows at least ``E2E_MIN_PER_MODULE`` percent line coverage (default
40 %).

The list is intentionally hand-curated to the modules each e2e file
claims to exercise — see the test docstrings. Adding a new e2e file
should mean adding the modules it touches here too; CI surfaces the
failure if a future refactor silently breaks the chain.

Usage:
    python scripts/check_e2e_coverage.py coverage-e2e.xml
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

# Modules each e2e file is supposed to exercise. Keys are file paths;
# values are the lower-bound % we require. The default floor (40 %)
# applies to every entry that doesn't override.
REQUIRED_MODULES: dict[str, float] = {
    # P1.1 / #183 — pre-trade guard chain
    "risk/pretrade_guard.py":            55.0,
    "adapters/broker/paper_adapter.py":  60.0,
    # P1.2 — risk dashboard chain
    "risk/metrics_exporter.py":          60.0,
    # P1.3 — bracket lifecycle (paper trader internals)
    "broker/paper_trader.py":            40.0,
    # P1.4 + P1.5 — polygon backfill + DuckDB cache chain
    "cron/polygon_backfill.py":          70.0,
    "adapters/market_data/polygon_adapter.py": 35.0,
    "data/duckdb_cache.py":              40.0,
    # P1.6 — MLflow retrain chain
    "cron/monthly_ml_retrain.py":        70.0,
    # P1.9 — event bus
    "bus/event_bus.py":                  30.0,
    "bus/events.py":                     75.0,
}


def _coverage_pct(rate: str | None) -> float:
    if rate is None:
        return 0.0
    try:
        return float(rate) * 100.0
    except ValueError:
        return 0.0


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: check_e2e_coverage.py <coverage-xml>", file=sys.stderr)
        return 2

    xml_path = args[0]
    try:
        root = ET.parse(xml_path).getroot()
    except (FileNotFoundError, ET.ParseError) as exc:
        print(f"::error::cannot read {xml_path}: {exc}")
        return 2

    default_min = float(os.environ.get("E2E_MIN_PER_MODULE", "40"))
    seen: dict[str, float] = {}
    for cls in root.iter("class"):
        filename = cls.attrib.get("filename", "").replace("\\", "/")
        if filename in REQUIRED_MODULES:
            seen[filename] = _coverage_pct(cls.attrib.get("line-rate"))

    failures: list[tuple[str, float, float]] = []
    misses: list[str] = []
    print(f"e2e per-module coverage floor (default ≥ {default_min:.0f}%):")
    for path, threshold in sorted(REQUIRED_MODULES.items()):
        floor = max(threshold, default_min) if default_min > threshold else threshold
        actual = seen.get(path)
        if actual is None:
            misses.append(path)
            print(f"  {path:<46s} MISSING from coverage XML")
            continue
        marker = "OK" if actual >= floor else "FAIL"
        print(f"  {path:<46s} {actual:5.1f}%  (≥ {floor:5.1f}%)  {marker}")
        if actual < floor:
            failures.append((path, actual, floor))

    if misses:
        print(
            "::error::e2e coverage XML is missing required modules: "
            + ", ".join(misses),
        )
    if failures:
        print(
            "::error::e2e per-module coverage floor not met:\n  "
            + "\n  ".join(
                f"{p}: {a:.1f}% < {f:.1f}%" for p, a, f in failures
            ),
        )
    return 1 if (failures or misses) else 0


if __name__ == "__main__":
    sys.exit(main())
