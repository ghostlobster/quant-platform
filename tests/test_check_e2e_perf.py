"""Tests for ``scripts/check_e2e_perf.py`` — e2e per-test + total perf gate."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_e2e_perf import _scan_junit, main


def _write_junit(
    path: Path,
    cases: list[tuple[str, str, float]],
    *,
    suite_total: float | None = None,
    skip_indices: set[int] | None = None,
    error_indices: set[int] | None = None,
) -> Path:
    """Write a junit XML with ``(classname, name, time)`` cases.

    ``skip_indices`` / ``error_indices`` add ``<skipped/>`` or
    ``<error/>`` children to the indicated cases — those should be
    excluded from the perf accounting.
    """
    skips = skip_indices or set()
    errors = error_indices or set()
    blocks: list[str] = []
    for i, (cls, name, t) in enumerate(cases):
        body = ""
        if i in skips:
            body = '\n      <skipped message="x"/>'
        elif i in errors:
            body = '\n      <error message="x"/>'
        blocks.append(
            f'    <testcase classname="{cls}" name="{name}" time="{t}">'
            f'{body}\n    </testcase>'
        )
    body = "\n".join(blocks)
    total_attr = (
        f' time="{suite_total:.4f}"' if suite_total is not None else ""
    )
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites>\n'
        f'  <testsuite name="pytest" tests="{len(cases)}"{total_attr}>\n'
        f"{body}\n"
        f"  </testsuite>\n"
        f"</testsuites>\n"
    )
    return path


# ── _scan_junit ──────────────────────────────────────────────────────────────


def test_scan_junit_returns_rows_and_total(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c1", "test_a", 0.5), ("c2", "test_b", 1.5)],
        suite_total=2.0,
    )
    rows, total = _scan_junit(junit)
    assert len(rows) == 2
    assert total == pytest.approx(2.0)


def test_scan_junit_excludes_skipped(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c1", "test_a", 0.1), ("c2", "test_b", 5.0), ("c3", "test_c", 0.2)],
        suite_total=5.3,
        skip_indices={1},
    )
    rows, _ = _scan_junit(junit)
    names = [n for _, n, _ in rows]
    assert names == ["test_a", "test_c"]


def test_scan_junit_excludes_errored(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c1", "test_a", 0.1), ("c2", "test_b", 0.2)],
        error_indices={0},
    )
    rows, _ = _scan_junit(junit)
    assert [n for _, n, _ in rows] == ["test_b"]


def test_scan_junit_falls_back_to_summed_when_total_missing(
    tmp_path: Path,
) -> None:
    """When the testsuite element has no ``time`` attribute, fall back
    to summing per-test times."""
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c1", "test_a", 0.4), ("c2", "test_b", 0.6)],
        suite_total=None,
    )
    _, total = _scan_junit(junit)
    assert total == pytest.approx(1.0)


def test_scan_junit_handles_malformed_time(tmp_path: Path) -> None:
    junit_path = tmp_path / "bad.xml"
    junit_path.write_text(
        '<?xml version="1.0"?>\n'
        '<testsuites><testsuite name="pytest" tests="1" time="abc">\n'
        '  <testcase classname="c" name="t" time="not-a-float" />\n'
        '</testsuite></testsuites>\n'
    )
    rows, total = _scan_junit(junit_path)
    assert rows == [("c", "t", 0.0)]
    assert total == 0.0


# ── main ─────────────────────────────────────────────────────────────────────


def test_main_passes_when_all_under_budget(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c", "test_a", 0.5), ("c", "test_b", 1.0)],
        suite_total=1.5,
    )
    monkeypatch.delenv("E2E_MAX_TEST_SECONDS", raising=False)
    monkeypatch.delenv("E2E_MAX_TOTAL_SECONDS", raising=False)
    rc = main([str(junit)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "perf gate" in out
    assert "FAIL" not in out


def test_main_fails_when_a_test_exceeds_per_test_budget(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c", "fast", 0.5), ("c", "slow", 7.5)],
        suite_total=8.0,
    )
    monkeypatch.setenv("E2E_MAX_TEST_SECONDS", "3.0")
    monkeypatch.setenv("E2E_MAX_TOTAL_SECONDS", "30.0")
    rc = main([str(junit)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "slow" in out
    assert "exceed the 3.0s per-test budget" in out


def test_main_fails_when_total_exceeds_budget(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Total breach with every individual test under per-test budget."""
    cases = [("c", f"t{i}", 2.5) for i in range(20)]
    junit = _write_junit(tmp_path / "j.xml", cases, suite_total=50.0)
    monkeypatch.setenv("E2E_MAX_TEST_SECONDS", "3.0")
    monkeypatch.setenv("E2E_MAX_TOTAL_SECONDS", "30.0")
    rc = main([str(junit)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "exceeds the 30.0s budget" in out


def test_main_passes_when_no_testcases(tmp_path: Path, capsys) -> None:
    junit = _write_junit(tmp_path / "j.xml", [], suite_total=0.0)
    rc = main([str(junit)])
    assert rc == 0
    assert "nothing to check" in capsys.readouterr().out


def test_main_returns_2_on_no_args(capsys) -> None:
    rc = main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


def test_main_returns_2_on_missing_xml(tmp_path: Path, capsys) -> None:
    rc = main([str(tmp_path / "nope.xml")])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_returns_2_on_malformed_xml(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<not-closed")
    rc = main([str(bad)])
    assert rc == 2


def test_main_respects_env_overrides(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Tighten the per-test budget — a test that was OK at 3.0 s flips
    to FAIL at 1.0 s."""
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c", "t", 1.5)],
        suite_total=1.5,
    )
    monkeypatch.setenv("E2E_MAX_TEST_SECONDS", "1.0")
    rc = main([str(junit)])
    assert rc == 1
    assert "1.5" in capsys.readouterr().out


def test_main_invalid_env_falls_back_to_default(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An unparseable env var → silently use the default — same
    behaviour as ``check_e2e_coverage.py``."""
    junit = _write_junit(
        tmp_path / "j.xml",
        [("c", "t", 0.5)],
        suite_total=0.5,
    )
    monkeypatch.setenv("E2E_MAX_TEST_SECONDS", "not-a-number")
    monkeypatch.setenv("E2E_MAX_TOTAL_SECONDS", "garbage")
    rc = main([str(junit)])
    assert rc == 0  # default 3.0/30.0 → 0.5s test, 0.5s total → pass
