"""Tests for ``scripts/aggregate_flake_runs.py`` — flake-rate aggregator."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.aggregate_flake_runs import (
    _aggregate,
    _flake_rate,
    _format_report,
    _scan_junit,
    main,
)


def _write_junit(
    path: Path,
    cases: list[tuple[str, str, str]],  # (classname, name, status)
) -> Path:
    blocks: list[str] = []
    for cls, name, status in cases:
        body = ""
        if status == "fail":
            body = '\n      <failure message="x"/>'
        elif status == "error":
            body = '\n      <error message="x"/>'
        elif status == "skip":
            body = '\n      <skipped message="x"/>'
        blocks.append(
            f'    <testcase classname="{cls}" name="{name}">'
            f'{body}\n    </testcase>'
        )
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<testsuites><testsuite name="pytest" tests="'
        f'{len(cases)}">\n'
        + "\n".join(blocks)
        + "\n</testsuite></testsuites>\n"
    )
    return path


# ── _scan_junit ─────────────────────────────────────────────────────────────


def test_scan_junit_classifies_pass_fail_error_skip(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [
            ("c", "ok", "pass"),
            ("c", "broken", "fail"),
            ("c", "exploded", "error"),
            ("c", "skipped_test", "skip"),
        ],
    )
    statuses = _scan_junit(junit)
    assert statuses["c::ok"] == "pass"
    assert statuses["c::broken"] == "fail"
    assert statuses["c::exploded"] == "error"
    assert statuses["c::skipped_test"] == "skip"


def test_scan_junit_uses_name_when_classname_empty(tmp_path: Path) -> None:
    junit = _write_junit(
        tmp_path / "j.xml",
        [("", "module-level", "pass")],
    )
    statuses = _scan_junit(junit)
    assert "module-level" in statuses


# ── _aggregate ──────────────────────────────────────────────────────────────


def test_aggregate_carries_status_per_run(tmp_path: Path) -> None:
    """Same test, different statuses across runs → list reflects each."""
    a = _write_junit(tmp_path / "a.xml", [("c", "t", "pass")])
    b = _write_junit(tmp_path / "b.xml", [("c", "t", "fail")])
    c = _write_junit(tmp_path / "c.xml", [("c", "t", "pass")])
    merged = _aggregate([a, b, c])
    assert merged["c::t"] == ["pass", "fail", "pass"]


def test_aggregate_marks_missing_test_as_missing(tmp_path: Path) -> None:
    """A test that only ran in some XMLs gets ``missing`` in the others."""
    a = _write_junit(tmp_path / "a.xml", [("c", "t", "pass")])
    b = _write_junit(tmp_path / "b.xml", [])
    merged = _aggregate([a, b])
    assert merged["c::t"] == ["pass", "missing"]


# ── _flake_rate ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "statuses, expected",
    [
        # always pass → rate 0
        (["pass"] * 5,                            (0, 5, 0.0)),
        # 1/5 fails → rate 0.2
        (["pass", "pass", "fail", "pass", "pass"], (1, 5, 0.2)),
        # 5/5 fails → rate 1.0 (broken, not flaky)
        (["fail"] * 5,                            (5, 5, 1.0)),
        # error counts as fail
        (["pass", "error", "pass"],               (1, 3, 1 / 3)),
        # skips and missing are excluded from denominator
        (["pass", "skip", "fail", "missing"],     (1, 2, 0.5)),
        # all-skip → 0/0 → 0.0 (no division by zero)
        (["skip", "skip"],                        (0, 0, 0.0)),
    ],
)
def test_flake_rate(
    statuses: list[str], expected: tuple[int, int, float]
) -> None:
    fails, total, rate = _flake_rate(statuses)
    assert (fails, total) == expected[:2]
    assert rate == pytest.approx(expected[2], abs=1e-9)


# ── _format_report ──────────────────────────────────────────────────────────


def test_report_clean_when_no_flakes() -> None:
    merged = {"c::ok": ["pass", "pass", "pass"]}
    report = _format_report(merged, n_runs=3)
    assert "No flakes detected" in report
    assert "✅" in report


def test_report_includes_only_partially_failing(tmp_path: Path) -> None:
    """Always-pass + always-fail tests are excluded — only the ones
    with mixed pass/fail show in the table."""
    merged = {
        "c::stable_pass":   ["pass", "pass", "pass"],
        "c::stable_broken": ["fail", "fail", "fail"],
        "c::flaky":         ["pass", "fail", "pass"],
    }
    report = _format_report(merged, n_runs=3)
    assert "c::flaky" in report
    assert "c::stable_pass" not in report
    assert "c::stable_broken" not in report


def test_report_sorts_by_fail_count_descending() -> None:
    merged = {
        "c::low_flake":  ["pass"] * 4 + ["fail"],
        "c::high_flake": ["fail"] * 3 + ["pass"] * 2,
    }
    report = _format_report(merged, n_runs=5)
    assert report.index("high_flake") < report.index("low_flake")


# ── main ────────────────────────────────────────────────────────────────────


def test_main_writes_report_md(tmp_path: Path, monkeypatch, capsys) -> None:
    a = _write_junit(tmp_path / "a.xml", [("c", "t", "pass")])
    b = _write_junit(tmp_path / "b.xml", [("c", "t", "fail")])
    monkeypatch.chdir(tmp_path)
    rc = main([str(a), str(b)])
    assert rc == 0
    out = (tmp_path / "flake-report.md").read_text()
    assert "c::t" in out
    assert "Flake report" in capsys.readouterr().out


def test_main_returns_2_on_no_args(capsys) -> None:
    rc = main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


def test_main_warns_on_missing_xml_but_proceeds(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Missing XMLs are warnings, not failures — a partial weekly run
    (one rerun crashed) should still produce a report from the
    survivors."""
    a = _write_junit(tmp_path / "a.xml", [("c", "t", "pass")])
    monkeypatch.chdir(tmp_path)
    rc = main([str(a), str(tmp_path / "missing.xml")])
    assert rc == 0
    err = capsys.readouterr().err
    assert "skipping 1 missing" in err


def test_main_returns_2_when_all_xmls_missing(
    tmp_path: Path, capsys
) -> None:
    rc = main([str(tmp_path / "nope.xml")])
    assert rc == 2
    assert "no junit XMLs" in capsys.readouterr().err
