"""Tests for ``scripts/check_changed_module_coverage.py`` — the per-PR
"excellent test" gate. Closes the harness-rule extension on #215.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.check_changed_module_coverage import (
    _changed_python_files,
    _combined_coverage,
    main,
)


def _write_cov_xml(path: Path, modules: list[tuple[str, float, float]]) -> Path:
    """Write a minimal cobertura XML with ``(filename, line_rate, branch_rate)``."""
    classes = "\n".join(
        f'        <class filename="{name}" line-rate="{lr:.4f}" branch-rate="{br:.4f}">\n'
        f"          <lines><line number=\"1\" hits=\"1\"/></lines>\n"
        f"        </class>"
        for name, lr, br in modules
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<coverage>\n'
        '  <packages><package name="."><classes>\n'
        f"{classes}\n"
        '  </classes></package></packages>\n'
        '</coverage>\n'
    )
    return path


# ── _changed_python_files ────────────────────────────────────────────────────

def test_changed_files_filters_omitted_prefixes() -> None:
    out = (
        "risk/var.py\n"
        "tests/test_var.py\n"
        "pages/chart.py\n"
        "scripts/foo.py\n"
        "app.py\n"
        "analysis/regime.py\n"
        "providers/__init__.py\n"
    )
    with patch(
        "scripts.check_changed_module_coverage.subprocess.check_output",
        return_value=out,
    ):
        files = _changed_python_files("origin/main")
    assert files == ["analysis/regime.py", "risk/var.py"]


def test_changed_files_handles_git_failure() -> None:
    import subprocess

    with patch(
        "scripts.check_changed_module_coverage.subprocess.check_output",
        side_effect=subprocess.CalledProcessError(1, ["git"]),
    ):
        assert _changed_python_files("origin/main") == []


def test_changed_files_skips_non_python() -> None:
    out = "README.md\nrisk/var.py\nrequirements.txt\n"
    with patch(
        "scripts.check_changed_module_coverage.subprocess.check_output",
        return_value=out,
    ):
        files = _changed_python_files("origin/main")
    assert files == ["risk/var.py"]


# ── _combined_coverage ───────────────────────────────────────────────────────

def test_combined_coverage_pulls_filename(tmp_path: Path) -> None:
    xml = _write_cov_xml(tmp_path / "cov.xml", [("risk/var.py", 0.95, 0.95)])
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    pct = _combined_coverage("risk/var.py", root)
    assert pct == pytest.approx(95.0, abs=0.5)


def test_combined_coverage_returns_none_for_unknown(tmp_path: Path) -> None:
    xml = _write_cov_xml(tmp_path / "cov.xml", [("risk/var.py", 0.95, 0.95)])
    import xml.etree.ElementTree as ET
    root = ET.parse(xml).getroot()
    assert _combined_coverage("doesnt/exist.py", root) is None


# ── main ─────────────────────────────────────────────────────────────────────

def test_main_passes_when_no_files_changed(tmp_path: Path, capsys) -> None:
    xml = _write_cov_xml(tmp_path / "cov.xml", [])
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=[],
    ):
        rc = main([str(xml)])
    assert rc == 0
    assert "no source files changed" in capsys.readouterr().out


def test_main_passes_when_all_above_floor(tmp_path: Path, capsys) -> None:
    xml = _write_cov_xml(
        tmp_path / "cov.xml",
        [("risk/var.py", 0.96, 0.95), ("analysis/regime.py", 1.0, 1.0)],
    )
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=["risk/var.py", "analysis/regime.py"],
    ):
        rc = main([str(xml)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out and "FAIL" not in out


def test_main_fails_when_a_file_is_below_floor(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    xml = _write_cov_xml(
        tmp_path / "cov.xml",
        [("risk/var.py", 0.96, 0.95), ("strategies/badfile.py", 0.50, 0.40)],
    )
    monkeypatch.setenv("CHANGED_MODULE_MIN_PCT", "85")
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=["risk/var.py", "strategies/badfile.py"],
    ):
        rc = main([str(xml)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "strategies/badfile.py" in out
    assert "Changed-module coverage gate not met" in out


def test_main_flags_changed_file_missing_from_xml(
    tmp_path: Path, capsys
) -> None:
    xml = _write_cov_xml(tmp_path / "cov.xml", [])
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=["analysis/regime.py"],
    ):
        rc = main([str(xml)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "MISSING from coverage XML" in out


def test_main_returns_2_on_no_args(capsys) -> None:
    rc = main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


def test_main_returns_2_on_missing_xml(tmp_path: Path, capsys) -> None:
    rc = main([str(tmp_path / "nope.xml")])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_respects_custom_floor_env(tmp_path: Path, monkeypatch, capsys) -> None:
    """A 90 % floor flips a 88 % file from OK to FAIL."""
    xml = _write_cov_xml(
        tmp_path / "cov.xml", [("risk/var.py", 0.88, 0.88)]
    )
    monkeypatch.setenv("CHANGED_MODULE_MIN_PCT", "90")
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=["risk/var.py"],
    ):
        rc = main([str(xml)])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_accepts_custom_base_ref(tmp_path: Path) -> None:
    xml = _write_cov_xml(tmp_path / "cov.xml", [("risk/var.py", 0.96, 0.95)])
    with patch(
        "scripts.check_changed_module_coverage._changed_python_files",
        return_value=["risk/var.py"],
    ) as mock_diff:
        rc = main([str(xml), "main"])
    assert rc == 0
    mock_diff.assert_called_with("main")
