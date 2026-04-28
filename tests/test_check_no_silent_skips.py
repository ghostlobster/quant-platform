"""Tests for ``scripts/check_no_silent_skips.py`` — the silent-skip CI guard."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_no_silent_skips import (
    _DIST_TO_IMPORTS,
    _extract_import_name,
    _required_imports,
    main,
)


def _write_junit(
    path: Path,
    skips: list[tuple[str, str, str]],
    *,
    text_skips: list[tuple[str, str, str, str]] | None = None,
) -> Path:
    """Write a minimal JUnit XML with skipped testcases.

    ``skips`` carry the reason in the ``message`` attribute (per-test
    skipif). ``text_skips`` carry it in the element text with a generic
    ``message`` attribute (module-level importorskip). Both shapes are
    emitted by pytest in real life.
    """
    blocks: list[str] = []
    for cls, name, msg in skips:
        blocks.append(
            f'    <testcase classname="{cls}" name="{name}">\n'
            f'      <skipped message="{msg}"/>\n'
            f"    </testcase>"
        )
    for cls, name, attr_msg, text_msg in text_skips or []:
        blocks.append(
            f'    <testcase classname="{cls}" name="{name}">\n'
            f'      <skipped message="{attr_msg}">{text_msg}</skipped>\n'
            f"    </testcase>"
        )
    body = "\n".join(blocks)
    total = len(skips) + len(text_skips or [])
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n'
        f'<testsuites>\n'
        f'  <testsuite name="pytest" tests="{total}" skipped="{total}">\n'
        f"{body}\n"
        f"  </testsuite>\n"
        f"</testsuites>\n"
    )
    return path


def _write_requirements(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n")
    return path


# ── _extract_import_name ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message, expected",
    [
        ("could not import 'ta': No module named 'ta'", "ta"),
        ("could not import 'duckdb'", "duckdb"),
        ("gensim not installed", "gensim"),
        ("torch not installed", "torch"),
        ("No module named 'ta'", "ta"),
        ("network unreachable", None),
        ("", None),
        # double-quote variant of importorskip
        ('could not import "ta"', "ta"),
        # mixed case in custom skipif
        ("LightGBM not installed", "lightgbm"),
    ],
)
def test_extract_import_name(message: str, expected: str | None) -> None:
    assert _extract_import_name(message) == expected


# ── _required_imports ────────────────────────────────────────────────────────

def test_required_imports_strips_versions(tmp_path: Path) -> None:
    req = _write_requirements(
        tmp_path / "requirements.txt",
        [
            "ta==0.11.0",
            "ccxt>=4.0.0",
            "lightgbm>=4.0.0",
            "uvicorn[standard]==0.30.0",
            "# a comment",
            "",
        ],
    )
    out = _required_imports(req)
    assert {"ta", "ccxt", "lightgbm", "uvicorn"} <= out


def test_required_imports_expands_aliases(tmp_path: Path) -> None:
    req = _write_requirements(
        tmp_path / "requirements.txt",
        ["scikit-learn>=1.3.0", "python-dateutil==2.9.0"],
    )
    out = _required_imports(req)
    assert "scikit-learn" in out and "sklearn" in out
    assert "python-dateutil" in out and "dateutil" in out


def test_required_imports_missing_file_returns_empty(tmp_path: Path) -> None:
    assert _required_imports(tmp_path / "nope.txt") == set()


# ── main ─────────────────────────────────────────────────────────────────────

def test_main_clean_when_no_skips(tmp_path: Path, capsys) -> None:
    junit = _write_junit(tmp_path / "junit.xml", [])
    req = _write_requirements(tmp_path / "requirements.txt", ["ta==0.11.0"])
    rc = main([str(junit), str(req)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "scanned 0 skip(s)" in out


def test_main_clean_when_skip_is_optional_dep(tmp_path: Path, capsys) -> None:
    """``torch`` is NOT in requirements.txt so a torch skip is allowed."""
    junit = _write_junit(
        tmp_path / "junit.xml",
        [("tests.test_torch", "test_x", "torch not installed")],
    )
    req = _write_requirements(tmp_path / "requirements.txt", ["pandas==3.0.2"])
    rc = main([str(junit), str(req)])
    assert rc == 0
    assert "no skips reference packages pinned" in capsys.readouterr().out


def test_main_fails_when_required_pkg_skipped(tmp_path: Path, capsys) -> None:
    junit = _write_junit(
        tmp_path / "junit.xml",
        [
            (
                "tests.test_strategies_indicators",
                "test_add_rsi",
                "could not import 'ta': No module named 'ta'",
            )
        ],
    )
    req = _write_requirements(tmp_path / "requirements.txt", ["ta==0.11.0"])
    rc = main([str(junit), str(req)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "silently skipped" in out
    assert "ta" in out
    assert "test_add_rsi" in out


def test_main_fails_for_alias_skip(tmp_path: Path, capsys) -> None:
    """A skip naming ``sklearn`` should fail when ``scikit-learn`` is pinned."""
    junit = _write_junit(
        tmp_path / "junit.xml",
        [("tests.t", "test_x", "could not import 'sklearn'")],
    )
    req = _write_requirements(
        tmp_path / "requirements.txt", ["scikit-learn>=1.3.0"]
    )
    rc = main([str(junit), str(req)])
    assert rc == 1
    assert "sklearn" in capsys.readouterr().out


def test_main_returns_2_on_missing_xml(tmp_path: Path, capsys) -> None:
    rc = main([str(tmp_path / "does-not-exist.xml")])
    assert rc == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_returns_2_on_no_args(capsys) -> None:
    rc = main([])
    assert rc == 2
    assert "usage:" in capsys.readouterr().err


def test_main_returns_2_on_malformed_xml(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.xml"
    bad.write_text("<not-valid")
    rc = main([str(bad)])
    assert rc == 2


# ── alias map sanity ─────────────────────────────────────────────────────────

def test_alias_map_keys_are_lowercase() -> None:
    """Distribution-name keys are matched case-insensitively after lower()."""
    for k in _DIST_TO_IMPORTS:
        assert k == k.lower(), f"alias key not lowercase: {k}"


def test_main_fails_for_module_level_importorskip(
    tmp_path: Path, capsys
) -> None:
    """Module-level ``pytest.importorskip`` puts the reason in the
    element *text* with ``message="collection skipped"`` — verify we
    still detect it."""
    junit = _write_junit(
        tmp_path / "junit.xml",
        skips=[],
        text_skips=[
            (
                "",
                "tests.test_strategies_indicators",
                "collection skipped",
                "Skipped: could not import 'ta': No module named 'ta'",
            )
        ],
    )
    req = _write_requirements(tmp_path / "requirements.txt", ["ta==0.11.0"])
    rc = main([str(junit), str(req)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "ta" in out
    assert "test_strategies_indicators" in out


def test_main_handles_multiple_skips_some_offending(
    tmp_path: Path, capsys
) -> None:
    junit = _write_junit(
        tmp_path / "junit.xml",
        [
            ("tests.t1", "ok", "torch not installed"),  # optional, allowed
            ("tests.t2", "bad", "could not import 'ta'"),  # required, fail
            ("tests.t3", "bad2", "gensim not installed"),  # required, fail
        ],
    )
    req = _write_requirements(
        tmp_path / "requirements.txt",
        ["ta==0.11.0", "gensim>=4.3.0"],
    )
    rc = main([str(junit), str(req)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "2 test(s) silently skipped" in out
    assert "test_x" not in out  # sanity — different name
    assert "torch" not in out  # optional dep is not flagged
