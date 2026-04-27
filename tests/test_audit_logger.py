"""
tests/test_audit_logger.py — append-only JSONL audit log + rotation.
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from audit import logger as audit


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "log"))
    return tmp_path / "log"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ── Validation ──────────────────────────────────────────────────────────────

def test_log_rejects_bad_kind(audit_dir):
    with pytest.raises(ValueError, match="kind"):
        audit._log("garbage", "rid", "AAPL", {})


def test_log_rejects_empty_run_id(audit_dir):
    with pytest.raises(ValueError, match="run_id"):
        audit.log_decision("", "AAPL")


def test_new_run_id_is_unique():
    a = audit.new_run_id()
    b = audit.new_run_id()
    assert a != b
    assert len(a) == 8


# ── Append + read back ─────────────────────────────────────────────────────

def test_log_decision_round_trip(audit_dir):
    audit.log_decision("rid-1", "AAPL", {"score": 0.4})
    audit.log_order("rid-1", "AAPL", {"qty": 10, "side": "buy"})
    audit.log_fill("rid-1", "AAPL", {"price": 100.0})
    audit.log_pnl("rid-1", "AAPL", {"pnl": 50.0})

    today = _today()
    path = audit_dir / f"{today}.jsonl"
    assert path.exists()
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 4
    parsed = [json.loads(line) for line in lines]
    assert [r["kind"] for r in parsed] == ["decision", "order", "fill", "pnl"]
    assert all(r["run_id"] == "rid-1" for r in parsed)
    assert all(r["ticker"] == "AAPL" for r in parsed)


def test_log_normalises_ticker_case(audit_dir):
    audit.log_order("rid-1", "  aapl ", {"qty": 1})
    rec = next(audit.iter_records())
    assert rec["ticker"] == "AAPL"


def test_iter_records_filters_by_date(audit_dir):
    today = _today()
    audit.log_decision("rid-1", "AAPL", {"i": 1})

    # Manually create a record from yesterday and tomorrow.
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    tomorrow  = (datetime.now(timezone.utc) + timedelta(days=1)).strftime("%Y-%m-%d")
    for date_str, payload in [
        (yesterday, {"i": "yesterday"}),
        (tomorrow,  {"i": "tomorrow"}),
    ]:
        path = audit_dir / f"{date_str}.jsonl"
        path.write_text(
            json.dumps(
                {"ts": "x", "run_id": "rid-2", "kind": "decision",
                 "ticker": "T", "details": payload}
            ) + "\n",
            encoding="utf-8",
        )

    only_today = list(audit.iter_records(start_date=today, end_date=today))
    assert len(only_today) == 1
    assert only_today[0]["details"]["i"] == 1

    everything = list(audit.iter_records())
    assert len(everything) == 3


def test_iter_records_handles_missing_dir(audit_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "nope"))
    assert list(audit.iter_records()) == []


def test_iter_records_skips_malformed_json(audit_dir):
    audit.log_order("rid-1", "AAPL", {"qty": 1})
    path = audit_dir / f"{_today()}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("not-json\n")
        f.write("\n")
    rows = list(audit.iter_records())
    assert len(rows) == 1


# ── Rotation ────────────────────────────────────────────────────────────────

def test_rotate_compresses_old_jsonl(audit_dir):
    today = datetime.now(timezone.utc).date()
    old_date = (today - timedelta(days=10)).isoformat()
    old_path = audit_dir / f"{old_date}.jsonl"
    old_path.parent.mkdir(parents=True, exist_ok=True)
    old_path.write_text("{\"x\":1}\n", encoding="utf-8")

    summary = audit.rotate(max_age_days=90, compress_after_days=7)
    assert summary["compressed"] == 1
    assert not old_path.exists()
    gz = audit_dir / f"{old_date}.jsonl.gz"
    assert gz.exists()
    assert gzip.decompress(gz.read_bytes()).decode() == "{\"x\":1}\n"


def test_rotate_deletes_past_max_age(audit_dir):
    today = datetime.now(timezone.utc).date()
    very_old = (today - timedelta(days=120)).isoformat()
    p = audit_dir / f"{very_old}.jsonl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(gzip.compress(b"{}\n"))

    summary = audit.rotate(max_age_days=90, compress_after_days=7)
    assert summary["deleted"] == 1
    assert not p.exists()


def test_rotate_handles_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "nope"))
    assert audit.rotate() == {"compressed": 0, "deleted": 0}


def test_rotate_rejects_bad_args(audit_dir):
    with pytest.raises(ValueError, match="max_age_days"):
        audit.rotate(max_age_days=3, compress_after_days=7)


def test_rotate_skips_unrelated_filenames(audit_dir):
    audit_dir.mkdir(parents=True, exist_ok=True)
    junk = audit_dir / "not-a-date.jsonl"
    junk.write_text("noise\n")
    # Should not delete or compress a non-dated file.
    audit.rotate()
    assert junk.exists()
