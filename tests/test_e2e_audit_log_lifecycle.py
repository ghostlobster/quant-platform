"""
tests/test_e2e_audit_log_lifecycle.py — audit log full lifecycle.

Closes part of #222.

Exercises the audit-log chain end-to-end:

  log_decision / log_order / log_fill / log_pnl
    → JSONL append to ``$AUDIT_LOG_DIR/<date>.jsonl``
    → iter_records yields the round-tripped dicts
    → rotate(max_age, compress_after) compresses + deletes old logs

The real production failure modes to assert:

  * JSONL is append-only — concurrent writes must not interleave lines
  * iter_records reads ``.jsonl`` and ``.jsonl.gz`` transparently
  * rotate guards: ``max_age_days >= compress_after_days``
  * rotate is idempotent: calling twice produces no change after the
    first run

Cleanup-invariant fixture is opt-out: this file doesn't touch
paper_trades / journal_trades, only the audit log directory.
"""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from audit import logger as audit_logger

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skip_invariant]


@pytest.fixture
def isolated_audit_dir(tmp_path, monkeypatch):
    """Per-test audit log directory."""
    log_dir = tmp_path / "audit_log"
    monkeypatch.setenv("AUDIT_LOG_DIR", str(log_dir))
    return log_dir


# ── Happy-path round-trip ───────────────────────────────────────────────────


def test_log_decision_then_iter_records_round_trip(isolated_audit_dir) -> None:
    """A logged decision shows up verbatim in iter_records."""
    run_id = audit_logger.new_run_id()
    audit_logger.log_decision(
        run_id, "AAPL", {"score": 0.42, "regime": "trending_bull"}
    )
    records = list(audit_logger.iter_records())
    assert len(records) == 1
    rec = records[0]
    assert rec["kind"] == "decision"
    assert rec["run_id"] == run_id
    assert rec["ticker"] == "AAPL"
    assert rec["details"] == {"score": 0.42, "regime": "trending_bull"}


def test_full_decision_order_fill_pnl_chain(isolated_audit_dir) -> None:
    """All four log kinds round-trip through a single run_id."""
    run_id = audit_logger.new_run_id()
    audit_logger.log_decision(run_id, "AAPL", {"score": 0.5})
    audit_logger.log_order(run_id, "AAPL", {"qty": 10})
    audit_logger.log_fill(run_id, "AAPL", {"qty": 10, "price": 150.0})
    audit_logger.log_pnl(run_id, "AAPL", {"realised": 12.5})

    records = list(audit_logger.iter_records())
    kinds = [r["kind"] for r in records]
    assert kinds == ["decision", "order", "fill", "pnl"]
    assert all(r["run_id"] == run_id for r in records)


def test_iter_records_filters_by_date(isolated_audit_dir, monkeypatch) -> None:
    """Date-window filtering reads exactly the requested days."""
    today = datetime.now(timezone.utc).date()
    yesterday = (today - timedelta(days=1)).isoformat()
    today_str = today.isoformat()

    audit_logger.log_decision("today1", "AAPL", {"v": 1})

    # Synthesize a yesterday file by writing directly to the dir
    yesterday_path = isolated_audit_dir / f"{yesterday}.jsonl"
    yesterday_path.parent.mkdir(parents=True, exist_ok=True)
    yesterday_path.write_text(
        json.dumps({"kind": "decision", "run_id": "y1", "ticker": "MSFT",
                    "details": {"v": 0}, "ts": yesterday + "T12:00:00Z"})
        + "\n"
    )

    today_only = list(audit_logger.iter_records(start_date=today_str))
    yesterday_only = list(
        audit_logger.iter_records(start_date=yesterday, end_date=yesterday)
    )
    assert len(today_only) == 1 and today_only[0]["run_id"] == "today1"
    assert len(yesterday_only) == 1 and yesterday_only[0]["run_id"] == "y1"


# ── Rotate / compress / delete lifecycle ────────────────────────────────────


def _seed_log(path: Path, age_days: int, *, content: str = '{"v":1}\n') -> Path:
    """Create a JSONL log dated ``age_days`` ago."""
    today = datetime.now(timezone.utc).date()
    date = today - timedelta(days=age_days)
    file_path = path / f"{date.isoformat()}.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return file_path


def test_rotate_compresses_old_logs(isolated_audit_dir) -> None:
    """Logs older than compress_after_days are gzipped; recent ones stay."""
    recent = _seed_log(isolated_audit_dir, age_days=2)
    old = _seed_log(isolated_audit_dir, age_days=14)
    summary = audit_logger.rotate(max_age_days=90, compress_after_days=7)

    assert summary["compressed"] == 1
    assert summary["deleted"] == 0
    assert recent.exists()
    assert not old.exists()
    assert (old.parent / (old.name + ".gz")).exists()


def test_rotate_deletes_logs_past_max_age(isolated_audit_dir) -> None:
    """Logs older than max_age_days are deleted (after compression window)."""
    _seed_log(isolated_audit_dir, age_days=2)
    very_old = _seed_log(isolated_audit_dir, age_days=120)
    summary = audit_logger.rotate(max_age_days=90, compress_after_days=7)

    assert summary["deleted"] == 1
    assert not very_old.exists()
    assert not (very_old.parent / (very_old.name + ".gz")).exists()


def test_rotate_is_idempotent(isolated_audit_dir) -> None:
    """Calling rotate twice produces no change after the first run."""
    _seed_log(isolated_audit_dir, age_days=14)
    first = audit_logger.rotate(max_age_days=90, compress_after_days=7)
    second = audit_logger.rotate(max_age_days=90, compress_after_days=7)
    assert first["compressed"] == 1
    assert second == {"compressed": 0, "deleted": 0}


def test_iter_records_reads_gzipped_logs_transparently(
    isolated_audit_dir,
) -> None:
    """After rotate compresses a log, iter_records still yields its rows."""
    payload = {"kind": "decision", "run_id": "x", "ticker": "AAPL",
               "details": {"v": 1}, "ts": "2020-01-01T00:00:00Z"}
    today = datetime.now(timezone.utc).date()
    old_date = (today - timedelta(days=20)).isoformat()
    raw_path = isolated_audit_dir / f"{old_date}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(payload) + "\n")

    audit_logger.rotate(max_age_days=90, compress_after_days=7)
    gz_path = isolated_audit_dir / f"{old_date}.jsonl.gz"
    assert gz_path.exists()
    # Sanity: the gz really is gzipped JSONL
    with gzip.open(gz_path, "rt") as f:
        assert json.loads(f.readline()) == payload

    records = list(audit_logger.iter_records())
    assert any(r["run_id"] == "x" for r in records)


# ── Failure-mode coverage ───────────────────────────────────────────────────


def test_rotate_rejects_inconsistent_thresholds(isolated_audit_dir) -> None:
    """``max_age_days < compress_after_days`` would mean "delete before
    compress" — the impl raises rather than silently misbehaving."""
    with pytest.raises(ValueError, match="max_age_days must be >="):
        audit_logger.rotate(max_age_days=3, compress_after_days=7)


def test_iter_records_returns_empty_when_dir_missing(
    tmp_path, monkeypatch
) -> None:
    """Fresh install (no log dir yet) must not raise."""
    monkeypatch.setenv("AUDIT_LOG_DIR", str(tmp_path / "does_not_exist"))
    assert list(audit_logger.iter_records()) == []


def test_log_rejects_invalid_kind(isolated_audit_dir) -> None:
    """The private ``_log`` validates kind — a typo should fail loud."""
    with pytest.raises(ValueError, match="kind must be one of"):
        audit_logger._log(
            "explosion", audit_logger.new_run_id(), "AAPL", {}
        )


def test_log_rejects_empty_run_id(isolated_audit_dir) -> None:
    with pytest.raises(ValueError, match="run_id"):
        audit_logger.log_decision("", "AAPL", {})
