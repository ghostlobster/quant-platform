"""
audit/logger.py — append-only JSONL audit logger.

Every meaningful order-lifecycle event lands as a single JSON line on
``audit/<run_id>.jsonl``. The format is intentionally trivial so an
operator can ``grep`` / ``jq`` the log without needing the platform
running. Each record carries:

    {
      "ts":       "2026-04-27T03:14:15+00:00",
      "run_id":   "<8-char>",
      "kind":     "decision" | "order" | "fill" | "pnl",
      "ticker":   "AAPL",
      "details":  { ... arbitrary JSON ... }
    }

The default log path is the ``audit/`` directory at the repo root. Set
``AUDIT_LOG_DIR`` to override (e.g. for tests).

Public API
----------
    log_decision(run_id, ticker, details)
    log_order(run_id, ticker, details)
    log_fill(run_id, ticker, details)
    log_pnl(run_id, ticker, details)
    iter_records(start: str | None, end: str | None) -> list[dict]
    rotate(max_age_days: int) -> int   compress + delete old logs
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
import uuid
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_DIR = Path(os.environ.get("AUDIT_LOG_DIR", "audit/log"))
_LOCK = threading.Lock()
_VALID_KINDS = ("decision", "order", "fill", "pnl")


def new_run_id() -> str:
    """Return a fresh 8-character hex run id (UUID4 prefix)."""
    return uuid.uuid4().hex[:8]


def _log_dir() -> Path:
    raw = os.environ.get("AUDIT_LOG_DIR")
    return Path(raw) if raw else _DEFAULT_DIR


def _today_path() -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _log_dir() / f"{today}.jsonl"


def _write_record(record: dict) -> None:
    path = _today_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")


def _log(kind: str, run_id: str, ticker: str, details: dict) -> None:
    if kind not in _VALID_KINDS:
        raise ValueError(
            f"kind must be one of {_VALID_KINDS}, got {kind!r}",
        )
    if not run_id:
        raise ValueError("run_id must be non-empty")
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "kind": kind,
        "ticker": ticker.upper().strip() if ticker else "",
        "details": dict(details or {}),
    }
    try:
        _write_record(record)
    except OSError as exc:  # pragma: no cover — disk full / permission denied
        logger.warning("audit_log: write failed", error=str(exc))


def log_decision(run_id: str, ticker: str, details: dict | None = None) -> None:
    _log("decision", run_id, ticker, details or {})


def log_order(run_id: str, ticker: str, details: dict | None = None) -> None:
    _log("order", run_id, ticker, details or {})


def log_fill(run_id: str, ticker: str, details: dict | None = None) -> None:
    _log("fill", run_id, ticker, details or {})


def log_pnl(run_id: str, ticker: str, details: dict | None = None) -> None:
    _log("pnl", run_id, ticker, details or {})


# ── Iteration / export ───────────────────────────────────────────────────────

def _date_for_path(path: Path) -> str:
    """Return the YYYY-MM-DD prefix for a known audit-log filename."""
    name = path.name
    if name.endswith(".jsonl"):
        return name[:-len(".jsonl")]
    if name.endswith(".jsonl.gz"):
        return name[:-len(".jsonl.gz")]
    return ""


def iter_records(
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterable[dict]:
    """Yield records from every audit log between ``start_date`` and
    ``end_date`` inclusive (ISO ``YYYY-MM-DD`` strings; both optional).

    Reads ``.jsonl`` and ``.jsonl.gz`` files transparently so callers
    don't need to know which dates have been rotated.
    """
    directory = _log_dir()
    if not directory.exists():
        return
    files = sorted(directory.glob("*.jsonl*"))
    for path in files:
        date_str = _date_for_path(path)
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def rotate(max_age_days: int = 90, compress_after_days: int = 7) -> dict:
    """Compress old logs (gzip) and delete logs past ``max_age_days``.

    Returns a summary dict ``{"compressed": N, "deleted": M}``.
    """
    if max_age_days < compress_after_days:
        raise ValueError(
            "max_age_days must be >= compress_after_days "
            f"({max_age_days} vs {compress_after_days})",
        )
    directory = _log_dir()
    if not directory.exists():
        return {"compressed": 0, "deleted": 0}

    today = datetime.now(timezone.utc).date()
    compress_cutoff = today - timedelta(days=compress_after_days)
    delete_cutoff = today - timedelta(days=max_age_days)

    compressed = 0
    deleted = 0
    for path in sorted(directory.glob("*.jsonl*")):
        date_str = _date_for_path(path)
        if not date_str:
            continue
        try:
            file_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_date < delete_cutoff:
            path.unlink()
            deleted += 1
            continue
        if path.suffix == ".jsonl" and file_date < compress_cutoff:
            gz_path = path.with_suffix(".jsonl.gz")
            with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            path.unlink()
            compressed += 1
    return {"compressed": compressed, "deleted": deleted}
