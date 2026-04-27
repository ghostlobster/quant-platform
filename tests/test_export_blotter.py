"""
tests/test_export_blotter.py — CSV blotter export from journal.
"""
from __future__ import annotations

import csv
import io
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import scripts.export_blotter as eb


@pytest.fixture
def fake_journal(monkeypatch):
    captured: dict = {}

    def _fake_get_journal(start_date=None, end_date=None, ticker=None):
        captured["args"] = (start_date, end_date, ticker)
        return pd.DataFrame(
            [
                {
                    "id": 1, "ticker": "AAPL", "side": "BUY", "qty": 10,
                    "entry_price": 100.0, "entry_time": "2026-04-01T10:00",
                    "exit_price": 110.0,  "exit_time":  "2026-04-15T10:00",
                    "pnl": 100.0, "exit_reason": "target",
                    "signal_source": "ml", "regime": "bull",
                    "extra_field": "drop me",  # extrasaction=ignore drops this
                },
                {
                    "id": 2, "ticker": "SPY", "side": "BUY", "qty": 5,
                    "entry_price": 450.0, "entry_time": "2026-04-05T10:00",
                    "exit_price": 445.0,  "exit_time":  "2026-04-09T10:00",
                    "pnl": -25.0, "exit_reason": "stop",
                    "signal_source": "test", "regime": "bear",
                },
            ]
        )

    monkeypatch.setattr(
        "journal.trading_journal.get_journal", _fake_get_journal,
    )
    return captured


def test_export_writes_csv_to_file(tmp_path, fake_journal):
    out_path = tmp_path / "blotter.csv"
    n = eb.export(
        start_date="2026-04-01", end_date="2026-04-30",
        ticker=None, out_path=str(out_path),
    )
    assert n == 2
    rows = list(csv.DictReader(out_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert rows[0]["ticker"] == "AAPL"
    assert rows[1]["pnl"] == "-25.0"
    assert "extra_field" not in rows[0]


def test_export_writes_csv_to_stdout(monkeypatch, fake_journal):
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    n = eb.export(start_date=None, end_date=None, ticker=None, out_path=None)
    assert n == 2
    output = captured.getvalue()
    assert "AAPL" in output
    assert "SPY"  in output
    assert "extra_field" not in output


def test_export_handles_empty_journal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "journal.trading_journal.get_journal",
        lambda **kw: pd.DataFrame(),
    )
    out_path = tmp_path / "empty.csv"
    n = eb.export(None, None, None, str(out_path))
    assert n == 0
    # Even with no rows, the header should be present.
    text = out_path.read_text(encoding="utf-8")
    assert "ticker" in text


def test_main_returns_zero(monkeypatch, fake_journal, tmp_path):
    out = tmp_path / "blotter.csv"
    rc = eb.main(["--from", "2026-04-01", "--to", "2026-04-30",
                  "--out", str(out)])
    assert rc == 0
    assert out.exists()


def test_main_passes_filters_to_get_journal(fake_journal, tmp_path):
    out = tmp_path / "blotter.csv"
    eb.main([
        "--from", "2026-04-01", "--to", "2026-04-30",
        "--ticker", "AAPL", "--out", str(out),
    ])
    assert fake_journal["args"] == ("2026-04-01", "2026-04-30", "AAPL")
