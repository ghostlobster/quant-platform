"""Tests for analysis/retrain_roi.py."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Redirect data.db to a throw-away SQLite file for isolation."""
    db_path = tmp_path / "quant.db"
    monkeypatch.setattr("data.db._DB_PATH", Path(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE model_metadata (
            model_name     TEXT NOT NULL,
            trained_at     REAL NOT NULL,
            train_ic       REAL,
            test_ic        REAL,
            n_tickers      INTEGER,
            period         TEXT,
            test_ic_delta  REAL,
            PRIMARY KEY (model_name, trained_at)
        )
        """
    )
    conn.commit()
    conn.close()
    return db_path


def _insert_deltas(db_path: Path, model_name: str, deltas: list[float | None]) -> None:
    """Insert rows with monotonically increasing trained_at timestamps."""
    conn = sqlite3.connect(str(db_path))
    with conn:
        for i, d in enumerate(deltas):
            conn.execute(
                "INSERT INTO model_metadata "
                "(model_name, trained_at, test_ic, test_ic_delta) VALUES (?, ?, ?, ?)",
                (model_name, 1_000.0 + float(i), 0.05, d),
            )
    conn.close()


def test_retrain_roi_empty_returns_empty_series(tmp_db):
    from analysis.retrain_roi import retrain_roi

    series, slope = retrain_roi("lgbm_alpha", n=6)
    assert series.empty
    assert slope == 0.0


def test_retrain_roi_returns_last_n_chronological(tmp_db):
    from analysis.retrain_roi import retrain_roi

    _insert_deltas(tmp_db, "lgbm_alpha", [-0.02, -0.01, 0.0, 0.01, 0.02])
    series, _ = retrain_roi("lgbm_alpha", n=3)
    # Latest 3 in chronological order: 0.0, 0.01, 0.02
    assert list(series.values) == pytest.approx([0.0, 0.01, 0.02])


def test_retrain_roi_improving_positive_slope(tmp_db):
    from analysis.retrain_roi import retrain_roi

    _insert_deltas(tmp_db, "lgbm_alpha", [0.01, 0.02, 0.03])
    _, slope = retrain_roi("lgbm_alpha", n=3)
    assert slope > 0


def test_retrain_roi_flat_zero_slope(tmp_db):
    from analysis.retrain_roi import retrain_roi

    _insert_deltas(tmp_db, "lgbm_alpha", [0.01, 0.01, 0.01])
    _, slope = retrain_roi("lgbm_alpha", n=3)
    assert slope == pytest.approx(0.0)


def test_retrain_roi_degrading_negative_slope(tmp_db):
    from analysis.retrain_roi import retrain_roi

    _insert_deltas(tmp_db, "lgbm_alpha", [0.03, 0.01, -0.02])
    _, slope = retrain_roi("lgbm_alpha", n=3)
    assert slope < 0


def test_retrain_roi_ignores_null_deltas(tmp_db):
    from analysis.retrain_roi import retrain_roi

    # Legacy rows with NULL delta should be filtered out by the query
    _insert_deltas(tmp_db, "lgbm_alpha", [None, None, 0.01, 0.02, 0.03])
    series, _ = retrain_roi("lgbm_alpha", n=10)
    assert len(series) == 3


def test_is_ic_plateau_flat_series_fires(tmp_db):
    from analysis.retrain_roi import is_ic_plateau

    _insert_deltas(tmp_db, "lgbm_alpha", [0.01, 0.01, 0.01])
    assert is_ic_plateau("lgbm_alpha", n=3) is True


def test_is_ic_plateau_degrading_series_fires(tmp_db):
    from analysis.retrain_roi import is_ic_plateau

    _insert_deltas(tmp_db, "lgbm_alpha", [0.03, 0.02, 0.01])
    assert is_ic_plateau("lgbm_alpha", n=3) is True


def test_is_ic_plateau_improving_series_silent(tmp_db):
    from analysis.retrain_roi import is_ic_plateau

    _insert_deltas(tmp_db, "lgbm_alpha", [0.01, 0.02, 0.03])
    assert is_ic_plateau("lgbm_alpha", n=3) is False


def test_is_ic_plateau_insufficient_history_silent(tmp_db):
    from analysis.retrain_roi import is_ic_plateau

    # Only 2 deltas — not yet enough to decide
    _insert_deltas(tmp_db, "lgbm_alpha", [0.01, 0.01])
    assert is_ic_plateau("lgbm_alpha", n=3) is False
