"""Unit tests for strategies/ml_tuning.py.

Core logic (purged splits, error paths) tests run without optuna. End-to-end
tuning tests are skipped when optuna is not installed.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.ml_tuning import (
    _OPTUNA_AVAILABLE,
    _purged_splits,
    load_best_params,
    save_best_params,
    tune_lgbm_hyperparams,
)


def _fake_feature_matrix(n_dates: int = 20, n_tickers: int = 4) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    tickers = [f"T{i}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    rng = np.random.RandomState(0)
    data = {
        "ret_1d": rng.randn(len(index)),
        "ret_5d": rng.randn(len(index)),
        "fwd_ret_5d": rng.randn(len(index)),
    }
    return pd.DataFrame(data, index=index)


def test_purged_splits_has_embargo_gap():
    fm = _fake_feature_matrix(n_dates=30)
    splits = _purged_splits(fm, n_splits=3, embargo=2)
    assert len(splits) > 0

    dates = fm.index.get_level_values("date")
    for train_idx, test_idx in splits:
        max_train_date = dates[train_idx].max()
        min_test_date = dates[test_idx].min()
        assert (min_test_date - max_train_date).days >= 2


def test_purged_splits_empty_for_small_data():
    fm = _fake_feature_matrix(n_dates=3)
    splits = _purged_splits(fm, n_splits=5, embargo=1)
    assert splits == []


def test_tune_lgbm_raises_when_optuna_missing(monkeypatch):
    monkeypatch.setattr("strategies.ml_tuning._OPTUNA_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="optuna"):
        tune_lgbm_hyperparams(["AAPL"], period="1y", n_trials=1)


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_tune_lgbm_raises_on_empty_feature_matrix(monkeypatch):
    monkeypatch.setattr("strategies.ml_tuning._LGBM_AVAILABLE", True)
    monkeypatch.setattr(
        "strategies.ml_tuning.build_feature_matrix",
        lambda *args, **kwargs: pd.DataFrame(),
    )
    with pytest.raises(RuntimeError, match="feature matrix is empty"):
        tune_lgbm_hyperparams(["AAPL"], period="1y", n_trials=1)


def test_save_and_load_best_params_round_trip(tmp_path, monkeypatch):
    """save_best_params → load_best_params recovers the original dict."""
    import sqlite3
    db_path = tmp_path / "params.db"

    def _fake_conn():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_best_params (
                model_name   TEXT    PRIMARY KEY,
                updated_at   REAL    NOT NULL,
                params_json  TEXT    NOT NULL,
                best_ic      REAL
            )
        """)
        return conn

    monkeypatch.setattr("data.db.get_connection", _fake_conn)

    params = {
        "n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
    }
    save_best_params("lgbm_alpha", params, best_ic=0.042)
    recovered = load_best_params("lgbm_alpha")
    assert recovered == params

    # Missing row returns None (not an exception)
    assert load_best_params("does_not_exist") is None


def test_save_best_params_upserts_on_conflict(tmp_path, monkeypatch):
    """Re-saving for the same model_name overwrites the previous row."""
    import sqlite3
    db_path = tmp_path / "params.db"

    def _fake_conn():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_best_params (
                model_name   TEXT    PRIMARY KEY,
                updated_at   REAL    NOT NULL,
                params_json  TEXT    NOT NULL,
                best_ic      REAL
            )
        """)
        return conn

    monkeypatch.setattr("data.db.get_connection", _fake_conn)

    save_best_params("lgbm_alpha", {"lr": 0.01}, best_ic=0.01)
    save_best_params("lgbm_alpha", {"lr": 0.05}, best_ic=0.08)
    assert load_best_params("lgbm_alpha") == {"lr": 0.05}


def test_load_best_params_handles_db_error(monkeypatch):
    """Transient DB errors return None rather than propagating."""
    def _broken(): raise RuntimeError("db down")
    monkeypatch.setattr("data.db.get_connection", _broken)
    assert load_best_params("lgbm_alpha") is None


@pytest.mark.skipif(not _OPTUNA_AVAILABLE, reason="optuna not installed")
def test_tune_lgbm_returns_valid_params(monkeypatch):
    """End-to-end smoke test with a tiny synthetic feature matrix."""
    pytest.importorskip("lightgbm")
    monkeypatch.setattr("strategies.ml_tuning._LGBM_AVAILABLE", True)
    fake = _fake_feature_matrix(n_dates=80, n_tickers=6)
    monkeypatch.setattr(
        "strategies.ml_tuning.build_feature_matrix",
        lambda *args, **kwargs: fake,
    )
    monkeypatch.setattr(
        "strategies.ml_tuning._FEATURE_COLS",
        ["ret_1d", "ret_5d"],
    )

    result = tune_lgbm_hyperparams(
        ["T0"], period="1y", n_trials=2, n_splits=2, embargo=1,
    )
    assert set(result) == {"best_params", "best_ic", "n_trials", "n_samples"}
    assert result["n_trials"] == 2
    assert "learning_rate" in result["best_params"]
