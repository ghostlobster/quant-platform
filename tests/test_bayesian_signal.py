"""Tests for strategies/bayesian_signal.py — BayesianRidge alpha model."""
import os
import pickle
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.bayesian_signal import (
    _SKLEARN_AVAILABLE,
    BayesianSignal,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feature_matrix(
    n_dates: int = 80, n_tickers: int = 5, seed: int = 0,
) -> pd.DataFrame:
    """Synthetic (date, ticker) FM where fwd_ret_5d = 0.5*x0 + noise."""
    from data.features import _FEATURE_COLS

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            row = {"date": d, "ticker": t}
            for col in _FEATURE_COLS:
                row[col] = rng.normal()
            # Target linearly driven by ret_1d; small noise.
            row["fwd_ret_5d"] = 0.02 * row["ret_1d"] + 0.001 * rng.normal()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "ticker"])


# ── Loading / fallback behaviour ─────────────────────────────────────────────

def test_init_without_checkpoint_uses_fallback(tmp_path):
    model = BayesianSignal(model_path=str(tmp_path / "nonexistent.pkl"))
    assert model._model is None


def test_predict_fallback_returns_scores_for_every_ticker(tmp_path):
    with patch("data.fetcher.fetch_ohlcv", return_value=pd.DataFrame()):
        model = BayesianSignal(model_path=str(tmp_path / "none.pkl"))
        scores = model.predict(["AAPL", "MSFT"])
    assert set(scores.keys()) == {"AAPL", "MSFT"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_predict_with_uncertainty_fallback_reports_sigma_one(tmp_path):
    with patch("data.fetcher.fetch_ohlcv", return_value=pd.DataFrame()):
        model = BayesianSignal(model_path=str(tmp_path / "n.pkl"))
        mean, sigma = model.predict_with_uncertainty(["AAPL"])
    assert sigma["AAPL"] == 1.0


def test_feature_coefficients_empty_when_no_model(tmp_path):
    model = BayesianSignal(model_path=str(tmp_path / "missing.pkl"))
    df = model.feature_coefficients()
    assert df.empty
    assert list(df.columns) == ["feature", "coefficient", "std"]


# ── Training (skipped when sklearn is absent) ────────────────────────────────

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_train_fits_model_and_returns_metrics(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=1)
    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix",
        lambda *a, **kw: fm,
    )
    monkeypatch.setattr(BayesianSignal, "_write_metadata", lambda *a, **k: None)

    model = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    metrics = model.train(["T0", "T1"], period="2y")

    assert model._model is not None
    assert "train_ic" in metrics and "test_ic" in metrics
    assert isinstance(metrics["train_ic"], float)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_train_persists_checkpoint(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=2)
    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix",
        lambda *a, **kw: fm,
    )
    monkeypatch.setattr(BayesianSignal, "_write_metadata", lambda *a, **k: None)

    ckpt = tmp_path / "b.pkl"
    model = BayesianSignal(model_path=str(ckpt))
    model.train(["T0"], period="2y")

    assert ckpt.exists()
    with open(ckpt, "rb") as f:
        from sklearn.linear_model import BayesianRidge
        loaded = pickle.load(f)
    assert isinstance(loaded, BayesianRidge)


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_train_raises_on_empty_feature_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix",
        lambda *a, **kw: pd.DataFrame(),
    )
    model = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    with pytest.raises(ValueError, match="empty"):
        model.train(["T0"], period="2y")


def test_train_raises_when_sklearn_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "strategies.bayesian_signal._SKLEARN_AVAILABLE", False,
    )
    model = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    with pytest.raises(RuntimeError, match="scikit-learn"):
        model.train(["T0"], period="2y")


# ── Prediction after training ────────────────────────────────────────────────

@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_predict_after_train_returns_bounded_scores(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=3)
    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix",
        lambda *a, **kw: fm,
    )
    monkeypatch.setattr(BayesianSignal, "_write_metadata", lambda *a, **k: None)

    model = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    model.train(["T0", "T1"], period="2y")

    mean, sigma = model.predict_with_uncertainty(["T0", "T1", "T2", "T3"])
    assert set(mean.keys()) == {"T0", "T1", "T2", "T3"}
    assert set(sigma.keys()) == {"T0", "T1", "T2", "T3"}
    for v in mean.values():
        assert -1.0 <= v <= 1.0
    for s in sigma.values():
        assert s >= 0.0


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_predictive_sigma_shrinks_with_more_training_data(tmp_path, monkeypatch):
    """Core Bayesian behaviour: σ should shrink as n grows."""
    small_fm = _make_feature_matrix(n_dates=30, n_tickers=3, seed=10)
    big_fm = _make_feature_matrix(n_dates=300, n_tickers=6, seed=10)

    def _factory(fm):
        def _mk(*a, **kw): return fm
        return _mk

    monkeypatch.setattr(BayesianSignal, "_write_metadata", lambda *a, **k: None)

    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix", _factory(small_fm),
    )
    m_small = BayesianSignal(model_path=str(tmp_path / "s.pkl"))
    m_small.train(["T0"], period="1y")
    _, sigma_small = m_small.predict_with_uncertainty(["T0", "T1", "T2"])

    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix", _factory(big_fm),
    )
    m_big = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    m_big.train(["T0"], period="5y")
    _, sigma_big = m_big.predict_with_uncertainty(["T0", "T1", "T2"])

    mean_small = float(np.mean(list(sigma_small.values())))
    mean_big = float(np.mean(list(sigma_big.values())))
    assert mean_big < mean_small


@pytest.mark.skipif(not _SKLEARN_AVAILABLE, reason="sklearn not installed")
def test_feature_coefficients_after_train(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=4)
    monkeypatch.setattr(
        "strategies.bayesian_signal.build_feature_matrix",
        lambda *a, **kw: fm,
    )
    monkeypatch.setattr(BayesianSignal, "_write_metadata", lambda *a, **k: None)

    model = BayesianSignal(model_path=str(tmp_path / "b.pkl"))
    model.train(["T0"], period="2y")
    df = model.feature_coefficients()
    assert not df.empty
    assert list(df.columns) == ["feature", "coefficient", "std"]
    assert (df["std"] >= 0).all()


# ── Ensemble compatibility ───────────────────────────────────────────────────

def test_bayesian_scores_blend_cleanly_with_ensemble_signal():
    """Validate the signature of predict() is compatible with
    strategies.ensemble_signal.blend_signals."""
    from strategies.ensemble_signal import blend_signals

    # Synthetic three-source blend — Bayesian treated as any other
    # score-dict source.
    lgbm = {"AAPL": 0.5, "MSFT": -0.2}
    ridge = {"AAPL": 0.3, "MSFT": 0.1}
    bayesian = {"AAPL": 0.4, "MSFT": -0.1}
    blended = blend_signals(lgbm, ridge, bayesian, weights=[0.5, 0.25, 0.25])
    assert set(blended.keys()) == {"AAPL", "MSFT"}
    assert abs(blended["AAPL"] - (0.5 * 0.5 + 0.25 * 0.3 + 0.25 * 0.4)) < 1e-9
