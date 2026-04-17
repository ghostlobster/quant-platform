"""Tests for strategies/dl_signal.py — LSTM alpha model."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.dl_signal import (
    _TORCH_AVAILABLE,
    DLSignal,
    _build_windowed_tensors,
)

# ── _build_windowed_tensors (pure NumPy, no torch) ───────────────────────────

def _make_feature_matrix(
    n_dates: int = 60, n_tickers: int = 4, seed: int = 0,
) -> pd.DataFrame:
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
            row["fwd_ret_5d"] = 0.01 * row["ret_1d"] + 0.001 * rng.normal()
            rows.append(row)
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_build_windowed_tensors_shapes():
    from data.features import _FEATURE_COLS

    fm = _make_feature_matrix(n_dates=40, n_tickers=3)
    feats = [c for c in _FEATURE_COLS if c in fm.columns]
    X, y, idx = _build_windowed_tensors(fm, feats, "fwd_ret_5d", window=5)
    assert X.ndim == 3
    assert X.shape[1] == 5
    assert X.shape[2] == len(feats)
    assert len(y) == X.shape[0]
    assert len(idx) == len(y)


def test_build_windowed_tensors_skips_tickers_with_too_few_bars():
    from data.features import _FEATURE_COLS

    fm = _make_feature_matrix(n_dates=40, n_tickers=3)
    feats = [c for c in _FEATURE_COLS if c in fm.columns]
    X, y, idx = _build_windowed_tensors(fm, feats, "fwd_ret_5d", window=50)
    # Window larger than series → nothing emitted.
    assert X is None and y is None and idx is None


def test_build_windowed_tensors_multiindex_order():
    from data.features import _FEATURE_COLS

    fm = _make_feature_matrix(n_dates=30, n_tickers=2)
    feats = [c for c in _FEATURE_COLS if c in fm.columns]
    _, _, idx = _build_windowed_tensors(fm, feats, "fwd_ret_5d", window=5)
    assert idx.names == ["date", "ticker"]


# ── DLSignal — torch-free paths (always run) ─────────────────────────────────

def test_init_without_checkpoint_starts_untrained(tmp_path):
    model = DLSignal(model_path=str(tmp_path / "nonexistent.pt"))
    assert not model.is_trained()
    assert model.info() is None


def test_predict_without_model_falls_back_to_momentum(tmp_path):
    with patch("data.fetcher.fetch_ohlcv", return_value=pd.DataFrame()):
        model = DLSignal(model_path=str(tmp_path / "none.pt"))
        scores = model.predict(["AAPL", "MSFT"])
    assert set(scores.keys()) == {"AAPL", "MSFT"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_train_raises_when_torch_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("strategies.dl_signal._TORCH_AVAILABLE", False)
    model = DLSignal(model_path=str(tmp_path / "x.pt"))
    with pytest.raises(RuntimeError, match="torch"):
        model.train(["T0"], period="2y")


def test_predict_when_torch_missing_uses_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr("strategies.dl_signal._TORCH_AVAILABLE", False)
    with patch("data.fetcher.fetch_ohlcv", return_value=pd.DataFrame()):
        model = DLSignal(model_path=str(tmp_path / "x.pt"))
        scores = model.predict(["AAPL"])
    assert scores["AAPL"] == 0.0


# ── DLSignal — torch path (skipped when torch is absent) ─────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_train_fits_and_persists_checkpoint(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=7)
    monkeypatch.setattr(
        "strategies.dl_signal.build_feature_matrix", lambda *a, **kw: fm,
    )

    ckpt = tmp_path / "dl.pt"
    model = DLSignal(model_path=str(ckpt), window=5)
    metrics = model.train(["T0"], period="1y", epochs=2)

    assert "train_ic" in metrics and "test_ic" in metrics
    assert isinstance(metrics["train_ic"], float)
    assert ckpt.exists()


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_predict_after_train_returns_bounded_scores(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=4, seed=8)
    monkeypatch.setattr(
        "strategies.dl_signal.build_feature_matrix", lambda *a, **kw: fm,
    )

    model = DLSignal(model_path=str(tmp_path / "dl.pt"), window=5)
    model.train(["T0"], period="1y", epochs=2)
    scores = model.predict(["T0", "T1", "T2"], period="6mo")
    assert set(scores.keys()) == {"T0", "T1", "T2"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
def test_checkpoint_round_trip_preserves_arch(tmp_path, monkeypatch):
    fm = _make_feature_matrix(n_dates=80, n_tickers=3, seed=9)
    monkeypatch.setattr(
        "strategies.dl_signal.build_feature_matrix", lambda *a, **kw: fm,
    )
    ckpt = tmp_path / "dl.pt"
    m1 = DLSignal(model_path=str(ckpt), window=5)
    m1.train(["T0"], period="1y", epochs=2, hidden=16)

    # Fresh instance should load the saved weights and metadata.
    m2 = DLSignal(model_path=str(ckpt))
    assert m2.is_trained()
    info = m2.info()
    assert info is not None
    assert info["window"] == 5


# ── Ensemble compatibility ───────────────────────────────────────────────────

def test_dl_scores_blend_cleanly_with_ensemble():
    """The predict() return-type must slot into ensemble_signal.blend_signals."""
    from strategies.ensemble_signal import blend_signals

    lgbm = {"AAPL": 0.3, "MSFT": -0.1}
    dl = {"AAPL": -0.2, "MSFT": 0.4}
    blended = blend_signals(lgbm, dl, weights=[0.6, 0.4])
    assert set(blended.keys()) == {"AAPL", "MSFT"}
    for v in blended.values():
        assert -1.0 <= v <= 1.0
