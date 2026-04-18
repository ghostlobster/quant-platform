"""Tests for strategies/cnn_signal.py — CNN over chart images."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import torch  # noqa: F401
    _TORCH = True
except ImportError:
    _TORCH = False

skip_no_torch = pytest.mark.skipif(not _TORCH, reason="torch not installed")


def _ohlcv(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close,
        "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def _feature_matrix(tickers: list[str], n_dates: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows, idx = [], []
    rng = np.random.default_rng(42)
    for date in dates:
        for ticker in tickers:
            rows.append({"fwd_ret_5d": float(rng.standard_normal()) * 0.01})
            idx.append((date, ticker))
    mi = pd.MultiIndex.from_tuples(idx, names=["date", "ticker"])
    return pd.DataFrame(rows, index=mi)


# ── Gating ────────────────────────────────────────────────────────────────────

def test_train_raises_when_torch_missing():
    from strategies import cnn_signal as mod
    with (
        patch.object(mod, "_TORCH_AVAILABLE", False),
        patch.object(mod.CNNSignal, "_load_if_available"),
    ):
        signal = mod.CNNSignal(model_path="/tmp/x.pt")
        with pytest.raises(RuntimeError, match="PyTorch"):
            signal.train(["AAPL"])


def test_predict_no_model_falls_back_to_momentum():
    from strategies import cnn_signal as mod
    with (
        patch.object(mod.CNNSignal, "_load_if_available"),
        patch.object(
            mod.CNNSignal, "_momentum_fallback", return_value={"AAPL": 0.4},
        ) as fb,
    ):
        signal = mod.CNNSignal(model_path="/tmp/x.pt")
        out = signal.predict(["AAPL"], period="6mo")
    fb.assert_called_once()
    assert out == {"AAPL": 0.4}


def test_momentum_fallback_handles_missing_data():
    from strategies.cnn_signal import CNNSignal
    with patch("strategies.cnn_signal.CNNSignal._load_if_available"):
        signal = CNNSignal(model_path="/tmp/x.pt")
    with patch("strategies.cnn_signal.fetch_ohlcv", return_value=None):
        out = signal._momentum_fallback(["AAPL"], "6mo")
    assert out == {"AAPL": 0.0}


def test_load_if_available_corrupt_file_does_not_raise(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not_a_real_torch_pickle")
    from strategies.cnn_signal import CNNSignal
    signal = CNNSignal(model_path=str(bad))
    assert signal._model is None


def test_write_metadata_silences_db_errors():
    from strategies.cnn_signal import CNNSignal
    with patch("strategies.cnn_signal.CNNSignal._load_if_available"):
        signal = CNNSignal(model_path="/tmp/x.pt")
    with patch("data.db.get_connection", side_effect=RuntimeError("no db")):
        signal._write_metadata(n_tickers=1, period="2y", train_ic=0.1, test_ic=0.05)


# ── _build_image_dataset ──────────────────────────────────────────────────────

def test_build_image_dataset_drops_short_history_and_nan_target():
    from strategies.cnn_signal import _build_image_dataset
    closes = {"AAPL": _ohlcv(n=80)["Close"]}
    fm = _feature_matrix(["AAPL"], n_dates=60)
    # Force one NaN target
    fm.iloc[0, fm.columns.get_loc("fwd_ret_5d")] = np.nan
    X, y, idx = _build_image_dataset(fm, closes, window=16)
    assert X is not None and y is not None
    assert X.shape[1:] == (1, 16, 16)
    assert len(y) == X.shape[0]
    assert len(idx) == len(y)


def test_build_image_dataset_returns_none_when_no_history():
    from strategies.cnn_signal import _build_image_dataset
    fm = _feature_matrix(["AAPL"], n_dates=10)
    closes = {"AAPL": _ohlcv(n=5)["Close"]}  # too short for window=16
    X, y, idx = _build_image_dataset(fm, closes, window=16)
    assert X is None and y is None and idx is None


# ── Torch-required full surface ───────────────────────────────────────────────

@skip_no_torch
def test_train_and_predict_end_to_end_with_synthetic_data():
    from strategies.cnn_signal import CNNSignal

    fm = _feature_matrix(["AAPL", "MSFT"], n_dates=60)

    def fake_fetch(ticker, period):
        return _ohlcv(n=80, seed=hash(ticker) % 1000)

    with (
        patch("strategies.cnn_signal.build_feature_matrix", return_value=fm),
        patch("strategies.cnn_signal.fetch_ohlcv", side_effect=fake_fetch),
        patch("strategies.cnn_signal.CNNSignal._load_if_available"),
        patch("strategies.cnn_signal.CNNSignal._write_metadata"),
        patch("strategies.cnn_signal.torch.save"),
        patch("strategies.cnn_signal.Path.mkdir"),
    ):
        signal = CNNSignal(model_path="/tmp/test_cnn.pt", window=16)
        result = signal.train(["AAPL", "MSFT"], period="2y", epochs=1, batch_size=16)
        assert result["n_train_samples"] > 0
        scores = signal.predict(["AAPL", "MSFT"], period="6mo")
        assert set(scores) == {"AAPL", "MSFT"}
        for v in scores.values():
            assert -1.0 <= v <= 1.0


@skip_no_torch
def test_train_raises_on_empty_feature_matrix():
    from strategies.cnn_signal import CNNSignal
    with (
        patch("strategies.cnn_signal.build_feature_matrix", return_value=pd.DataFrame()),
        patch("strategies.cnn_signal.CNNSignal._load_if_available"),
    ):
        signal = CNNSignal(model_path="/tmp/x.pt")
        with pytest.raises(ValueError, match="empty"):
            signal.train(["AAPL"])


@skip_no_torch
def test_predict_handles_missing_history_gracefully():
    from strategies.cnn_signal import CNNSignal
    with patch("strategies.cnn_signal.CNNSignal._load_if_available"):
        signal = CNNSignal(model_path="/tmp/x.pt", window=16)
    signal._model = MagicMock()
    with patch("strategies.cnn_signal.fetch_ohlcv", return_value=None):
        out = signal.predict(["AAPL"], period="6mo")
    assert out == {"AAPL": 0.0}
