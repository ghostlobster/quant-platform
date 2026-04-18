"""Tests for strategies/mlp_signal.py — feed-forward MLP alpha."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import sklearn  # noqa: F401
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

skip_no_sklearn = pytest.mark.skipif(not _SKLEARN, reason="scikit-learn not installed")


def _make_feature_matrix(n_dates: int = 40, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_21d",
        "skew_21d", "kurt_21d", "autocorr_1", "realised_vol_21d",
        "vol_ratio_20d", "vol_zscore_20d",
        "vpin_50d", "kyle_lambda_21d",
    ]
    rng = np.random.default_rng(42)
    rows, idx = [], []
    for date in dates:
        for ticker in tickers:
            row = dict(zip(feature_cols, rng.standard_normal(len(feature_cols))))
            row["fwd_ret_5d"] = rng.standard_normal()
            rows.append(row)
            idx.append((date, ticker))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["date", "ticker"]))


def _mock_mlp(n_features: int = 12, hidden: int = 8):
    mock = MagicMock()
    mock.coefs_ = [np.random.default_rng(0).standard_normal((n_features, hidden))]
    mock.predict.side_effect = lambda X: np.random.default_rng(1).standard_normal(len(X))
    return mock


# ── No-model / gating tests ──────────────────────────────────────────────────

def test_feature_importances_empty_without_model():
    from strategies.mlp_signal import MLPSignal
    with patch("strategies.mlp_signal.MLPSignal._load_if_available"):
        model = MLPSignal(model_path="/nonexistent.pkl")
    df = model.feature_importances()
    assert df.empty
    assert list(df.columns) == ["feature", "importance"]


def test_predict_falls_back_to_momentum_with_no_model():
    from strategies.mlp_signal import MLPSignal
    with (
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
        patch(
            "strategies.mlp_signal.MLPSignal._momentum_fallback",
            return_value={"AAPL": 0.42},
        ) as fb,
    ):
        model = MLPSignal(model_path="/nonexistent.pkl")
        result = model.predict(["AAPL"], period="6mo")
    fb.assert_called_once()
    assert result == {"AAPL": 0.42}


def test_train_raises_when_sklearn_missing():
    from strategies.mlp_signal import MLPSignal
    with (
        patch("strategies.mlp_signal._SKLEARN_AVAILABLE", False),
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
    ):
        model = MLPSignal(model_path="/tmp/x.pkl")
        with pytest.raises(RuntimeError, match="scikit-learn"):
            model.train(["AAPL"])


def test_momentum_fallback_returns_zero_on_missing_data():
    from strategies.mlp_signal import MLPSignal
    with patch("strategies.mlp_signal.MLPSignal._load_if_available"):
        model = MLPSignal(model_path="/tmp/x.pkl")
    with patch("data.fetcher.fetch_ohlcv", return_value=None):
        out = model._momentum_fallback(["AAPL"], "6mo")
    assert out == {"AAPL": 0.0}


# ── Mocked-MLP tests (no real sklearn run) ────────────────────────────────────

def test_train_with_mocked_mlp():
    fm = _make_feature_matrix()
    mock = _mock_mlp()
    with (
        patch("strategies.mlp_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.mlp_signal.MLPRegressor", return_value=mock),
        patch("strategies.mlp_signal.build_feature_matrix", return_value=fm),
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
        patch("strategies.mlp_signal.MLPSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.mlp_signal.pickle.dump"),
        patch("strategies.mlp_signal.Path.mkdir"),
    ):
        from strategies.mlp_signal import MLPSignal
        model = MLPSignal(model_path="models/test_mlp.pkl")
        result = model.train(["AAPL", "MSFT"], period="2y")
    for key in ("train_ic", "test_ic", "train_icir", "test_icir",
                "n_train_samples", "n_test_samples"):
        assert key in result
    assert result["n_train_samples"] > 0


def test_predict_scores_in_range_with_mocked_mlp():
    fm = _make_feature_matrix(tickers=["AAPL", "MSFT", "GOOG"])
    mock = _mock_mlp()
    with (
        patch("strategies.mlp_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.mlp_signal.MLPRegressor", return_value=mock),
        patch("strategies.mlp_signal.build_feature_matrix", return_value=fm),
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
        patch("strategies.mlp_signal.MLPSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.mlp_signal.pickle.dump"),
        patch("strategies.mlp_signal.Path.mkdir"),
    ):
        from strategies.mlp_signal import MLPSignal
        model = MLPSignal(model_path="models/test_mlp.pkl")
        model.train(["AAPL", "MSFT", "GOOG"], period="2y")
        scores = model.predict(["AAPL", "MSFT", "GOOG"], period="6mo")
    assert set(scores) == {"AAPL", "MSFT", "GOOG"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_feature_importances_sorted_descending():
    mock = _mock_mlp()
    with patch("strategies.mlp_signal.MLPSignal._load_if_available"):
        from strategies.mlp_signal import MLPSignal
        model = MLPSignal(model_path="/tmp/x.pkl")
        model._model = mock
    df = model.feature_importances()
    assert list(df.columns) == ["feature", "importance"]
    imps = df["importance"].tolist()
    assert imps == sorted(imps, reverse=True)


# ── Persistence ───────────────────────────────────────────────────────────────

def test_load_if_available_corrupt_file_does_not_raise(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not_a_real_pickle")
    from strategies.mlp_signal import MLPSignal
    model = MLPSignal(model_path=str(bad))
    assert model._model is None


def test_load_if_available_success_path(tmp_path):
    import pickle as pk
    pkl_path = tmp_path / "mlp.pkl"
    with open(pkl_path, "wb") as f:
        pk.dump({"coefs_": [np.eye(3)]}, f)
    from strategies.mlp_signal import MLPSignal
    model = MLPSignal(model_path=str(pkl_path))
    assert model._model is not None


def test_write_metadata_silences_db_errors():
    from strategies.mlp_signal import MLPSignal
    with patch("strategies.mlp_signal.MLPSignal._load_if_available"):
        model = MLPSignal(model_path="/tmp/x.pkl")
    with patch("data.db.get_connection", side_effect=RuntimeError("no db")):
        model._write_metadata(n_tickers=3, period="2y", train_ic=0.1, test_ic=0.05)


def test_predict_empty_feature_matrix_falls_back():
    with (
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
        patch("strategies.mlp_signal.build_feature_matrix", return_value=pd.DataFrame()),
        patch(
            "strategies.mlp_signal.MLPSignal._momentum_fallback",
            return_value={"AAPL": 0.1},
        ) as fb,
    ):
        from strategies.mlp_signal import MLPSignal
        model = MLPSignal(model_path="/tmp/x.pkl")
        model._model = MagicMock()
        out = model.predict(["AAPL"], period="6mo")
    fb.assert_called_once()
    assert out == {"AAPL": 0.1}


# ── End-to-end with real MLPRegressor ─────────────────────────────────────────

@skip_no_sklearn
def test_end_to_end_train_and_predict_with_real_mlp():
    fm = _make_feature_matrix(tickers=["AAPL", "MSFT", "GOOG"])
    with (
        patch("strategies.mlp_signal.build_feature_matrix", return_value=fm),
        patch("strategies.mlp_signal.MLPSignal._load_if_available"),
        patch("strategies.mlp_signal.MLPSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.mlp_signal.pickle.dump"),
        patch("strategies.mlp_signal.Path.mkdir"),
    ):
        from strategies.mlp_signal import MLPSignal
        model = MLPSignal(model_path="models/test_mlp.pkl")
        result = model.train(
            ["AAPL", "MSFT", "GOOG"], period="2y",
            hidden_layer_sizes=(8,), max_iter=20,
        )
        scores = model.predict(["AAPL", "MSFT", "GOOG"], period="6mo")
        df = model.feature_importances()

    assert result["n_train_samples"] > 0 and result["n_test_samples"] > 0
    assert set(scores) == {"AAPL", "MSFT", "GOOG"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0
    # Real first-layer coefs → real importances column ranked descending
    assert not df.empty
    assert df["importance"].tolist() == sorted(df["importance"].tolist(), reverse=True)
