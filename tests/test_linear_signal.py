"""
tests/test_linear_signal.py — Unit tests for strategies/linear_signal.py.

All fetch_ohlcv / build_feature_matrix calls are mocked — no network access.
sklearn-dependent tests are skipped when scikit-learn is not installed.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_feature_matrix(n_dates: int = 40, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        tickers = ["AAPL", "MSFT"]
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_21d",
        "skew_21d", "kurt_21d", "autocorr_1", "realised_vol_21d",
        "vol_ratio_20d", "vol_zscore_20d",
    ]
    rng = np.random.default_rng(42)
    rows = []
    idx = []
    for date in dates:
        for ticker in tickers:
            row = dict(zip(feature_cols, rng.standard_normal(len(feature_cols))))
            row["fwd_ret_5d"] = rng.standard_normal()
            rows.append(row)
            idx.append((date, ticker))

    mi = pd.MultiIndex.from_tuples(idx, names=["date", "ticker"])
    return pd.DataFrame(rows, index=mi)


# ── No-model tests (no sklearn required) ─────────────────────────────────────

def test_feature_coefficients_empty_without_model():
    from strategies.linear_signal import LinearSignal
    with patch("strategies.linear_signal.LinearSignal._load_if_available"):
        model = LinearSignal(model_path="/nonexistent/path.pkl")
    df = model.feature_coefficients()
    assert df.empty
    assert list(df.columns) == ["feature", "coefficient"]


def test_predict_falls_back_to_momentum_with_no_model():
    from strategies.linear_signal import LinearSignal
    with (
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch(
            "strategies.linear_signal.LinearSignal._momentum_fallback",
            return_value={"AAPL": 0.5},
        ) as mock_fb,
    ):
        model = LinearSignal(model_path="/nonexistent/path.pkl")
        result = model.predict(["AAPL"], period="6mo")

    mock_fb.assert_called_once()
    assert "AAPL" in result


def test_train_raises_when_sklearn_missing():
    from strategies.linear_signal import LinearSignal
    with (
        patch("strategies.linear_signal._SKLEARN_AVAILABLE", False),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
    ):
        model = LinearSignal(model_path="/tmp/x.pkl")
        with pytest.raises(RuntimeError, match="scikit-learn"):
            model.train(["AAPL"])


try:
    import sklearn as _sklearn_mod  # noqa: F401
    _SKLEARN_INSTALLED = True
except ImportError:
    _SKLEARN_INSTALLED = False

_skip_no_sklearn = pytest.mark.skipif(not _SKLEARN_INSTALLED, reason="scikit-learn not installed")


# ── Tests using mocked Ridge (no sklearn required) ────────────────────────────

def _make_mock_ridge(coefs: list[float] | None = None):
    """Return a MagicMock that quacks like a fitted Ridge model."""
    import numpy as np
    mock = MagicMock()
    mock.coef_ = np.array(coefs or [0.1, -0.2, 0.3, 0.0, 0.1, -0.1, 0.2, -0.05, 0.15, 0.05])
    mock.predict.side_effect = lambda X: np.random.default_rng(0).standard_normal(len(X))
    return mock


def test_train_with_mocked_ridge():
    fm = _make_feature_matrix()
    mock_ridge = _make_mock_ridge()

    with (
        patch("strategies.linear_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.linear_signal.Ridge", return_value=mock_ridge),
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="models/test_ridge.pkl")
        result = model.train(["AAPL", "MSFT"], period="2y")

    for key in ("train_ic", "test_ic", "n_train_samples", "n_test_samples"):
        assert key in result
    assert result["n_train_samples"] > 0


def test_predict_with_mocked_ridge_model():
    fm = _make_feature_matrix()
    mock_ridge = _make_mock_ridge()

    with (
        patch("strategies.linear_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.linear_signal.Ridge", return_value=mock_ridge),
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="models/test_ridge.pkl")
        model.train(["AAPL", "MSFT"], period="2y")
        scores = model.predict(["AAPL", "MSFT"], period="6mo")

    assert set(scores.keys()) == {"AAPL", "MSFT"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_feature_coefficients_with_mocked_model():
    fm = _make_feature_matrix()
    mock_ridge = _make_mock_ridge()

    with (
        patch("strategies.linear_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.linear_signal.Ridge", return_value=mock_ridge),
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="models/test_ridge.pkl")
        model.train(["AAPL", "MSFT"], period="2y")
        coef_df = model.feature_coefficients()

    assert list(coef_df.columns) == ["feature", "coefficient"]
    assert not coef_df.empty
    abs_vals = coef_df["coefficient"].abs().tolist()
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_write_metadata_silences_db_errors():
    with patch("strategies.linear_signal.LinearSignal._load_if_available"):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="/tmp/x.pkl")
    # Should not raise even if db is unavailable
    with patch("data.db.get_connection", side_effect=RuntimeError("no db")):
        model._write_metadata(n_tickers=5, period="2y", train_ic=0.1, test_ic=0.05)


def test_predict_empty_feature_matrix_falls_back():
    with (
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.build_feature_matrix", return_value=pd.DataFrame()),
        patch(
            "strategies.linear_signal.LinearSignal._momentum_fallback",
            return_value={"AAPL": 0.1},
        ) as mock_fb,
    ):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="/tmp/x.pkl")
        model._model = MagicMock()  # pretend model exists
        result = model.predict(["AAPL"], period="6mo")

    mock_fb.assert_called_once()
    assert "AAPL" in result


def test_load_if_available_success_path(tmp_path):
    import pickle as pk
    pkl_path = tmp_path / "ridge.pkl"
    # Pickle a plain dict as a stand-in for a model object
    with open(pkl_path, "wb") as f:
        pk.dump({"coef_": [0.1, 0.2]}, f)

    from strategies.linear_signal import LinearSignal
    model = LinearSignal(model_path=str(pkl_path))
    assert model._model is not None


def test_load_if_available_corrupt_file(tmp_path):
    bad_path = tmp_path / "bad.pkl"
    bad_path.write_bytes(b"not_a_pickle")

    from strategies.linear_signal import LinearSignal
    model = LinearSignal(model_path=str(bad_path))
    assert model._model is None


def test_train_empty_after_dropna():
    # Feature matrix where ALL fwd_ret_5d are NaN → empty after dropna
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_21d",
        "skew_21d", "kurt_21d", "autocorr_1", "realised_vol_21d",
        "vol_ratio_20d", "vol_zscore_20d",
    ]
    rows = [dict(zip(feature_cols, [0.0] * len(feature_cols)), fwd_ret_5d=float("nan"))
            for _ in dates]
    idx = pd.MultiIndex.from_tuples(
        [(d, "AAPL") for d in dates], names=["date", "ticker"]
    )
    fm = pd.DataFrame(rows, index=idx)

    with (
        patch("strategies.linear_signal._SKLEARN_AVAILABLE", True),
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
    ):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="/tmp/x.pkl")
        with pytest.raises(ValueError, match="non-NaN"):
            model.train(["AAPL"])


def test_momentum_fallback_returns_zero_on_missing_data():
    with patch("strategies.linear_signal.LinearSignal._load_if_available"):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="/tmp/x.pkl")

    # _momentum_fallback uses lazy imports; patch via the module they're imported from
    with (
        patch("data.fetcher.fetch_ohlcv", return_value=None),
    ):
        result = model._momentum_fallback(["AAPL"], "6mo")

    assert "AAPL" in result
    assert result["AAPL"] == 0.0


def test_write_metadata_success_path():
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("strategies.linear_signal.LinearSignal._load_if_available"):
        from strategies.linear_signal import LinearSignal
        model = LinearSignal(model_path="/tmp/x.pkl")

    # _write_metadata uses a lazy `from data.db import get_connection`; patch via data.db
    with patch("data.db.get_connection", return_value=mock_conn):
        model._write_metadata(n_tickers=3, period="2y", train_ic=0.1, test_ic=0.05)


# ── sklearn-required tests ────────────────────────────────────────────────────

@_skip_no_sklearn
def test_train_returns_ic_metrics():
    from strategies.linear_signal import LinearSignal
    fm = _make_feature_matrix()

    with (
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        model = LinearSignal(model_path="models/test_ridge.pkl")
        result = model.train(["AAPL", "MSFT"], period="2y")

    for key in ("train_ic", "test_ic", "train_icir", "test_icir", "n_train_samples", "n_test_samples"):
        assert key in result, f"Missing key: {key}"
    assert result["n_train_samples"] > 0
    assert result["n_test_samples"] > 0


@_skip_no_sklearn
def test_train_raises_on_empty_feature_matrix():
    from strategies.linear_signal import LinearSignal
    with (
        patch("strategies.linear_signal.build_feature_matrix", return_value=pd.DataFrame()),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
    ):
        model = LinearSignal(model_path="/tmp/test.pkl")
        with pytest.raises(ValueError, match="empty"):
            model.train(["AAPL"])


@_skip_no_sklearn
def test_feature_coefficients_sorted_by_abs():
    from strategies.linear_signal import LinearSignal
    fm = _make_feature_matrix()

    with (
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        model = LinearSignal(model_path="models/test_ridge.pkl")
        model.train(["AAPL", "MSFT"], period="2y")
        coef_df = model.feature_coefficients()

    assert list(coef_df.columns) == ["feature", "coefficient"]
    assert not coef_df.empty
    abs_coefs = coef_df["coefficient"].abs().tolist()
    assert abs_coefs == sorted(abs_coefs, reverse=True)


@_skip_no_sklearn
def test_predict_scores_in_range():
    from strategies.linear_signal import LinearSignal
    fm = _make_feature_matrix()

    with (
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        model = LinearSignal(model_path="models/test_ridge.pkl")
        model.train(["AAPL", "MSFT"], period="2y")
        scores = model.predict(["AAPL", "MSFT"], period="6mo")

    for ticker, score in scores.items():
        assert -1.0 <= score <= 1.0, f"Score for {ticker} out of range: {score}"


@_skip_no_sklearn
def test_predict_returns_all_tickers():
    from strategies.linear_signal import LinearSignal
    fm = _make_feature_matrix(tickers=["AAPL", "MSFT", "GOOG"])

    with (
        patch("strategies.linear_signal.build_feature_matrix", return_value=fm),
        patch("strategies.linear_signal.LinearSignal._load_if_available"),
        patch("strategies.linear_signal.LinearSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.linear_signal.pickle.dump"),
        patch("strategies.linear_signal.Path.mkdir"),
    ):
        model = LinearSignal(model_path="models/test_ridge.pkl")
        model.train(["AAPL", "MSFT", "GOOG"], period="2y")
        scores = model.predict(["AAPL", "MSFT", "GOOG"], period="6mo")

    assert set(scores.keys()) == {"AAPL", "MSFT", "GOOG"}
