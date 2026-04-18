"""Tests for strategies/rf_long_short.py — Random-Forest long-short alpha."""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


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
    mi = pd.MultiIndex.from_tuples(idx, names=["date", "ticker"])
    return pd.DataFrame(rows, index=mi)


def _make_mock_rf(n_features: int = 12):
    mock = MagicMock()
    mock.feature_importances_ = np.linspace(0.3, 0.01, n_features)
    mock.predict.side_effect = lambda X: np.random.default_rng(0).standard_normal(len(X))
    return mock


# ── No-model tests ────────────────────────────────────────────────────────────

def test_feature_importances_empty_without_model():
    from strategies.rf_long_short import RFLongShortSignal
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        model = RFLongShortSignal(model_path="/nonexistent.pkl")
    df = model.feature_importances()
    assert df.empty
    assert list(df.columns) == ["feature", "importance"]


def test_predict_no_model_falls_back_to_momentum():
    from strategies.rf_long_short import RFLongShortSignal
    with (
        patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"),
        patch(
            "strategies.rf_long_short.RFLongShortSignal._momentum_fallback",
            return_value={"AAPL": 0.5},
        ) as mock_fb,
    ):
        model = RFLongShortSignal(model_path="/nonexistent.pkl")
        result = model.predict(["AAPL"], period="6mo")
    mock_fb.assert_called_once()
    assert result == {"AAPL": 0.5}


def test_train_raises_when_sklearn_missing():
    from strategies.rf_long_short import RFLongShortSignal
    with (
        patch("strategies.rf_long_short._SKLEARN_AVAILABLE", False),
        patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"),
    ):
        model = RFLongShortSignal(model_path="/tmp/x.pkl")
        with pytest.raises(RuntimeError, match="scikit-learn"):
            model.train(["AAPL"])


# ── Tests using mocked RF (no sklearn required at runtime) ────────────────────

def test_train_returns_metrics_with_mocked_rf():
    fm = _make_feature_matrix()
    mock_rf = _make_mock_rf()
    with (
        patch("strategies.rf_long_short._SKLEARN_AVAILABLE", True),
        patch("strategies.rf_long_short.RandomForestRegressor", return_value=mock_rf),
        patch("strategies.rf_long_short.build_feature_matrix", return_value=fm),
        patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"),
        patch("strategies.rf_long_short.RFLongShortSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.rf_long_short.pickle.dump"),
        patch("strategies.rf_long_short.Path.mkdir"),
    ):
        from strategies.rf_long_short import RFLongShortSignal
        model = RFLongShortSignal(model_path="models/test_rf.pkl")
        result = model.train(["AAPL", "MSFT"], period="2y")

    for key in ("train_ic", "test_ic", "train_icir", "test_icir",
                "n_train_samples", "n_test_samples"):
        assert key in result
    assert result["n_train_samples"] > 0
    assert result["n_test_samples"] > 0


def test_predict_scores_in_range_with_mocked_rf():
    fm = _make_feature_matrix(tickers=["AAPL", "MSFT", "GOOG"])
    mock_rf = _make_mock_rf()
    with (
        patch("strategies.rf_long_short._SKLEARN_AVAILABLE", True),
        patch("strategies.rf_long_short.RandomForestRegressor", return_value=mock_rf),
        patch("strategies.rf_long_short.build_feature_matrix", return_value=fm),
        patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"),
        patch("strategies.rf_long_short.RFLongShortSignal._write_metadata"),
        patch("builtins.open", MagicMock()),
        patch("strategies.rf_long_short.pickle.dump"),
        patch("strategies.rf_long_short.Path.mkdir"),
    ):
        from strategies.rf_long_short import RFLongShortSignal
        model = RFLongShortSignal(model_path="models/test_rf.pkl")
        model.train(["AAPL", "MSFT", "GOOG"], period="2y")
        scores = model.predict(["AAPL", "MSFT", "GOOG"], period="6mo")

    assert set(scores.keys()) == {"AAPL", "MSFT", "GOOG"}
    for v in scores.values():
        assert -1.0 <= v <= 1.0


def test_feature_importances_sorted_descending():
    mock_rf = _make_mock_rf()
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        from strategies.rf_long_short import RFLongShortSignal
        model = RFLongShortSignal(model_path="/tmp/x.pkl")
        model._model = mock_rf

    df = model.feature_importances()
    assert list(df.columns) == ["feature", "importance"]
    imps = df["importance"].tolist()
    assert imps == sorted(imps, reverse=True)


# ── long_short_portfolio ──────────────────────────────────────────────────────

def test_long_short_portfolio_dollar_neutral_top_quintile():
    from strategies.rf_long_short import RFLongShortSignal
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        model = RFLongShortSignal(model_path="/tmp/x.pkl")

    # 10 tickers with monotonically increasing scores
    scores = {f"T{i}": float(i) for i in range(10)}
    with patch.object(model, "predict", return_value=scores):
        weights = model.long_short_portfolio(list(scores), top_pct=0.2)

    # 2 longs (top), 2 shorts (bottom), 6 flat
    longs = [t for t, w in weights.items() if w == 1]
    shorts = [t for t, w in weights.items() if w == -1]
    flats = [t for t, w in weights.items() if w == 0]
    assert sorted(longs) == ["T8", "T9"]
    assert sorted(shorts) == ["T0", "T1"]
    assert len(flats) == 6


def test_long_short_portfolio_empty_scores_returns_zero_weights():
    from strategies.rf_long_short import RFLongShortSignal
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        model = RFLongShortSignal(model_path="/tmp/x.pkl")

    with patch.object(model, "predict", return_value={}):
        weights = model.long_short_portfolio(["AAPL", "MSFT"], top_pct=0.2)
    assert weights == {"AAPL": 0, "MSFT": 0}


def test_long_short_portfolio_top_pct_clipped_to_half():
    """top_pct > 0.5 should not split a 10-ticker universe into 11 picks each side."""
    from strategies.rf_long_short import RFLongShortSignal
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        model = RFLongShortSignal(model_path="/tmp/x.pkl")

    scores = {f"T{i}": float(i) for i in range(10)}
    with patch.object(model, "predict", return_value=scores):
        weights = model.long_short_portfolio(list(scores), top_pct=0.99)

    longs = sum(1 for w in weights.values() if w == 1)
    shorts = sum(1 for w in weights.values() if w == -1)
    assert longs <= 5 and shorts <= 5


# ── Persistence ───────────────────────────────────────────────────────────────

def test_load_if_available_corrupt_file_does_not_raise(tmp_path):
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(b"not_a_pickle")
    from strategies.rf_long_short import RFLongShortSignal
    model = RFLongShortSignal(model_path=str(bad))
    assert model._model is None


def test_write_metadata_silences_db_errors():
    from strategies.rf_long_short import RFLongShortSignal
    with patch("strategies.rf_long_short.RFLongShortSignal._load_if_available"):
        model = RFLongShortSignal(model_path="/tmp/x.pkl")
    with patch("data.db.get_connection", side_effect=RuntimeError("no db")):
        # Must not raise
        model._write_metadata(n_tickers=3, period="2y", train_ic=0.1, test_ic=0.05)
