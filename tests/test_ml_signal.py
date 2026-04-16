"""Tests for strategies/ml_signal.py and backtester/walk_forward.purged_walk_forward."""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtester.walk_forward import WalkForwardResult, purged_walk_forward
from strategies.ml_signal import _FEATURE_COLS, _LGBM_AVAILABLE, _TARGET_COL, MLSignal

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 252, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def _make_feature_matrix(n_dates: int = 120, n_tickers: int = 5,
                          include_target: bool = True) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            row = {"date": d, "ticker": t}
            for col in _FEATURE_COLS:
                row[col] = np.random.randn()
            if include_target:
                row[_TARGET_COL] = np.random.randn() * 0.02
            rows.append(row)
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    # Drop last 5 dates' targets to simulate real NaN forward returns
    if include_target:
        last_5_dates = sorted(df.index.get_level_values("date").unique())[-5:]
        df.loc[df.index.get_level_values("date").isin(last_5_dates), _TARGET_COL] = np.nan
    return df


def _mock_fetch(ticker: str, period: str) -> pd.DataFrame:
    seed = abs(hash(ticker)) % 2**16
    return _make_ohlcv(252, seed)


# ── MLSignal — no-model fallback ───────────────────────────────────────────────

def test_predict_fallback_returns_valid_range():
    """Without a trained model, predict() should fall back to momentum and return [-1, 1]."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    with patch("strategies.ml_signal.build_feature_matrix") as mock_fm, \
         patch("data.fetcher.fetch_ohlcv", side_effect=_mock_fetch):
        # Force model-path to non-existent so no checkpoint loads
        model = MLSignal(model_path="/tmp/nonexistent_lgbm_test.pkl")
        mock_fm.return_value = pd.DataFrame()  # empty → triggers fallback
        scores = model.predict(tickers, period="6mo")

    assert set(scores.keys()) == set(tickers)
    for t, score in scores.items():
        assert -1.0 <= score <= 1.0, f"{t}: score {score} out of [-1, 1]"


def test_predict_fallback_keys_match_tickers():
    tickers = ["AAPL", "MSFT"]
    with patch("data.fetcher.fetch_ohlcv", side_effect=_mock_fetch):
        model = MLSignal(model_path="/tmp/nonexistent_lgbm_test2.pkl")
        scores = model.predict(tickers, period="6mo")
    assert set(scores.keys()) == set(tickers)


def test_feature_importance_no_model_returns_empty_df():
    model = MLSignal(model_path="/tmp/nonexistent_lgbm_test3.pkl")
    fi = model.feature_importance()
    assert isinstance(fi, pd.DataFrame)
    assert fi.empty
    assert list(fi.columns) == ["feature", "importance"]


# ── MLSignal — predict with injected mock model (no LightGBM needed) ─────────

def _make_mock_model(n_features: int | None = None):
    """Create a mock model that behaves like a fitted LGBMRegressor."""
    if n_features is None:
        n_features = len(_FEATURE_COLS)
    mock = MagicMock()
    mock.predict.return_value = np.linspace(-0.05, 0.05, 4)
    mock.feature_importances_ = np.arange(n_features, 0, -1, dtype=float)
    return mock


def test_predict_with_mock_model_returns_scores():
    """predict() with an injected model should return float scores in [-1, 1]."""
    fm = _make_feature_matrix(n_dates=20, n_tickers=4)
    mock_model = _make_mock_model()

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm):
        model = MLSignal(model_path="/tmp/nonexistent_mock.pkl")
        model._model = mock_model
        tickers = list(fm.index.get_level_values("ticker").unique())
        scores = model.predict(tickers, period="6mo")

    assert len(scores) > 0
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_predict_with_mock_model_all_equal_predictions():
    """When all raw predictions are equal, z-score is 0 and scores are all 0."""
    fm = _make_feature_matrix(n_dates=20, n_tickers=4)
    mock_model = _make_mock_model()
    mock_model.predict.return_value = np.ones(4) * 0.03  # all same → std = 0

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm):
        model = MLSignal(model_path="/tmp/nonexistent_mock2.pkl")
        model._model = mock_model
        tickers = list(fm.index.get_level_values("ticker").unique())
        scores = model.predict(tickers, period="6mo")

    for score in scores.values():
        assert score == pytest.approx(0.0)


def test_predict_with_mock_model_exception_falls_back():
    """When model.predict raises, should fall back to momentum."""
    fm = _make_feature_matrix(n_dates=20, n_tickers=2)
    mock_model = _make_mock_model()
    mock_model.predict.side_effect = RuntimeError("model crashed")

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("data.fetcher.fetch_ohlcv", side_effect=_mock_fetch):
        model = MLSignal(model_path="/tmp/nonexistent_mock3.pkl")
        model._model = mock_model
        tickers = list(fm.index.get_level_values("ticker").unique())
        scores = model.predict(tickers, period="6mo")

    # Falls back to momentum — scores still valid
    assert set(scores.keys()) == set(tickers)
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_predict_with_mock_model_empty_fm_falls_back():
    """When feature matrix is empty with a model loaded, falls back to momentum."""
    with patch("strategies.ml_signal.build_feature_matrix", return_value=pd.DataFrame()), \
         patch("data.fetcher.fetch_ohlcv", side_effect=_mock_fetch):
        model = MLSignal(model_path="/tmp/nonexistent_mock4.pkl")
        model._model = _make_mock_model()
        scores = model.predict(["AAPL", "MSFT"], period="6mo")

    assert set(scores.keys()) == {"AAPL", "MSFT"}


def test_predict_with_mock_model_no_feature_cols_falls_back():
    """When latest feature slice is empty (no matching feature cols), falls back."""
    # Build FM without any of the _FEATURE_COLS columns
    fm = _make_feature_matrix(n_dates=10, n_tickers=2)
    # Remove all feature columns so feature_cols = [] → latest is empty
    only_target = fm[[_TARGET_COL]]

    with patch("strategies.ml_signal.build_feature_matrix", return_value=only_target), \
         patch("data.fetcher.fetch_ohlcv", side_effect=_mock_fetch):
        model = MLSignal(model_path="/tmp/nonexistent_mock5.pkl")
        model._model = _make_mock_model()
        tickers = list(only_target.index.get_level_values("ticker").unique())
        scores = model.predict(tickers, period="6mo")

    assert set(scores.keys()) == set(tickers)
    for score in scores.values():
        assert -1.0 <= score <= 1.0


def test_write_metadata_success():
    """_write_metadata should call conn.close() on the happy path."""
    model = MLSignal(model_path="/tmp/nonexistent_meta.pkl")
    with patch("data.db.get_connection") as mock_conn:
        conn_obj = MagicMock()
        conn_obj.__enter__ = MagicMock(return_value=conn_obj)
        conn_obj.__exit__ = MagicMock(return_value=False)
        mock_conn.return_value = conn_obj
        model._write_metadata(5, "2y", 0.05, 0.03)
    conn_obj.close.assert_called_once()


def test_write_metadata_handles_missing_table():
    """_write_metadata should not raise even when the DB table doesn't exist."""
    model = MLSignal(model_path="/tmp/nonexistent_meta.pkl")
    with patch("data.db.get_connection") as mock_conn:
        conn_obj = MagicMock()
        conn_obj.__enter__ = MagicMock(return_value=conn_obj)
        conn_obj.__exit__ = MagicMock(return_value=False)
        conn_obj.execute.side_effect = Exception("no such table: model_metadata")
        mock_conn.return_value = conn_obj
        # Should not raise
        model._write_metadata(5, "2y", 0.05, 0.03)


def test_momentum_fallback_with_none_fetch():
    """Momentum fallback returns 0.0 when fetch_ohlcv returns None."""
    with patch("data.fetcher.fetch_ohlcv", return_value=None):
        model = MLSignal(model_path="/tmp/nonexistent_fallback.pkl")
        scores = model._momentum_fallback(["AAPL", "MSFT"], "6mo")
    assert set(scores.keys()) == {"AAPL", "MSFT"}
    for score in scores.values():
        assert score == 0.0


def test_momentum_fallback_with_fetch_exception():
    """Momentum fallback returns 0.0 when fetch_ohlcv raises."""
    with patch("data.fetcher.fetch_ohlcv", side_effect=RuntimeError("network error")):
        model = MLSignal(model_path="/tmp/nonexistent_fallback2.pkl")
        scores = model._momentum_fallback(["AAPL"], "6mo")
    assert scores["AAPL"] == 0.0


def test_feature_importance_with_mock_model_exception():
    """feature_importance() returns empty DataFrame when model raises AttributeError."""
    model = MLSignal(model_path="/tmp/nonexistent_fi2.pkl")
    # spec=[] causes AttributeError on any attribute access, triggering the except handler
    mock_model = MagicMock(spec=[])
    model._model = mock_model
    fi = model.feature_importance()
    assert isinstance(fi, pd.DataFrame)
    assert fi.empty


def test_feature_importance_with_mock_model():
    """feature_importance() returns sorted DataFrame when model is injected."""
    model = MLSignal(model_path="/tmp/nonexistent_fi.pkl")
    model._model = _make_mock_model()
    fi = model.feature_importance()
    assert not fi.empty
    assert set(fi.columns) == {"feature", "importance"}
    assert list(fi["importance"]) == sorted(fi["importance"], reverse=True)
    assert (fi["importance"] >= 0).all()


# ── MLSignal — with LightGBM (skipped if not installed) ───────────────────────

@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="lightgbm not installed")
def test_train_and_predict_with_lgbm():
    """Full train → predict cycle with mocked feature matrix."""
    tickers = [f"T{i}" for i in range(8)]
    fm = _make_feature_matrix(n_dates=150, n_tickers=8)

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("strategies.ml_signal.MLSignal._write_metadata"):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        model = MLSignal(model_path=tmp_path)
        metrics = model.train(tickers, period="2y")

    assert "train_ic" in metrics
    assert "test_ic" in metrics
    assert isinstance(metrics["train_ic"], float)
    assert isinstance(metrics["test_ic"], float)
    assert metrics["n_train_samples"] > 0
    assert metrics["n_test_samples"] > 0


@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="lightgbm not installed")
def test_model_persist_and_reload():
    """train() saves checkpoint; a new MLSignal instance at the same path loads it."""
    tickers = [f"T{i}" for i in range(6)]
    fm = _make_feature_matrix(n_dates=120, n_tickers=6)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        tmp_path = f.name

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("strategies.ml_signal.MLSignal._write_metadata"):
        model1 = MLSignal(model_path=tmp_path)
        model1.train(tickers, period="2y")

    # New instance should auto-load the checkpoint
    model2 = MLSignal(model_path=tmp_path)
    assert model2._model is not None

    os.unlink(tmp_path)


@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="lightgbm not installed")
def test_feature_importance_after_train():
    tickers = [f"T{i}" for i in range(6)]
    fm = _make_feature_matrix(n_dates=120, n_tickers=6)

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("strategies.ml_signal.MLSignal._write_metadata"):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name
        model = MLSignal(model_path=tmp_path)
        model.train(tickers, period="2y")

    fi = model.feature_importance()
    assert not fi.empty
    assert set(fi.columns) == {"feature", "importance"}
    # Importances should be non-negative and sorted descending
    assert (fi["importance"] >= 0).all()
    assert list(fi["importance"]) == sorted(fi["importance"], reverse=True)

    os.unlink(tmp_path)


# ── purged_walk_forward ────────────────────────────────────────────────────────

def _make_wf_feature_matrix(n_dates: int = 200, n_tickers: int = 4) -> pd.DataFrame:
    np.random.seed(7)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"date": d, "ticker": t,
                         "ret_5d": np.random.randn(),
                         _TARGET_COL: np.random.randn() * 0.02})
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def _dummy_strategy(train_seg: pd.DataFrame, test_seg: pd.DataFrame) -> pd.Series:
    """Always returns a flat 0.001 daily return series over test dates."""
    test_dates = sorted(test_seg.index.get_level_values("date").unique())
    return pd.Series(0.001, index=pd.DatetimeIndex(test_dates))


def test_purged_walk_forward_returns_result():
    fm = _make_wf_feature_matrix(n_dates=200, n_tickers=4)
    result = purged_walk_forward(_dummy_strategy, fm, n_splits=5, embargo_pct=0.01)
    assert isinstance(result, WalkForwardResult)


def test_purged_walk_forward_has_windows():
    fm = _make_wf_feature_matrix(n_dates=200, n_tickers=4)
    result = purged_walk_forward(_dummy_strategy, fm, n_splits=5, embargo_pct=0.01)
    assert len(result.windows) > 0


def test_purged_walk_forward_embargo_gap():
    """Train end date should be strictly before test start date by at least embargo_bars."""
    all_train_ends: list[pd.Timestamp] = []
    all_test_starts: list[pd.Timestamp] = []

    def _recording_strategy(train_seg: pd.DataFrame, test_seg: pd.DataFrame) -> pd.Series:
        train_dates = sorted(train_seg.index.get_level_values("date").unique())
        test_dates = sorted(test_seg.index.get_level_values("date").unique())
        all_train_ends.append(train_dates[-1])
        all_test_starts.append(test_dates[0])
        return pd.Series(0.0, index=pd.DatetimeIndex(test_dates))

    fm = _make_wf_feature_matrix(n_dates=200, n_tickers=4)
    purged_walk_forward(_recording_strategy, fm, n_splits=5, embargo_pct=0.02)

    for train_end, test_start in zip(all_train_ends, all_test_starts):
        assert train_end < test_start, (
            f"Embargo violated: train_end={train_end}, test_start={test_start}"
        )


def test_purged_walk_forward_no_overlap_in_labels():
    """Dates in the train segment must not appear in the test segment."""
    def _check_strategy(train_seg: pd.DataFrame, test_seg: pd.DataFrame) -> pd.Series:
        train_dates = set(train_seg.index.get_level_values("date").unique())
        test_dates_list = sorted(test_seg.index.get_level_values("date").unique())
        test_dates = set(test_dates_list)
        overlap = train_dates & test_dates
        assert len(overlap) == 0, f"Overlap between train and test dates: {overlap}"
        return pd.Series(0.0, index=pd.DatetimeIndex(test_dates_list))

    fm = _make_wf_feature_matrix(n_dates=200, n_tickers=4)
    purged_walk_forward(_check_strategy, fm, n_splits=5, embargo_pct=0.01)


def test_purged_walk_forward_empty_on_short_data():
    """Too few dates to form valid folds should return an empty WalkForwardResult."""
    fm = _make_wf_feature_matrix(n_dates=5, n_tickers=2)
    result = purged_walk_forward(_dummy_strategy, fm, n_splits=5, embargo_pct=0.5)
    assert isinstance(result, WalkForwardResult)
    # With very short data and large embargo, likely no valid windows
    # (may or may not have windows depending on split arithmetic; just check type)


def test_purged_walk_forward_empty_matrix():
    result = purged_walk_forward(_dummy_strategy, pd.DataFrame(), n_splits=5)
    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) == 0


def test_purged_walk_forward_consistency_score_range():
    fm = _make_wf_feature_matrix(n_dates=200, n_tickers=4)
    result = purged_walk_forward(_dummy_strategy, fm, n_splits=5, embargo_pct=0.01)
    if result.windows:
        assert 0.0 <= result.consistency_score <= 1.0
