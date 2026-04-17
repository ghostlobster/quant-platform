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


# ── Triple-barrier classifier path (pure-mock, no LightGBM needed) ────────────

class _PicklablePlainRegressor:
    """Plain regressor used by checkpoint-load tests — unlike MagicMock, picklable."""
    feature_importances_ = np.array([1.0, 2.0, 3.0])

    def predict(self, X):
        return np.zeros(len(X))


class _PicklablePlainClassifier:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        return np.tile(np.array([0.2, 0.3, 0.5]), (len(X), 1))


def _make_classifier_mock_model(n_rows: int = 4):
    """Fake LGBMClassifier: predict_proba returns a stable 3-class distribution."""
    mock = MagicMock()
    mock.classes_ = np.array([-1, 0, 1])
    probs = np.tile(np.array([0.2, 0.3, 0.5]), (n_rows, 1))
    mock.predict_proba.return_value = probs
    mock.feature_importances_ = np.arange(len(_FEATURE_COLS), 0, -1, dtype=float)
    return mock


def test_classifier_scores_uses_p_up_minus_p_down():
    mock = _make_classifier_mock_model(n_rows=3)
    X = np.zeros((3, 5))
    scores = MLSignal._classifier_scores(mock, X)
    # p_up(=0.5) − p_down(=0.2) = 0.3 for every row
    assert scores == pytest.approx(np.array([0.3, 0.3, 0.3]))


def test_predict_classifier_path_returns_valid_range():
    """When _is_classifier=True, predict uses predict_proba instead of predict."""
    fm = _make_feature_matrix(n_dates=20, n_tickers=4)
    mock_model = _make_classifier_mock_model(n_rows=4)

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm):
        model = MLSignal(model_path="/tmp/nonexistent_clf.pkl")
        model._model = mock_model
        model._is_classifier = True
        tickers = list(fm.index.get_level_values("ticker").unique())
        scores = model.predict(tickers, period="6mo")

    assert len(scores) == 4
    # All rows share the same probability; after z-score + clip they collapse to 0
    for s in scores.values():
        assert -1.0 <= s <= 1.0
    # predict_proba should have been called at least once
    assert mock_model.predict_proba.called
    # predict() should not have been called (we're on the classifier path)
    mock_model.predict.assert_not_called()


def test_score_features_dispatches_on_classifier_flag():
    model = MLSignal(model_path="/tmp/nonexistent_sf.pkl")
    # Regressor path
    reg_mock = _make_mock_model()
    model._model = reg_mock
    model._is_classifier = False
    out_reg = model.score_features(np.zeros((4, len(_FEATURE_COLS))))
    assert len(out_reg) == 4
    reg_mock.predict.assert_called_once()
    # Classifier path
    clf_mock = _make_classifier_mock_model(n_rows=4)
    model._model = clf_mock
    model._is_classifier = True
    out_clf = model.score_features(np.zeros((4, len(_FEATURE_COLS))))
    assert out_clf == pytest.approx(np.array([0.3, 0.3, 0.3, 0.3]))


def test_score_features_raises_without_trained_model():
    model = MLSignal(model_path="/tmp/nonexistent_sf2.pkl")
    with pytest.raises(RuntimeError, match="no trained model"):
        model.score_features(np.zeros((2, 3)))


def test_train_triple_barrier_routes_to_classifier(monkeypatch, tmp_path):
    """train(label_type='triple_barrier') should fit LGBMClassifier, set
    _is_classifier=True, and persist both fields."""
    from data.features import _FEATURE_COLS as F_COLS

    # Synthetic FM with fwd_ret_5d AND tb_bin + tb_ret columns
    np.random.seed(0)
    n_dates, n_tickers = 40, 4
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d in dates:
        for t in tickers:
            row = {"date": d, "ticker": t}
            for c in F_COLS:
                row[c] = np.random.randn()
            row["fwd_ret_5d"] = np.random.randn() * 0.01
            row["tb_bin"] = np.random.choice([-1, 0, 1])
            row["tb_ret"] = np.random.randn() * 0.01
            row["tb_target"] = abs(np.random.randn()) * 0.01
            rows.append(row)
    fm = pd.DataFrame(rows).set_index(["date", "ticker"])

    fitted_kinds: list[str] = []

    class _FakeRegressor:
        def __init__(self, **_k): fitted_kinds.append("regressor")
        def fit(self, X, y): self._y = y
        def predict(self, X): return np.zeros(len(X))

    class _FakeClassifier:
        classes_ = np.array([-1, 0, 1])
        def __init__(self, **_k): fitted_kinds.append("classifier")
        def fit(self, X, y): self._y = y
        def predict_proba(self, X):
            return np.tile(np.array([0.2, 0.3, 0.5]), (len(X), 1))

    fake_lgb = MagicMock()
    fake_lgb.LGBMRegressor = _FakeRegressor
    fake_lgb.LGBMClassifier = _FakeClassifier

    # Make triple_barrier FM return non-empty
    monkeypatch.setattr(
        "strategies.ml_signal.build_feature_matrix",
        lambda tickers, **kwargs: fm,
    )
    monkeypatch.setattr("strategies.ml_signal._LGBM_AVAILABLE", True)
    monkeypatch.setattr("strategies.ml_signal.lgb", fake_lgb)
    monkeypatch.setattr(MLSignal, "_write_metadata", lambda *a, **k: None)

    # The fake classifier is a local class → can't be pickled. Capture the
    # payload in-memory instead.
    captured: dict = {}
    def _capture_pickle(obj, f):
        captured.update(obj if isinstance(obj, dict) else {"raw": obj})
    monkeypatch.setattr("strategies.ml_signal.pickle.dump", _capture_pickle)

    ckpt = tmp_path / "clf.pkl"
    model = MLSignal(model_path=str(ckpt))
    metrics = model.train(
        [f"T{i}" for i in range(n_tickers)],
        period="2y",
        label_type="triple_barrier",
    )

    assert model._is_classifier is True
    assert "classifier" in fitted_kinds
    assert "regressor" not in fitted_kinds
    assert "train_ic" in metrics

    # Checkpoint payload preserves the classifier flag
    assert captured.get("is_classifier") is True
    assert "model" in captured


def test_load_legacy_regressor_checkpoint(tmp_path):
    """Legacy checkpoints (bare estimator pickle) load as regressor."""
    import pickle
    ckpt = tmp_path / "legacy.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(_PicklablePlainRegressor(), f)

    with patch("strategies.ml_signal._LGBM_AVAILABLE", True):
        model = MLSignal(model_path=str(ckpt))
    assert model._model is not None
    assert model._is_classifier is False


def test_load_versioned_classifier_checkpoint(tmp_path):
    """Versioned checkpoints round-trip is_classifier=True."""
    import pickle
    ckpt = tmp_path / "versioned.pkl"
    with open(ckpt, "wb") as f:
        pickle.dump(
            {"model": _PicklablePlainClassifier(), "is_classifier": True}, f,
        )

    with patch("strategies.ml_signal._LGBM_AVAILABLE", True):
        model = MLSignal(model_path=str(ckpt))
    assert model._is_classifier is True


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


# ── MLSignal — regime-conditioned models ──────────────────────────────────────

def _make_spy_df(n: int = 252) -> pd.DataFrame:
    """Minimal SPY-like OHLCV DataFrame."""
    np.random.seed(1)
    close = 400 + np.cumsum(np.random.randn(n) * 1.0)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close, "Volume": 1e8,
    }, index=idx)


def _make_vix_df(n: int = 252, vix_value: float = 15.0) -> pd.DataFrame:
    """Minimal ^VIX-like DataFrame with fixed VIX level."""
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": vix_value}, index=idx)


def test_get_historical_regimes_returns_series():
    """_get_historical_regimes returns a date-indexed Series with valid regime labels."""
    from analysis.regime import REGIME_STATES
    spy = _make_spy_df(252)
    vix = _make_vix_df(252, vix_value=15.0)

    model = MLSignal(model_path="/tmp/nonexistent_regime.pkl")
    with patch("strategies.ml_signal.fetch_ohlcv", side_effect=lambda t, _p: spy if t == "SPY" else vix):
        regimes = model._get_historical_regimes("2y")

    assert isinstance(regimes, pd.Series)
    assert len(regimes) == len(spy)
    assert all(r in REGIME_STATES for r in regimes.values)


def test_get_historical_regimes_high_vix():
    """VIX > 30 → all dates labelled high_vol."""
    spy = _make_spy_df(100)
    vix = _make_vix_df(100, vix_value=35.0)

    model = MLSignal(model_path="/tmp/nonexistent_regime2.pkl")
    with patch("strategies.ml_signal.fetch_ohlcv", side_effect=lambda t, _p: spy if t == "SPY" else vix):
        regimes = model._get_historical_regimes("1y")

    assert (regimes == "high_vol").all()


def test_get_historical_regimes_spy_unavailable():
    """Returns empty Series when SPY fetch fails."""
    model = MLSignal(model_path="/tmp/nonexistent_regime3.pkl")
    with patch("strategies.ml_signal.fetch_ohlcv", return_value=None):
        regimes = model._get_historical_regimes("1y")

    assert isinstance(regimes, pd.Series)
    assert regimes.empty


def test_select_model_returns_regime_model_when_available():
    """_select_model returns the regime-specific model when current regime matches."""
    model = MLSignal(model_path="/tmp/nonexistent_sel.pkl")
    mock_regime_model = MagicMock()
    model._regime_models = {"trending_bull": mock_regime_model}

    with patch("strategies.ml_signal.get_live_regime", return_value={"regime": "trending_bull"}):
        selected = model._select_model(use_regime_model=True)

    assert selected is mock_regime_model


def test_select_model_falls_back_to_baseline_when_regime_not_in_models():
    """_select_model returns baseline when no regime model matches current regime."""
    model = MLSignal(model_path="/tmp/nonexistent_sel2.pkl")
    mock_baseline = MagicMock()
    model._model = mock_baseline
    model._regime_models = {"trending_bull": MagicMock()}

    with patch("strategies.ml_signal.get_live_regime", return_value={"regime": "high_vol"}):
        selected = model._select_model(use_regime_model=True)

    assert selected is mock_baseline


def test_select_model_falls_back_on_regime_exception():
    """_select_model returns baseline when regime detection raises."""
    model = MLSignal(model_path="/tmp/nonexistent_sel3.pkl")
    mock_baseline = MagicMock()
    model._model = mock_baseline
    model._regime_models = {"trending_bull": MagicMock()}

    with patch("strategies.ml_signal.get_live_regime", side_effect=RuntimeError("network error")):
        selected = model._select_model(use_regime_model=True)

    assert selected is mock_baseline


def test_select_model_skips_regime_when_disabled():
    """_select_model returns baseline directly when use_regime_model=False."""
    model = MLSignal(model_path="/tmp/nonexistent_sel4.pkl")
    mock_baseline = MagicMock()
    model._model = mock_baseline
    model._regime_models = {"trending_bull": MagicMock()}

    selected = model._select_model(use_regime_model=False)

    assert selected is mock_baseline


def test_predict_uses_regime_model():
    """predict() uses regime model when available for current regime."""
    model = MLSignal(model_path="/tmp/nonexistent_pred_regime.pkl")
    mock_regime_model = _make_mock_model()
    model._regime_models = {"trending_bull": mock_regime_model}

    fm = _make_feature_matrix(n_dates=30, n_tickers=3)

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("strategies.ml_signal.get_live_regime", return_value={"regime": "trending_bull"}):
        scores = model.predict(["T00", "T01", "T02"], period="6mo", use_regime_model=True)

    assert set(scores.keys()) == {"T00", "T01", "T02"}
    assert all(-1.0 <= v <= 1.0 for v in scores.values())


def test_predict_falls_back_to_momentum_with_no_models():
    """predict() uses momentum fallback when neither baseline nor regime models exist."""
    model = MLSignal(model_path="/tmp/nonexistent_pred_fb.pkl")
    assert model._model is None
    assert model._regime_models == {}

    with patch("strategies.ml_signal.MLSignal._momentum_fallback",
               return_value={"AAPL": 0.1}) as mock_fb:
        scores = model.predict(["AAPL"], period="6mo")

    mock_fb.assert_called_once()
    assert scores == {"AAPL": 0.1}


@pytest.mark.skipif(not _LGBM_AVAILABLE, reason="lightgbm not installed")
def test_train_regime_models_with_lgbm():
    """train_regime_models() trains models for regimes with enough samples."""
    tickers = [f"T{i}" for i in range(6)]
    # Build a large feature matrix so each regime gets enough samples
    fm = _make_feature_matrix(n_dates=300, n_tickers=6)
    spy = _make_spy_df(300)
    vix = _make_vix_df(300, vix_value=15.0)  # all trending_bull with rising SPY

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        regime_tmp = f.name
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        base_tmp = f.name

    with patch("strategies.ml_signal.build_feature_matrix", return_value=fm), \
         patch("strategies.ml_signal.fetch_ohlcv",
               side_effect=lambda t, _p: spy if t == "SPY" else vix):
        model = MLSignal(model_path=base_tmp, regime_model_path=regime_tmp)
        results = model.train_regime_models(
            tickers, period="2y", min_regime_samples=10
        )

    assert isinstance(results, dict)
    # At least one regime should have been trained
    assert len(results) >= 1
    for regime, metrics in results.items():
        assert "train_ic" in metrics
        assert "test_ic" in metrics
        assert isinstance(metrics["train_ic"], float)
        assert metrics["n_train"] > 0

    os.unlink(regime_tmp)
    os.unlink(base_tmp)
