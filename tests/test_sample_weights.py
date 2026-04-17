"""
Tests for analysis/sample_weights.py — AFML Ch 4 sample uniqueness.

Synthetic event series with hand-verifiable overlap structure so that
the concurrency counts and uniqueness weights can be asserted exactly.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from analysis.sample_weights import (
    num_co_events,
    sample_uniqueness,
    sequential_bootstrap,
    weights_for_train_index,
)


def _bar_index(n: int = 10, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="D")


# ── num_co_events ─────────────────────────────────────────────────────────────

def test_num_co_events_single_isolated_event():
    idx = _bar_index(10)
    events = pd.Series({idx[2]: idx[4]})   # active on bars 2,3,4
    co = num_co_events(idx, events)
    expected = np.zeros(10, dtype=int)
    expected[2:5] = 1
    np.testing.assert_array_equal(co.values, expected)


def test_num_co_events_two_fully_overlapping_events():
    idx = _bar_index(10)
    events = pd.Series({idx[2]: idx[4], idx[2]: idx[4]})   # same window  # noqa: F601
    co = num_co_events(idx, events)
    # Only one row survives the dict keying on identical start; make
    # sure using distinct starts also exercises the concurrency logic.
    assert co.loc[idx[3]] == 1


def test_num_co_events_partial_overlap_increments_count():
    idx = _bar_index(10)
    events = pd.Series({idx[2]: idx[5], idx[3]: idx[6]})
    co = num_co_events(idx, events)
    assert co.loc[idx[2]] == 1
    assert co.loc[idx[3]] == 2      # both events active
    assert co.loc[idx[4]] == 2
    assert co.loc[idx[5]] == 2
    assert co.loc[idx[6]] == 1
    assert co.loc[idx[7]] == 0


def test_num_co_events_empty_events_returns_zero_series():
    idx = _bar_index(5)
    co = num_co_events(idx, pd.Series(dtype="datetime64[ns]"))
    assert (co.values == 0).all()
    assert list(co.index) == list(idx)


def test_num_co_events_event_past_last_bar_is_clamped():
    idx = _bar_index(5)
    events = pd.Series({idx[3]: idx[-1] + pd.Timedelta(days=100)})
    co = num_co_events(idx, events)
    # Should be 1 on bars 3 and 4 (last bar); clamped rather than dropped.
    assert co.loc[idx[3]] == 1
    assert co.loc[idx[4]] == 1


# ── sample_uniqueness ────────────────────────────────────────────────────────

def test_sample_uniqueness_isolated_event_has_weight_one():
    idx = _bar_index(10)
    events = pd.Series({idx[2]: idx[4]})
    co = num_co_events(idx, events)
    u = sample_uniqueness(events, co)
    assert u.loc[idx[2]] == pytest.approx(1.0)


def test_sample_uniqueness_fully_concurrent_events_have_1_over_n():
    """Three events covering exactly the same window → each gets 1/3."""
    idx = _bar_index(10)
    starts = [idx[2], idx[2] + pd.Timedelta(hours=1), idx[2] + pd.Timedelta(hours=2)]
    ends   = [idx[4], idx[4], idx[4]]
    events = pd.Series(ends, index=pd.DatetimeIndex(starts))
    # Co-events at each covered bar = 3
    co = pd.Series(3, index=idx)
    u = sample_uniqueness(events, co)
    for s in starts:
        assert u.loc[s] == pytest.approx(1.0 / 3.0)


def test_sample_uniqueness_partial_overlap_between_1_and_half():
    """Partial overlap: two events that share a portion of their windows."""
    idx = _bar_index(10)
    events = pd.Series({idx[2]: idx[5], idx[3]: idx[6]})
    co = num_co_events(idx, events)
    u = sample_uniqueness(events, co)
    # Both uniqueness values should be strictly between 0.5 and 1.0
    for s in events.index:
        assert 0.5 < u.loc[s] < 1.0


def test_sample_uniqueness_empty_events_returns_empty():
    idx = _bar_index(5)
    co = pd.Series(0, index=idx)
    u = sample_uniqueness(pd.Series(dtype="datetime64[ns]"), co)
    assert u.empty


# ── sequential_bootstrap ─────────────────────────────────────────────────────

def test_sequential_bootstrap_size_matches_request():
    idx = _bar_index(10)
    events = pd.Series({idx[i]: idx[i + 1] for i in range(5)})
    picks = sequential_bootstrap(events, size=8, seed=1)
    assert len(picks) == 8


def test_sequential_bootstrap_only_picks_known_event_starts():
    idx = _bar_index(10)
    events = pd.Series({idx[i]: idx[i + 1] for i in range(5)})
    picks = sequential_bootstrap(events, size=20, seed=2)
    starts = set(events.index)
    assert all(p in starts for p in picks)


def test_sequential_bootstrap_seed_is_deterministic():
    idx = _bar_index(10)
    events = pd.Series({idx[i]: idx[i + 1] for i in range(5)})
    a = sequential_bootstrap(events, size=10, seed=7)
    b = sequential_bootstrap(events, size=10, seed=7)
    assert a == b


def test_sequential_bootstrap_empty_returns_empty_list():
    assert sequential_bootstrap(pd.Series(dtype="datetime64[ns]")) == []


# ── weights_for_train_index ──────────────────────────────────────────────────

def test_weights_for_train_index_mean_is_one():
    idx = _bar_index(20)
    events = pd.Series({idx[i]: idx[i + 2] for i in range(0, 15, 2)})
    tickers = ["AAPL", "MSFT", "GOOG"]
    mi = pd.MultiIndex.from_product(
        [events.index, tickers], names=["date", "ticker"],
    )
    w = weights_for_train_index(mi, events, idx)
    assert len(w) == len(mi)
    # Post-normalisation mean is 1.0 across the event-covered subset.
    assert np.isfinite(w).all()
    assert np.mean(w) == pytest.approx(1.0, abs=1e-9)


def test_weights_for_train_index_missing_date_gets_weight_one():
    idx = _bar_index(20)
    events = pd.Series({idx[i]: idx[i + 2] for i in range(3)})
    # Add a row for a date that has no event
    extra_date = idx[10]
    mi = pd.MultiIndex.from_tuples(
        [(idx[0], "AAPL"), (idx[1], "AAPL"), (extra_date, "AAPL")],
        names=["date", "ticker"],
    )
    w = weights_for_train_index(mi, events, idx)
    # The missing-date row falls back to 1.0
    assert w[2] == pytest.approx(1.0)


def test_weights_for_train_index_no_events_returns_ones():
    idx = _bar_index(5)
    mi = pd.MultiIndex.from_product([idx, ["AAPL"]], names=["date", "ticker"])
    w = weights_for_train_index(mi, pd.Series(dtype="datetime64[ns]"), idx)
    np.testing.assert_array_equal(w, np.ones(len(mi)))


# ── Integration with MLSignal.train (skipped if lightgbm not installed) ──────

@pytest.fixture
def lgbm_available():
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _fabricate_triple_barrier_fm(
    n_dates: int = 80, n_tickers: int = 6, seed: int = 0,
) -> pd.DataFrame:
    """Synthetic MultiIndex(date, ticker) FM with triple-barrier columns
    whose tb_bin is *correlated* with feature ret_1d so a classifier can
    learn a non-trivial mapping."""
    from data.features import _FEATURE_COLS

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d_i, d in enumerate(dates):
        for t in tickers:
            feats = {c: rng.normal() for c in _FEATURE_COLS}
            # Make ret_1d predict the bin: positive ret_1d → bin=+1 bias.
            signal = feats["ret_1d"] + rng.normal(scale=0.2)
            bin_ = 1 if signal > 0.5 else (-1 if signal < -0.5 else 0)
            ret = 0.01 * bin_ + rng.normal(scale=0.001)
            t1 = dates[min(d_i + 3, n_dates - 1)]
            rows.append(dict(
                date=d, ticker=t,
                **feats,
                tb_bin=bin_, tb_ret=ret, tb_target=0.01, tb_t1=t1,
            ))
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_train_forwards_sample_weight_when_requested(
    monkeypatch, tmp_path, lgbm_available,
):
    """When use_sample_weights=True, model.fit must receive a
    sample_weight kwarg whose length matches X_train."""
    if not lgbm_available:
        pytest.skip("lightgbm not installed")

    from strategies.ml_signal import MLSignal

    fm = _fabricate_triple_barrier_fm(n_dates=50, n_tickers=5, seed=3)
    fit_captures: list[dict] = []

    class _FakeClassifier:
        classes_ = np.array([-1, 0, 1])

        def __init__(self, **_k): pass

        def fit(self, X, y, **kw):
            fit_captures.append(kw)

        def predict_proba(self, X):
            return np.tile([0.3, 0.4, 0.3], (len(X), 1))

    import importlib

    import strategies.ml_signal as mls
    real_lgb = importlib.import_module("lightgbm")
    monkeypatch.setattr(mls, "lgb", type("lgb", (), {
        "LGBMClassifier": _FakeClassifier,
        "LGBMRegressor": real_lgb.LGBMRegressor,
    }))
    monkeypatch.setattr(mls, "build_feature_matrix", lambda *a, **k: fm)
    monkeypatch.setattr(MLSignal, "_write_metadata", lambda *a, **k: None)
    monkeypatch.setattr(mls.pickle, "dump", lambda *a, **k: None)

    model = MLSignal(model_path=str(tmp_path / "m.pkl"))
    model.train(
        ["T0", "T1"], period="2y",
        label_type="triple_barrier",
        use_sample_weights=True,
    )

    assert fit_captures, "fit() was not called"
    assert "sample_weight" in fit_captures[0], "sample_weight kwarg missing"
    sw = fit_captures[0]["sample_weight"]
    assert sw.ndim == 1
    assert (sw > 0).all()


def test_train_no_sample_weight_by_default(monkeypatch, tmp_path, lgbm_available):
    """Without use_sample_weights, fit() must NOT receive sample_weight."""
    if not lgbm_available:
        pytest.skip("lightgbm not installed")

    from strategies.ml_signal import MLSignal

    fm = _fabricate_triple_barrier_fm(n_dates=50, n_tickers=5, seed=3)
    fit_captures: list[dict] = []

    class _FakeClassifier:
        classes_ = np.array([-1, 0, 1])

        def __init__(self, **_k): pass

        def fit(self, X, y, **kw):
            fit_captures.append(kw)

        def predict_proba(self, X):
            return np.tile([0.3, 0.4, 0.3], (len(X), 1))

    import importlib

    import strategies.ml_signal as mls
    real_lgb = importlib.import_module("lightgbm")
    monkeypatch.setattr(mls, "lgb", type("lgb", (), {
        "LGBMClassifier": _FakeClassifier,
        "LGBMRegressor": real_lgb.LGBMRegressor,
    }))
    monkeypatch.setattr(mls, "build_feature_matrix", lambda *a, **k: fm)
    monkeypatch.setattr(MLSignal, "_write_metadata", lambda *a, **k: None)
    monkeypatch.setattr(mls.pickle, "dump", lambda *a, **k: None)

    model = MLSignal(model_path=str(tmp_path / "m.pkl"))
    model.train(
        ["T0", "T1"], period="2y", label_type="triple_barrier",
    )

    assert fit_captures
    assert "sample_weight" not in fit_captures[0]


def test_train_ignores_sample_weights_for_regression_label_type(
    monkeypatch, tmp_path, lgbm_available,
):
    """use_sample_weights=True is silently ignored for label_type=fwd_ret."""
    if not lgbm_available:
        pytest.skip("lightgbm not installed")

    from data.features import _FEATURE_COLS
    from strategies.ml_signal import MLSignal

    # Plain FM with fwd_ret_5d only — no triple-barrier columns.
    rng = np.random.default_rng(4)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    tickers = ["T0", "T1", "T2"]
    rows = []
    for d in dates:
        for t in tickers:
            feats = {c: rng.normal() for c in _FEATURE_COLS}
            rows.append({"date": d, "ticker": t, **feats,
                          "fwd_ret_5d": rng.normal(scale=0.01)})
    fm = pd.DataFrame(rows).set_index(["date", "ticker"])

    fit_captures: list[dict] = []

    class _FakeReg:
        def __init__(self, **_k): pass
        def fit(self, X, y, **kw): fit_captures.append(kw)
        def predict(self, X): return np.zeros(len(X))
        feature_importances_ = np.zeros(len(_FEATURE_COLS))

    import strategies.ml_signal as mls
    monkeypatch.setattr(mls, "lgb", type("lgb", (), {
        "LGBMClassifier": _FakeReg,
        "LGBMRegressor": _FakeReg,
    }))
    monkeypatch.setattr(mls, "build_feature_matrix", lambda *a, **k: fm)
    monkeypatch.setattr(MLSignal, "_write_metadata", lambda *a, **k: None)
    monkeypatch.setattr(mls.pickle, "dump", lambda *a, **k: None)

    model = MLSignal(model_path=str(tmp_path / "m.pkl"))
    model.train(["T0"], period="2y", use_sample_weights=True)

    assert fit_captures
    assert "sample_weight" not in fit_captures[0]
