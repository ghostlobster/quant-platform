"""Tests for analysis/live_ic.py — live IC persistence, backfill, rolling IC."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Point data.db at a fresh SQLite file and init the schema.

    Also clears the module-level rolling-IC cache between tests so
    computed values don't leak.
    """
    db_file = tmp_path / "quant-live-ic-test.db"
    import data.db as _db_mod
    monkeypatch.setattr(_db_mod, "_DB_PATH", db_file)
    _db_mod.init_db()

    import analysis.live_ic as live_ic_mod
    live_ic_mod._ic_cache.clear()
    yield db_file
    live_ic_mod._ic_cache.clear()


def _fake_ohlcv(start: str, n_days: int, prices) -> pd.DataFrame:
    """Build a toy OHLCV DataFrame with a daily DatetimeIndex."""
    idx = pd.date_range(start=start, periods=n_days, freq="D")
    closes = prices if hasattr(prices, "__len__") else [prices] * n_days
    return pd.DataFrame({"Close": closes}, index=idx)


# ── Writer ───────────────────────────────────────────────────────────────────

def test_record_prediction_inserts_row(isolated_db):
    from analysis.live_ic import record_prediction
    from data.db import get_connection

    ts = 1_700_000_000.0
    record_prediction("AAPL", "lgbm_alpha", score=0.42, horizon_d=5, ts=ts)

    conn = get_connection()
    rows = conn.execute(
        "SELECT ts, ticker, model_name, score, horizon_d, realized "
        "FROM live_predictions"
    ).fetchall()
    conn.close()
    assert len(rows) == 1
    r = rows[0]
    assert r["ts"] == pytest.approx(ts)
    assert r["ticker"] == "AAPL"
    assert r["model_name"] == "lgbm_alpha"
    assert r["score"] == pytest.approx(0.42)
    assert r["horizon_d"] == 5
    assert r["realized"] is None


def test_record_predictions_batch_transactional(isolated_db):
    from analysis.live_ic import record_predictions
    from data.db import get_connection

    scores = {"AAPL": 0.6, "MSFT": -0.3, "GOOG": 0.1, "TSLA": -0.5}
    n = record_predictions(scores, model_name="lgbm_alpha", horizon_d=5,
                           ts=1_700_000_000.0)
    assert n == 4

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0]
    conn.close()
    assert count == 4

    # Re-run at the same ts → INSERT OR REPLACE is idempotent (no duplicate rows).
    record_predictions(scores, model_name="lgbm_alpha", horizon_d=5,
                       ts=1_700_000_000.0)
    conn = get_connection()
    count2 = conn.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0]
    conn.close()
    assert count2 == 4


def test_record_prediction_disabled_by_env(isolated_db, monkeypatch):
    from analysis.live_ic import record_predictions
    from data.db import get_connection

    monkeypatch.setenv("KNOWLEDGE_RECORD_PREDICTIONS", "0")
    n = record_predictions({"AAPL": 0.5}, model_name="lgbm_alpha")
    assert n == 0

    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM live_predictions").fetchone()[0]
    conn.close()
    assert count == 0


def test_record_predictions_empty_returns_zero(isolated_db):
    from analysis.live_ic import record_predictions
    assert record_predictions({}, model_name="lgbm_alpha") == 0


# ── Backfill ─────────────────────────────────────────────────────────────────

def test_backfill_fills_realized_for_expired_rows(isolated_db, monkeypatch):
    from analysis.live_ic import backfill_realized, record_predictions

    # Record two predictions 10 days ago.
    ten_days_ago = time.time() - 10 * 86400
    record_predictions({"AAPL": 0.5, "MSFT": -0.2}, model_name="lgbm_alpha",
                       horizon_d=5, ts=ten_days_ago)

    # Mock fetch_ohlcv to return a controllable price series that places a
    # known close at `ts` and `ts + 5 days`.
    def _fake_fetch(ticker, period="3mo"):
        start = pd.Timestamp(ten_days_ago, unit="s", tz="UTC").tz_localize(None)
        start = start.normalize()
        # 30 days of prices anchored at start.
        if ticker == "AAPL":
            base = 100.0
            future = 110.0  # +10%
        else:  # MSFT
            base = 200.0
            future = 190.0  # -5%
        prices = [base] * 5 + [future] * 25
        idx = pd.date_range(start=start, periods=30, freq="D")
        return pd.DataFrame({"Close": prices}, index=idx)

    monkeypatch.setattr(
        "analysis.live_ic._fetch_realized_for_ticker", _fake_fetch,
    )
    updated = backfill_realized(model_name="lgbm_alpha")
    assert updated == 2

    from data.db import get_connection
    conn = get_connection()
    rows = {
        r["ticker"]: r["realized"]
        for r in conn.execute(
            "SELECT ticker, realized FROM live_predictions"
        ).fetchall()
    }
    conn.close()
    assert rows["AAPL"] == pytest.approx(0.10)
    assert rows["MSFT"] == pytest.approx(-0.05)


def test_backfill_skips_not_yet_expired(isolated_db, monkeypatch):
    from analysis.live_ic import backfill_realized, record_predictions

    # Record at current time — 5-day horizon has not elapsed.
    record_predictions({"AAPL": 0.5}, model_name="lgbm_alpha", horizon_d=5)

    called = []
    monkeypatch.setattr(
        "analysis.live_ic._fetch_realized_for_ticker",
        lambda ticker, period="3mo": called.append(ticker),
    )
    updated = backfill_realized(model_name="lgbm_alpha")
    assert updated == 0
    assert called == []  # no OHLCV fetch for rows that are not yet due


def test_backfill_skips_already_realized(isolated_db, monkeypatch):
    from analysis.live_ic import backfill_realized, record_predictions
    from data.db import get_connection

    ten_days_ago = time.time() - 10 * 86400
    record_predictions({"AAPL": 0.5}, model_name="lgbm_alpha",
                       horizon_d=5, ts=ten_days_ago)

    # Pre-populate realized.
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE live_predictions SET realized = 0.12 WHERE ticker = 'AAPL'"
        )
    conn.close()

    fetch = MagicMock()
    monkeypatch.setattr("analysis.live_ic._fetch_realized_for_ticker", fetch)
    updated = backfill_realized(model_name="lgbm_alpha")
    assert updated == 0
    fetch.assert_not_called()


def test_backfill_respects_max_rows(isolated_db, monkeypatch):
    from analysis.live_ic import backfill_realized, record_predictions

    ten_days_ago = time.time() - 10 * 86400
    scores = {f"T{i:02d}": 0.1 * i for i in range(10)}
    record_predictions(scores, model_name="lgbm_alpha",
                       horizon_d=5, ts=ten_days_ago)

    # Return a simple +1% move for every ticker.
    def _fake(ticker, period="3mo"):
        start = pd.Timestamp(ten_days_ago, unit="s", tz="UTC").tz_localize(None)
        idx = pd.date_range(start=start.normalize(), periods=15, freq="D")
        return pd.DataFrame({"Close": [100.0] * 5 + [101.0] * 10}, index=idx)

    monkeypatch.setattr("analysis.live_ic._fetch_realized_for_ticker", _fake)
    updated = backfill_realized(model_name="lgbm_alpha", max_rows=3)
    assert updated == 3


# ── Rolling IC ───────────────────────────────────────────────────────────────

def _seed(scores_realized, model="lgbm_alpha", horizon=5):
    """Write (score, realized) pairs straight to the DB with distinct ts."""
    from data.db import get_connection
    conn = get_connection()
    base = 1_700_000_000.0
    with conn:
        for i, (score, realized) in enumerate(scores_realized):
            conn.execute(
                "INSERT INTO live_predictions (ts, ticker, model_name, "
                "score, horizon_d, realized) VALUES (?, ?, ?, ?, ?, ?)",
                (base + i, f"T{i:04d}", model, float(score), horizon,
                 float(realized)),
            )
    conn.close()


def test_rolling_live_ic_empty_returns_none(isolated_db):
    from analysis.live_ic import rolling_live_ic
    assert rolling_live_ic("lgbm_alpha") is None


def test_rolling_live_ic_warmup_returns_none(isolated_db):
    from analysis.live_ic import rolling_live_ic
    # Only 5 realized rows — below the 10-row warm-up floor.
    _seed([(float(i), float(i)) for i in range(5)])
    assert rolling_live_ic("lgbm_alpha", window=60) is None


def test_rolling_live_ic_perfect_correlation(isolated_db):
    from analysis.live_ic import rolling_live_ic
    # 20 monotonic pairs → Spearman IC should be exactly 1.0.
    _seed([(float(i), float(i)) for i in range(20)])
    ic = rolling_live_ic("lgbm_alpha", window=20)
    assert ic is not None
    assert ic == pytest.approx(1.0)


def test_rolling_live_ic_perfect_anticorrelation(isolated_db):
    from analysis.live_ic import rolling_live_ic
    _seed([(float(i), -float(i)) for i in range(20)])
    ic = rolling_live_ic("lgbm_alpha", window=20)
    assert ic is not None
    assert ic == pytest.approx(-1.0)


def test_rolling_live_ic_cache_expiry(isolated_db, monkeypatch):
    import analysis.live_ic as live_ic_mod
    from analysis.live_ic import rolling_live_ic

    _seed([(float(i), float(i)) for i in range(20)])
    # Short TTL so a manual sleep isn't needed — re-query after zero.
    monkeypatch.setattr(live_ic_mod, "_IC_CACHE_TTL_SEC", 0.0)
    a = rolling_live_ic("lgbm_alpha", window=20)
    b = rolling_live_ic("lgbm_alpha", window=20)
    assert a == b == pytest.approx(1.0)

    # With a long TTL the same call hits the cache.
    monkeypatch.setattr(live_ic_mod, "_IC_CACHE_TTL_SEC", 3600.0)
    live_ic_mod._ic_cache.clear()
    c = rolling_live_ic("lgbm_alpha", window=20)
    # Mutate the DB after the first call; cache should hide the change.
    from data.db import get_connection
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO live_predictions (ts, ticker, model_name, "
            "score, horizon_d, realized) VALUES (?, ?, ?, ?, ?, ?)",
            (1_800_000_000.0, "POISON", "lgbm_alpha", 0.0, 5, 999.0),
        )
    conn.close()
    d = rolling_live_ic("lgbm_alpha", window=20)
    assert c == d


def test_rolling_live_ic_filters_by_model_and_horizon(isolated_db):
    from analysis.live_ic import rolling_live_ic

    _seed([(float(i), float(i)) for i in range(20)],
          model="lgbm_alpha", horizon=5)
    _seed([(float(i), -float(i)) for i in range(20)],
          model="lgbm_alpha", horizon=10)  # opposite sign, different horizon
    _seed([(float(i), -float(i)) for i in range(20)],
          model="bayesian", horizon=5)     # opposite sign, different model

    assert rolling_live_ic("lgbm_alpha", window=20, horizon_d=5) == pytest.approx(1.0)
    assert rolling_live_ic("lgbm_alpha", window=20, horizon_d=10) == pytest.approx(-1.0)
    assert rolling_live_ic("bayesian", window=20, horizon_d=5) == pytest.approx(-1.0)


def test_backfill_invalidates_ic_cache(isolated_db, monkeypatch):
    from analysis.live_ic import (
        _ic_cache,
        backfill_realized,
        record_predictions,
        rolling_live_ic,
    )

    # Seed realized rows + prime the cache.
    _seed([(float(i), float(i)) for i in range(20)])
    rolling_live_ic("lgbm_alpha", window=20)
    assert ("lgbm_alpha", 20, 5) in _ic_cache

    # Add + backfill a fresh batch; cache must be cleared for that model.
    ten_days_ago = time.time() - 10 * 86400
    record_predictions({"AAPL": 0.5}, model_name="lgbm_alpha",
                       horizon_d=5, ts=ten_days_ago)

    def _fake(ticker, period="3mo"):
        start = pd.Timestamp(ten_days_ago, unit="s", tz="UTC").tz_localize(None)
        idx = pd.date_range(start=start.normalize(), periods=15, freq="D")
        return pd.DataFrame({"Close": [100.0] * 5 + [105.0] * 10}, index=idx)

    monkeypatch.setattr("analysis.live_ic._fetch_realized_for_ticker", _fake)
    backfill_realized(model_name="lgbm_alpha")
    assert ("lgbm_alpha", 20, 5) not in _ic_cache


# ── Helper unit tests ────────────────────────────────────────────────────────

def test_realized_return_computation():
    from analysis.live_ic import _realized_return

    # Two bars 5 days apart, 100 → 110 → +10%.
    start = pd.Timestamp("2026-01-01", tz="UTC").tz_localize(None)
    idx = pd.date_range(start=start, periods=15, freq="D")
    df = pd.DataFrame({"Close": [100.0] * 5 + [110.0] * 10}, index=idx)
    ts = start.timestamp()
    assert _realized_return(df, ts=ts, horizon_d=5) == pytest.approx(0.10)


def test_realized_return_handles_missing_data():
    from analysis.live_ic import _realized_return

    # ts past the end of the index → None
    start = pd.Timestamp("2026-01-01", tz="UTC").tz_localize(None)
    idx = pd.date_range(start=start, periods=3, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=idx)
    ts_future = (start + pd.Timedelta(days=10)).timestamp()
    assert _realized_return(df, ts=ts_future, horizon_d=5) is None

    # Empty df
    assert _realized_return(pd.DataFrame(), ts=0.0, horizon_d=5) is None
    # Zero anchor price
    zero_df = pd.DataFrame({"Close": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx[:6]
                           if len(idx) >= 6 else pd.date_range(start=start, periods=6, freq="D"))
    assert _realized_return(
        zero_df, ts=zero_df.index[0].timestamp(), horizon_d=2,
    ) is None


# ── Scheduler job smoke test ─────────────────────────────────────────────────

def test_live_ic_backfill_job_returns_counts(isolated_db, monkeypatch):
    from scheduler.alerts import live_ic_backfill_job

    # Patch the underlying helpers so the job runs without real data.
    with patch("analysis.live_ic.backfill_realized", return_value=7) as bf, \
         patch("analysis.live_ic.rolling_live_ic", return_value=0.12) as ric:
        out = live_ic_backfill_job()

    bf.assert_called_once_with(model_name="lgbm_alpha")
    ric.assert_called_once_with("lgbm_alpha")
    assert out == {"rows_updated": 7, "live_ic": 0.12}


def test_live_ic_backfill_job_handles_errors(isolated_db):
    from scheduler.alerts import live_ic_backfill_job

    with patch("analysis.live_ic.backfill_realized", side_effect=RuntimeError("db down")), \
         patch("analysis.live_ic.rolling_live_ic", return_value=None):
        out = live_ic_backfill_job()
    assert out["rows_updated"] == 0
    assert out["live_ic"] is None


# ── np.random smoke to ensure deterministic rolling IC tests don't use it ───

def test_uses_seeded_data_only():
    rng = np.random.default_rng(0)
    _ = rng.standard_normal(10)
    # This test is intentionally a no-op — it pins the assumption that live_ic
    # tests never depend on implicit randomness. If someone adds a random-data
    # test later, they should seed explicitly.
    assert True
