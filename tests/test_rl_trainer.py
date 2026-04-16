"""Tests for analysis/rl_trainer.py — _prepare_training_df (pure Python, no gymnasium needed)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.rl_sizer import REGIME_STATES
from analysis.rl_trainer import _prepare_training_df


def _make_trades(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Build a minimal trades DataFrame with realised_pnl column."""
    np.random.seed(seed)
    return pd.DataFrame({"realised_pnl": np.random.randn(n) * 100})


# ── _prepare_training_df ───────────────────────────────────────────────────────

def test_raises_without_realised_pnl():
    df = pd.DataFrame({"foo": [1, 2, 3]})
    with pytest.raises(ValueError, match="realised_pnl"):
        _prepare_training_df(df)


def test_adds_regime_column_when_missing():
    df = _make_trades()
    result = _prepare_training_df(df)
    assert "regime" in result.columns
    assert (result["regime"] == "trending_bull").all()


def test_preserves_existing_regime():
    df = _make_trades()
    df["regime"] = "high_vol"
    result = _prepare_training_df(df)
    assert (result["regime"] == "high_vol").all()


def test_unknown_regimes_replaced_with_default():
    df = _make_trades(n=5)
    df["regime"] = ["trending_bull", "invalid_regime", "high_vol", "bogus", "ranging"]
    result = _prepare_training_df(df)
    # "invalid_regime" and "bogus" should be replaced with "trending_bull"
    valid = set(REGIME_STATES)
    assert all(r in valid for r in result["regime"])


def test_adds_volatility_when_missing():
    df = _make_trades()
    result = _prepare_training_df(df)
    assert "volatility" in result.columns
    assert (result["volatility"] == 0.20).all()


def test_preserves_existing_volatility():
    df = _make_trades()
    df["volatility"] = 0.30
    result = _prepare_training_df(df)
    assert (result["volatility"] == 0.30).all()


def test_computes_win_rate_when_missing():
    df = _make_trades(n=40)
    result = _prepare_training_df(df)
    assert "win_rate" in result.columns
    assert result["win_rate"].between(0, 1).all()


def test_preserves_existing_win_rate():
    df = _make_trades(n=10)
    df["win_rate"] = 0.55
    result = _prepare_training_df(df)
    assert (result["win_rate"] == 0.55).all()


def test_computes_drawdown_when_missing():
    df = _make_trades(n=30)
    result = _prepare_training_df(df)
    assert "drawdown" in result.columns
    assert result["drawdown"].between(0, 1).all()


def test_preserves_existing_drawdown():
    df = _make_trades(n=10)
    df["drawdown"] = 0.05
    result = _prepare_training_df(df)
    assert (result["drawdown"] == 0.05).all()


def test_drops_rows_with_nan_realised_pnl():
    df = _make_trades(n=10)
    df.loc[2, "realised_pnl"] = float("nan")
    result = _prepare_training_df(df)
    assert result["realised_pnl"].notna().all()
    assert len(result) == 9


def test_does_not_mutate_input():
    df = _make_trades(n=20)
    original_cols = list(df.columns)
    _prepare_training_df(df)
    assert list(df.columns) == original_cols


def test_win_rate_rolling_window():
    """Rolling win_rate should be bounded [0, 1]."""
    np.random.seed(42)
    df = pd.DataFrame({"realised_pnl": np.random.randn(60) * 100})
    result = _prepare_training_df(df)
    assert result["win_rate"].between(0, 1).all()


def test_all_columns_present_in_output():
    df = _make_trades(n=20)
    result = _prepare_training_df(df)
    for col in ("realised_pnl", "regime", "volatility", "win_rate", "drawdown"):
        assert col in result.columns
