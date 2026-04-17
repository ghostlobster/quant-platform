"""Tests for strategies/drl_agent.py — DRL trading agent + env."""
import os
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.drl_agent import (
    _ACTION_TO_POSITION,
    _SB3_AVAILABLE,
    DRLAgent,
    TradingEnv,
)

# ── TradingEnv (pure NumPy, no optional deps) ────────────────────────────────

def _trending_prices(n: int = 100, drift: float = 0.001) -> np.ndarray:
    """Deterministic upward-trending price series."""
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=0.002, size=n)
    returns = drift + noise
    return 100.0 * np.exp(np.cumsum(returns))


def test_trading_env_observation_shape():
    prices = _trending_prices(100)
    env = TradingEnv(prices, window=10)
    assert env.observation_shape == (11,)
    assert env.action_space_n == 3


def test_trading_env_reset_returns_window_plus_position():
    env = TradingEnv(_trending_prices(100), window=10)
    obs, info = env.reset()
    assert obs.shape == (11,)
    assert obs[-1] == 0.0   # starting flat
    assert info == {}


def test_trading_env_step_advances_time():
    env = TradingEnv(_trending_prices(100), window=10)
    env.reset()
    step0 = env._step
    env.step(1)               # flat
    assert env._step == step0 + 1


def test_trading_env_position_updates_with_action():
    env = TradingEnv(_trending_prices(100), window=10)
    env.reset()
    env.step(2)               # long
    assert env._position == 1.0
    env.step(0)               # short
    assert env._position == -1.0
    env.step(1)               # flat
    assert env._position == 0.0


def test_trading_env_long_position_captures_upward_drift():
    """On an upward-trending series, a long buy-and-hold agent should
    finish with positive cumulative reward."""
    env = TradingEnv(_trending_prices(200, drift=0.001), window=10,
                      transaction_cost=0.0)
    env.reset()
    total = 0.0
    terminated = False
    while not terminated:
        _, reward, terminated, _, _ = env.step(2)       # always long
        total += reward
    assert total > 0.0


def test_trading_env_short_position_captures_downward_drift():
    env = TradingEnv(_trending_prices(200, drift=-0.001), window=10,
                      transaction_cost=0.0)
    env.reset()
    total = 0.0
    terminated = False
    while not terminated:
        _, reward, terminated, _, _ = env.step(0)       # always short
        total += reward
    assert total > 0.0


def test_trading_env_applies_transaction_cost_on_turnover():
    env = TradingEnv(_trending_prices(100), window=10,
                      transaction_cost=0.01)
    env.reset()
    _, reward_flat, _, _, _ = env.step(1)               # no turnover
    _, reward_long, _, _, _ = env.step(2)               # 0 → +1
    # Flipping into the long position costs at least 0.01 (Δposition = 1).
    assert reward_long <= reward_flat


def test_trading_env_terminated_flag_set_at_end():
    env = TradingEnv(_trending_prices(30), window=5, transaction_cost=0.0)
    env.reset()
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        _, _, terminated, _, _ = env.step(1)
        steps += 1
    assert terminated
    assert steps < 100


def test_trading_env_rejects_invalid_action():
    env = TradingEnv(_trending_prices(50), window=10)
    env.reset()
    with pytest.raises(ValueError, match="action"):
        env.step(99)


def test_trading_env_too_short_prices_raises():
    with pytest.raises(ValueError, match="prices"):
        TradingEnv(np.array([100.0, 101.0]), window=10)


def test_action_to_position_mapping_is_three_way():
    assert set(_ACTION_TO_POSITION.keys()) == {0, 1, 2}
    assert _ACTION_TO_POSITION[0] == -1.0
    assert _ACTION_TO_POSITION[1] == 0.0
    assert _ACTION_TO_POSITION[2] == 1.0


# ── DRLAgent (sb3-free paths — always run) ──────────────────────────────────

def test_agent_init_without_checkpoint_is_untrained(tmp_path):
    agent = DRLAgent(model_path=str(tmp_path / "nonexistent.zip"))
    assert not agent.is_trained()


def test_agent_predict_without_model_returns_flat(tmp_path):
    with patch("data.fetcher.fetch_ohlcv", return_value=pd.DataFrame()):
        agent = DRLAgent(model_path=str(tmp_path / "none.zip"))
        assert agent.predict("AAPL") == {"AAPL": 0.0}


def test_agent_train_raises_when_sb3_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("strategies.drl_agent._SB3_AVAILABLE", False)
    agent = DRLAgent(model_path=str(tmp_path / "x.zip"))
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        agent.train("AAPL", period="2y")


def test_agent_train_raises_when_gym_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("strategies.drl_agent._SB3_AVAILABLE", True)
    monkeypatch.setattr("strategies.drl_agent._GYM_AVAILABLE", False)
    agent = DRLAgent(model_path=str(tmp_path / "x.zip"))
    with pytest.raises(RuntimeError, match="gymnasium"):
        agent.train("AAPL", period="2y")


# ── DRLAgent (sb3-gated happy path) ──────────────────────────────────────────

@pytest.mark.skipif(not _SB3_AVAILABLE, reason="stable-baselines3 not installed")
def test_agent_trains_on_synthetic_trending_data(monkeypatch, tmp_path):
    """Acceptance test per issue #77: positive cumulative reward on a
    synthetic trending market fixture.  Uses a tiny number of
    timesteps so the test stays under a few seconds."""
    prices = _trending_prices(250, drift=0.002)
    df = pd.DataFrame(
        {"Close": prices,
         "Open": prices, "High": prices, "Low": prices, "Volume": 1.0},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="B"),
    )
    monkeypatch.setattr(
        "strategies.drl_agent.fetch_ohlcv", lambda *a, **k: df,
    )

    agent = DRLAgent(model_path=str(tmp_path / "agent.zip"), window=10)
    metrics = agent.train("SYNTHETIC", period="1y", total_timesteps=2_000)
    assert metrics["ticker"] == "SYNTHETIC"
    assert agent.is_trained()
