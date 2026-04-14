"""
analysis/rl_trainer.py — PPO training loop for the RL position sizer.

Requires: stable-baselines3 >= 2.0.0, gymnasium >= 0.29.0

All imports are guarded so the rest of the platform continues to work when
these heavy ML dependencies are not installed.

ENV vars
--------
    RL_SIZER_MODEL_PATH     destination for saved checkpoint (default: models/rl_sizer.zip)
    RL_SIZER_TIMESTEPS      PPO training timesteps (default: 50000)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.rl_sizer import (
    _ACTION_HIGH,
    _ACTION_LOW,
    _OBS_DIM,
    REGIME_STATES,
    RLSizerObservation,
    _obs_to_array,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = os.environ.get("RL_SIZER_MODEL_PATH", "models/rl_sizer.zip")
_DEFAULT_TIMESTEPS = int(os.environ.get("RL_SIZER_TIMESTEPS", "50000"))


def _build_env(trades_df: pd.DataFrame):
    """
    Build a gymnasium Env from a DataFrame of closed trades.

    Each row must contain at minimum:
        regime        : str (one of REGIME_STATES)
        volatility    : float
        win_rate      : float  (rolling up to that trade)
        drawdown      : float
        realised_pnl  : float  (reward signal)

    Returns a QuantTradingEnv instance.
    """
    try:
        import gymnasium as gym  # type: ignore[import]
        from gymnasium import spaces  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "gymnasium is required for RL training. "
            "Install it with: pip install gymnasium"
        ) from exc

    class QuantTradingEnv(gym.Env):
        """Single-episode environment: walks through the trade DataFrame."""

        metadata = {"render_modes": []}

        def __init__(self, df: pd.DataFrame) -> None:
            super().__init__()
            self._df = df.reset_index(drop=True)
            self._idx = 0
            self.observation_space = spaces.Box(
                low=np.zeros(_OBS_DIM, dtype=np.float32),
                high=np.ones(_OBS_DIM, dtype=np.float32) * 10,
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=np.array([_ACTION_LOW], dtype=np.float32),
                high=np.array([_ACTION_HIGH], dtype=np.float32),
                dtype=np.float32,
            )

        def _get_obs(self) -> np.ndarray:
            row = self._df.iloc[self._idx]
            obs = RLSizerObservation(
                regime=str(row.get("regime", "high_vol")),
                volatility=float(row.get("volatility", 0.20)),
                win_rate=float(row.get("win_rate", 0.5)),
                drawdown=float(row.get("drawdown", 0.0)),
            )
            return _obs_to_array(obs)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._idx = 0
            return self._get_obs(), {}

        def step(self, action):
            row = self._df.iloc[self._idx]
            pnl = float(row.get("realised_pnl", 0.0))
            multiplier = float(np.clip(action[0], _ACTION_LOW, _ACTION_HIGH))
            reward = pnl * multiplier  # reward = scaled P&L
            self._idx += 1
            done = self._idx >= len(self._df)
            obs = self._get_obs() if not done else np.zeros(_OBS_DIM, dtype=np.float32)
            return obs, reward, done, False, {}

    return QuantTradingEnv(trades_df)


def _prepare_training_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich raw trade records with rolling win_rate and dummy regime/vol/drawdown
    columns where they are absent.
    """
    df = trades_df.copy()

    # Ensure required columns exist
    if "realised_pnl" not in df.columns:
        raise ValueError("trades_df must contain a 'realised_pnl' column")

    if "regime" not in df.columns:
        df["regime"] = "trending_bull"  # neutral assumption when missing
    df["regime"] = df["regime"].fillna("trending_bull")
    # Validate and replace unknown regimes
    df["regime"] = df["regime"].where(df["regime"].isin(REGIME_STATES), "trending_bull")

    if "volatility" not in df.columns:
        df["volatility"] = 0.20

    # Rolling win rate (window=20)
    if "win_rate" not in df.columns:
        is_win = (df["realised_pnl"] > 0).astype(float)
        df["win_rate"] = is_win.rolling(20, min_periods=1).mean()

    if "drawdown" not in df.columns:
        pnl_cum = df["realised_pnl"].cumsum()
        running_max = pnl_cum.cummax()
        df["drawdown"] = ((running_max - pnl_cum) / running_max.replace(0, 1)).clip(0, 1)

    return df.dropna(subset=["realised_pnl"])


def train(
    trades_df: pd.DataFrame,
    total_timesteps: int = _DEFAULT_TIMESTEPS,
    save_path: str = _DEFAULT_MODEL_PATH,
) -> str:
    """
    Train a PPO position sizer on the given trade DataFrame and save the checkpoint.

    Parameters
    ----------
    trades_df       : DataFrame of closed trades with at least 'realised_pnl' column
    total_timesteps : PPO training steps (default: RL_SIZER_TIMESTEPS env var)
    save_path       : destination .zip path for the checkpoint

    Returns
    -------
    Absolute path to the saved checkpoint file.

    Raises
    ------
    ImportError  if stable-baselines3 or gymnasium are not installed
    ValueError   if trades_df is missing required columns or has < 10 rows
    """
    try:
        from stable_baselines3 import PPO  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for RL training. "
            "Install it with: pip install stable-baselines3"
        ) from exc

    df = _prepare_training_df(trades_df)
    if len(df) < 10:
        raise ValueError(
            f"Need at least 10 closed trades to train the RL sizer; got {len(df)}."
        )

    # Ensure model directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    env = _build_env(df)
    logger.info(
        "RL sizer: starting PPO training on %d trades, %d timesteps",
        len(df), total_timesteps,
    )
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)

    abs_path = str(Path(save_path).resolve())
    logger.info("RL sizer: checkpoint saved to %s", abs_path)
    return abs_path
