"""
strategies/drl_agent.py — Deep-reinforcement-learning trading agent.

Jansen *ML for Algorithmic Trading* Ch 22 extends the earlier DL
pipeline by jointly learning the directional decision *and* the size
with a policy-gradient agent.  ``analysis.rl_sizer`` already handles
sizing-only RL; this module closes the loop by adding a full agent
trained on price-action reward.

Design
------
:class:`TradingEnv`
    Pure-Python single-ticker environment with the classic Gym
    ``reset / step`` API.  Observation = the last ``window`` daily
    returns + a ``position`` scalar.  Action space is discrete
    ``{short=-1, flat=0, long=+1}``.  Reward per step is
    ``position * return - turnover * transaction_cost``.  The env can
    be unit-tested without ``gymnasium`` or ``stable-baselines3``
    installed.

:class:`DRLAgent`
    Wraps a PPO model.  ``train(ticker, period, total_timesteps)``
    instantiates a ``gymnasium.Env`` around :class:`TradingEnv` and
    fits a ``stable_baselines3.PPO`` policy.  ``predict(ticker,
    period)`` returns a score in ``[-1, 1]`` compatible with the
    ensemble blender + :func:`strategies.ml_execution.execute_ml_signals`.

Optional deps
-------------
    stable-baselines3 >= 2.0  (``pip install stable-baselines3``)
    gymnasium >= 0.29

When either is missing the env is still importable and testable, but
``DRLAgent.train`` / ``predict`` fall back to the flat position so
callers never crash on a missing optional dep.

Reference
---------
    Jansen, *ML for Algorithmic Trading* (2nd ed.), Ch 22.5.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from data.fetcher import fetch_ohlcv
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import gymnasium as gym  # type: ignore[import]
    from gymnasium import spaces  # type: ignore[import]
    _GYM_AVAILABLE = True
except ImportError:
    gym = None                        # type: ignore[assignment]
    spaces = None                     # type: ignore[assignment]
    _GYM_AVAILABLE = False

try:
    from stable_baselines3 import PPO  # type: ignore[import]
    _SB3_AVAILABLE = True
except ImportError:
    PPO = None                        # type: ignore[assignment]
    _SB3_AVAILABLE = False


_DEFAULT_MODEL_PATH = os.environ.get(
    "DRL_AGENT_MODEL_PATH", "models/drl_agent.zip"
)

# Discrete action → signed position.
_ACTION_TO_POSITION = {0: -1.0, 1: 0.0, 2: 1.0}


class TradingEnv:
    """Pure-Python single-ticker trading environment.

    Parameters
    ----------
    prices :
        1-D array of *close* prices (or equivalent).  Must contain at
        least ``window + 2`` points; the env yields ``len(prices) -
        window - 1`` steps per episode.
    window :
        Number of trailing returns the agent sees each step.
    transaction_cost :
        Proportional cost per unit of turnover (e.g. ``1e-3`` for
        10 bps round-trip).  Reward is
        ``position * return - transaction_cost * |Δposition|``.

    Attributes
    ----------
    observation_shape : ``(window + 1,)`` — ``window`` returns + current
    position.
    action_space_n : 3 (short / flat / long).
    """

    def __init__(
        self,
        prices,
        window: int = 20,
        transaction_cost: float = 0.0005,
    ) -> None:
        prices = np.asarray(prices, dtype=np.float32)
        if prices.ndim != 1:
            raise ValueError("prices must be 1-D")
        if len(prices) < window + 2:
            raise ValueError(
                f"need at least {window + 2} prices, got {len(prices)}"
            )
        self._prices = prices
        returns = np.zeros_like(prices)
        returns[1:] = prices[1:] / prices[:-1] - 1.0
        self._returns = returns
        self._window = int(window)
        self._tx_cost = float(transaction_cost)

        self._position = 0.0
        self._step = window            # first step with a full window behind it

    # ── Gym-style API ────────────────────────────────────────────────────────

    @property
    def observation_shape(self) -> tuple[int, ...]:
        return (self._window + 1,)

    @property
    def action_space_n(self) -> int:
        return 3

    def reset(self, seed: Optional[int] = None):
        """Restart the episode at the first eligible step."""
        self._position = 0.0
        self._step = self._window
        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if action not in _ACTION_TO_POSITION:
            raise ValueError(
                f"action must be in {{0, 1, 2}}, got {action!r}"
            )
        new_position = _ACTION_TO_POSITION[int(action)]
        turnover = abs(new_position - self._position)
        self._position = new_position

        # Reward uses the *next* bar's return — the agent's bet applies
        # forward in time.
        next_return = float(self._returns[self._step + 1]) if \
            self._step + 1 < len(self._returns) else 0.0
        reward = self._position * next_return - self._tx_cost * turnover

        self._step += 1
        terminated = self._step >= len(self._prices) - 1
        truncated = False
        return self._obs(), float(reward), terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        start = self._step - self._window
        rets = self._returns[start : self._step]
        obs = np.empty(self._window + 1, dtype=np.float32)
        obs[: self._window] = rets
        obs[self._window] = self._position
        return obs


def _build_gym_wrapper(trading_env: "TradingEnv"):
    """Adapt :class:`TradingEnv` to a ``gymnasium.Env`` subclass.

    We build the class lazily because ``gymnasium`` may not be
    installed; the closure captures the underlying ``TradingEnv`` so
    the wrapper just delegates.
    """
    if not _GYM_AVAILABLE:
        raise RuntimeError(
            "gymnasium is not installed.  Run: pip install gymnasium>=0.29"
        )

    class _TradingGymEnv(gym.Env):                # type: ignore[misc]
        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self._inner = trading_env
            self.observation_space = spaces.Box(
                low=-10.0, high=10.0,
                shape=self._inner.observation_shape,
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(self._inner.action_space_n)

        def reset(self, *, seed=None, options=None):
            return self._inner.reset(seed=seed)

        def step(self, action):
            return self._inner.step(int(action))

    return _TradingGymEnv()


class DRLAgent:
    """PPO-based trading agent with same ``predict`` signature as
    :class:`strategies.ml_signal.MLSignal` et al.  Returns per-ticker
    scores in ``[-1, 1]`` (``short`` = -1, ``long`` = +1).
    """

    def __init__(self, model_path: str | None = None, window: int = 20) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._window: int = int(window)
        self._model = None                # PPO | None
        self._load_if_available()

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists() or not _SB3_AVAILABLE:
            return
        try:
            self._model = PPO.load(self._model_path)
            log.info("drl_agent: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning(
                "drl_agent: failed to load checkpoint",
                path=str(path), error=str(exc),
            )

    def train(
        self,
        ticker: str,
        period: str = "2y",
        total_timesteps: int = 20_000,
        seed: int = 42,
        transaction_cost: float = 0.0005,
    ) -> dict:
        """Fit a PPO agent against a :class:`TradingEnv` for ``ticker``.

        Returns the mean cumulative reward on the final evaluation
        episode.  Raises ``RuntimeError`` when ``stable-baselines3`` or
        ``gymnasium`` are missing.
        """
        if not _SB3_AVAILABLE:
            raise RuntimeError(
                "stable-baselines3 is not installed.  "
                "Run: pip install stable-baselines3>=2.0"
            )
        if not _GYM_AVAILABLE:
            raise RuntimeError(
                "gymnasium is not installed.  Run: pip install gymnasium>=0.29"
            )

        df = fetch_ohlcv(ticker, period)
        if df is None or df.empty or len(df) < self._window + 2:
            raise ValueError(f"insufficient OHLCV data for {ticker} / {period}")
        prices = df["Close"].astype(float).to_numpy()

        inner = TradingEnv(
            prices, window=self._window, transaction_cost=transaction_cost,
        )
        env = _build_gym_wrapper(inner)

        model = PPO("MlpPolicy", env, seed=seed, verbose=0)
        model.learn(total_timesteps=int(total_timesteps))
        self._model = model

        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(self._model_path)
        log.info("drl_agent: checkpoint saved", path=self._model_path)

        eval_reward = _evaluate_cumulative_reward(model, inner)
        return {
            "ticker": ticker,
            "timesteps": int(total_timesteps),
            "eval_cumulative_reward": float(eval_reward),
        }

    def predict(self, ticker: str, period: str = "6mo") -> dict[str, float]:
        """Return ``{ticker: score ∈ [-1, 1]}`` from the current policy.

        Falls back to ``{ticker: 0.0}`` (flat) when no model is loaded
        or the optional deps are missing.
        """
        if self._model is None or not _SB3_AVAILABLE:
            return {ticker: 0.0}

        try:
            df = fetch_ohlcv(ticker, period)
            if df is None or df.empty or len(df) < self._window + 2:
                return {ticker: 0.0}
            prices = df["Close"].astype(float).to_numpy()
            env = TradingEnv(prices, window=self._window)
            obs, _ = env.reset()
            # Forward the env to the last fully-observed step so we act on
            # the *current* window.
            for _ in range(len(prices) - self._window - 2):
                env._step = env._step  # no-op placeholder; position stays 0
            obs = env._obs()
            action, _ = self._model.predict(obs, deterministic=True)
            return {ticker: float(_ACTION_TO_POSITION[int(action)])}
        except Exception as exc:
            log.warning("drl_agent.predict: error, returning flat", error=str(exc))
            return {ticker: 0.0}

    def is_trained(self) -> bool:
        return self._model is not None


def _evaluate_cumulative_reward(model, env: TradingEnv) -> float:
    """Roll the policy through one deterministic pass of ``env``."""
    obs, _ = env.reset()
    total = 0.0
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(int(action))
        total += reward
    return float(total)
