"""
analysis/rl_sizer.py — Reinforcement Learning position sizer.

Trains a PPO agent (Stable-Baselines3) on historical journal trade data to
output a position size multiplier in [0.0, 2.0].  Falls back to Kelly criterion
when the model is not loaded or when stable-baselines3 is not installed.

ENV vars
--------
    RL_SIZER_MODEL_PATH     path to SB3 zip checkpoint (default: models/rl_sizer.zip)
    RL_SIZER_RETRAIN        set to '1' to trigger retraining in monthly cron

Optional dependencies (guarded):
    stable-baselines3 >= 2.0.0
    gymnasium >= 0.29.0
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analysis.regime import REGIME_STATES, kelly_regime_multiplier
from risk.kelly import kelly_fraction
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = os.environ.get("RL_SIZER_MODEL_PATH", "models/rl_sizer.zip")

# Observation vector layout:
#   [0..3] regime one-hot (trending_bull, trending_bear, mean_reverting, high_vol)
#   [4]    30-day realised volatility (annualised, e.g. 0.20 = 20%)
#   [5]    recent win rate (last 20 trades)
#   [6]    current drawdown from peak (e.g. 0.05 = 5%)
_OBS_DIM = 7
_ACTION_LOW, _ACTION_HIGH = 0.0, 2.0


@dataclass
class RLSizerObservation:
    """Structured observation for the RL position sizer."""

    regime: str          # one of REGIME_STATES
    volatility: float    # annualised realised vol, e.g. 0.20
    win_rate: float      # recent win rate in [0, 1]
    drawdown: float      # current drawdown from equity peak, in [0, 1]


def _obs_to_array(obs: RLSizerObservation) -> np.ndarray:
    """Encode observation to a fixed-length float32 array of shape (7,)."""
    regime_oh = [1.0 if r == obs.regime else 0.0 for r in REGIME_STATES]
    return np.array(
        regime_oh + [obs.volatility, obs.win_rate, obs.drawdown],
        dtype=np.float32,
    )


class RLPositionSizer:
    """
    Position size multiplier using a trained PPO agent.

    Falls back to Kelly-based sizing when:
    - stable-baselines3 is not installed
    - the model checkpoint file does not exist
    - any inference error occurs
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or _DEFAULT_MODEL_PATH
        self._model = None
        self._load_if_available()

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            logger.info("RL sizer: model not found at %s, will use Kelly fallback", path)
            return
        try:
            from stable_baselines3 import PPO  # type: ignore[import]
            self._model = PPO.load(str(path))
            logger.info("RL sizer: loaded model from %s", path)
        except ImportError:
            logger.info("stable-baselines3 not installed; RL sizer using Kelly fallback")
        except Exception as exc:
            logger.warning("RL sizer: failed to load model: %s", exc)

    def load(self, path: str) -> None:
        """Hot-reload a new checkpoint without restarting."""
        self._model_path = path
        self._model = None
        self._load_if_available()

    def predict(self, obs: RLSizerObservation) -> float:
        """
        Return a position size multiplier in [0.0, 2.0].

        Uses the PPO model when available; otherwise delegates to Kelly fallback.
        """
        if self._model is None:
            return self._kelly_fallback(obs)
        try:
            arr = _obs_to_array(obs)
            action, _ = self._model.predict(arr, deterministic=True)
            multiplier = float(np.clip(action, _ACTION_LOW, _ACTION_HIGH))
            if math.isnan(multiplier):
                logger.warning("RL sizer: model returned NaN; using Kelly fallback")
                return self._kelly_fallback(obs)
            logger.debug("RL sizer: obs=%s → multiplier=%.3f", obs, multiplier)
            return multiplier
        except Exception as exc:
            logger.warning("RL sizer predict error: %s; using Kelly fallback", exc)
            return self._kelly_fallback(obs)

    def _kelly_fallback(self, obs: RLSizerObservation) -> float:
        """Derive a size multiplier from Kelly fraction and regime adjustment."""
        # Conservative Kelly parameters inferred from win rate
        avg_win = 0.05   # assume 5% average win
        avg_loss = 0.03  # assume 3% average loss
        kelly = kelly_fraction(obs.win_rate, avg_win, avg_loss, max_fraction=0.5)
        regime_mult = kelly_regime_multiplier(obs.regime)
        # Normalise Kelly [0, 0.5] to multiplier [0, 2], then apply regime factor
        raw = (kelly / 0.25) * regime_mult  # 0.25 Kelly → 1.0x multiplier
        return float(np.clip(raw, _ACTION_LOW, _ACTION_HIGH))


def build_observation_from_live() -> RLSizerObservation:
    """
    Build an RLSizerObservation from live platform data.

    Fetches regime, SPY volatility, journal win rate, and paper account drawdown.
    Falls back gracefully if any source is unavailable.
    """
    # 1. Regime
    try:
        from analysis.regime import get_live_regime
        regime_data = get_live_regime()
        regime = regime_data["regime"]
    except Exception as exc:
        logger.warning("build_observation_from_live: regime fetch failed: %s", exc)
        regime = "high_vol"  # conservative fallback

    # 2. Realised volatility (SPY 30-day)
    try:
        from data.fetcher import fetch_ohlcv
        spy_df = fetch_ohlcv("SPY", period="3mo")
        if not spy_df.empty and "Close" in spy_df.columns:
            returns = spy_df["Close"].pct_change().dropna()
            volatility = float(returns.std() * (252 ** 0.5))  # annualised
        else:
            volatility = 0.20
    except Exception:
        volatility = 0.20

    # 3. Recent win rate (last 20 closed trades from journal)
    win_rate = 0.5  # neutral default
    try:
        from journal.trading_journal import get_trades
        trades = get_trades()
        if trades is not None and not trades.empty:
            closes = trades[trades.get("action", trades.columns[0]).str.upper() == "SELL"] \
                if "action" in trades.columns else trades
            if "realised_pnl" in closes.columns:
                recent = closes["realised_pnl"].dropna().tail(20)
                if len(recent) > 0:
                    win_rate = float((recent > 0).mean())
    except Exception:
        pass

    # 4. Current drawdown from paper account
    drawdown = 0.0
    try:
        from broker.paper_trader import get_account
        acct = get_account()
        starting = acct.get("starting_cash", 100_000)
        total = acct.get("total_value", starting)
        if starting > 0:
            drawdown = float(max(0.0, (starting - total) / starting))
    except Exception:
        pass

    return RLSizerObservation(
        regime=regime,
        volatility=volatility,
        win_rate=win_rate,
        drawdown=drawdown,
    )
