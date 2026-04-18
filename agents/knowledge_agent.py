"""
agents/knowledge_agent.py — Reviews how well the platform's ML knowledge stays
adapted to live markets (Jansen Ch 12 LGBM + AFML regime conditioning).

The monthly retrain cron (`cron/monthly_ml_retrain.py`) pickles regime-
conditioned LightGBM models to ``models/lgbm_alpha.pkl`` and
``models/lgbm_regime_models.pkl``.  If a retrain silently fails or a new
regime appears without coverage, ``MLSignal.predict()`` quietly falls back to
momentum and P&L erodes without anyone noticing.

This agent checks three signals on every invocation:

    1. Baseline pickle freshness (mtime age in days).
    2. Regime-models pickle freshness + current regime coverage.
    3. IC degradation (``test_ic`` from the most recent training, persisted in
       the ``model_metadata`` table, compared against a caller-supplied
       ``live_ic`` when available).

The agent returns an ``AgentSignal`` plus a ``metadata["recommendation"]``
tag in ``{"fresh", "monitor", "retrain"}`` that the MetaAgent uses to scale
its output confidence and that ``strategies/ml_execution.py`` uses to scale
the Kelly fraction.  A stale verdict also triggers a throttled alert via
``providers.alert`` (24h cooldown) so silent cron failures surface.

ENV vars
--------
    LGBM_ALPHA_MODEL_PATH       path to baseline pickle
                                (default: models/lgbm_alpha.pkl)
    LGBM_REGIME_MODELS_PATH     path to regime-models pickle
                                (default: models/lgbm_regime_models.pkl)
    KNOWLEDGE_STALE_DAYS        age > this → bearish / retrain (default 45)
    KNOWLEDGE_MONITOR_DAYS      age > this → neutral / monitor (default 35)
    KNOWLEDGE_ALERT_COOLDOWN    seconds between retrain alerts (default 86400)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

from agents.base import AgentSignal
from utils.logger import get_logger

logger = get_logger(__name__)

# Default thresholds — overridable via env.
_DEFAULT_STALE_DAYS = 45.0
_DEFAULT_MONITOR_DAYS = 35.0
_IC_DROP_BEARISH = 0.5      # live_ic / trained_ic < this → retrain
_IC_DROP_NEUTRAL = 0.75     # < this → monitor
_CACHE_TTL_SEC = 60.0
_DEFAULT_ALERT_COOLDOWN = 24 * 3600.0

# Fed to MetaAgent / ml_execution so they can discount conviction without
# changing direction.
_RECOMMENDATION_MULTIPLIER: dict[str, float] = {
    "fresh": 1.0,
    "monitor": 0.7,
    "retrain": 0.4,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def recommendation_multiplier(recommendation: str) -> float:
    """Return the confidence / sizing multiplier for a recommendation tag."""
    return _RECOMMENDATION_MULTIPLIER.get(recommendation, 1.0)


def _resolve_path(env_var: str, default_rel: str) -> Path:
    """Resolve a model pickle path — env override wins, else repo-relative default."""
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return (_REPO_ROOT / default_rel).resolve()


def _age_days(path: Path, now: float) -> float | None:
    """Return mtime age in days for an existing file, else ``None``."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    return round(max(0.0, (now - mtime) / 86400.0), 2)


def _read_regime_coverage(path: Path) -> list[str]:
    """Return the list of regime names present in the regime-models pickle.

    Supports both the canonical ``{"models": {...}, "is_classifier": bool}``
    payload and the legacy bare-dict format (see
    ``strategies/ml_signal.py:143–151``).
    """
    if not path.exists():
        return []
    with open(path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301  — trusted local file
    if isinstance(payload, dict):
        inner = payload.get("models") if "models" in payload else payload
        if isinstance(inner, dict):
            return sorted(inner.keys())
    return []


def _read_trained_ic(model_name: str) -> float | None:
    """Return the most recent ``test_ic`` for ``model_name`` from model_metadata.

    Silently returns ``None`` if the table is missing, empty, or the import
    graph of ``data.db`` fails for any reason (e.g., during unit tests that
    haven't bootstrapped the schema).
    """
    try:
        from data.db import get_connection
    except Exception:
        return None
    try:
        conn = get_connection()
    except Exception:
        return None
    try:
        row = conn.execute(
            "SELECT test_ic FROM model_metadata "
            "WHERE model_name = ? ORDER BY trained_at DESC LIMIT 1",
            (model_name,),
        ).fetchone()
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if row is None:
        return None
    try:
        value = row["test_ic"] if hasattr(row, "keys") else row[0]
    except Exception:
        return None
    return float(value) if value is not None else None


def _resolve_regime(context: dict) -> str | None:
    """Resolve the current regime from context or the cached live classifier."""
    regime = context.get("regime")
    if regime:
        return str(regime)
    try:
        from analysis.regime import get_cached_live_regime
        return str(get_cached_live_regime(use_llm=False).get("regime") or "") or None
    except Exception as exc:
        logger.debug("knowledge_agent: live regime lookup failed: %s", exc)
        return None


class _Cache:
    """Tiny TTL cache keyed on (path, mtime) so the hot path stays cheap."""

    def __init__(self, ttl: float = _CACHE_TTL_SEC) -> None:
        self._ttl = ttl
        self._coverage: tuple[tuple[str, float] | None, float, list[str]] = (
            None, 0.0, [],
        )

    def coverage(self, path: Path, loader) -> list[str]:
        """Return cached regime coverage, re-reading if mtime changed or TTL expired."""
        now = time.time()
        try:
            mtime = path.stat().st_mtime
            key = (str(path), mtime)
        except OSError:
            key = None
        cached_key, cached_at, cached_val = self._coverage
        if key is not None and key == cached_key and (now - cached_at) < self._ttl:
            return cached_val
        try:
            result = loader(path)
        except Exception as exc:
            logger.debug("knowledge_agent: regime pickle load failed: %s", exc)
            result = []
        self._coverage = (key, now, result)
        return result


class KnowledgeAdaptionAgent:
    """Reviews model freshness, regime coverage, and IC drift."""

    name = "knowledge_agent"

    # Injection points for tests.  All default to module-level helpers.
    _clock = staticmethod(time.time)
    _coverage_reader = staticmethod(_read_regime_coverage)
    _metadata_reader = staticmethod(_read_trained_ic)

    def __init__(self) -> None:
        self._cache = _Cache()
        self._last_alert_at: float = 0.0

    # ── Main entrypoint ────────────────────────────────────────────────────

    def run(self, context: dict) -> AgentSignal:
        try:
            return self._run(context or {})
        except Exception as exc:
            logger.warning("KnowledgeAdaptionAgent: unexpected failure: %s", exc)
            return AgentSignal(
                agent_name=self.name,
                signal="neutral",
                confidence=0.3,
                reasoning="Knowledge state unavailable",
                metadata={"recommendation": "monitor"},
            )

    def _run(self, context: dict) -> AgentSignal:
        now = self._clock()

        baseline_path = _resolve_path("LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl")
        regime_path = _resolve_path(
            "LGBM_REGIME_MODELS_PATH", "models/lgbm_regime_models.pkl"
        )

        baseline_age = _age_days(baseline_path, now)
        regime_age = _age_days(regime_path, now)
        regime = _resolve_regime(context)
        regime_coverage = self._cache.coverage(regime_path, self._coverage_reader)

        trained_ic = self._metadata_reader("lgbm_alpha")
        live_ic_raw = context.get("live_ic")
        live_ic = float(live_ic_raw) if live_ic_raw is not None else None
        ic_ratio = _safe_ratio(live_ic, trained_ic)

        stale_days = _env_float("KNOWLEDGE_STALE_DAYS", _DEFAULT_STALE_DAYS)
        monitor_days = _env_float("KNOWLEDGE_MONITOR_DAYS", _DEFAULT_MONITOR_DAYS)

        verdict = self._classify(
            baseline_age=baseline_age,
            regime_age=regime_age,
            regime=regime,
            regime_coverage=regime_coverage,
            ic_ratio=ic_ratio,
            stale_days=stale_days,
            monitor_days=monitor_days,
        )

        if verdict["recommendation"] == "retrain":
            self._maybe_alert(verdict["reasoning"], now)

        metadata: dict[str, Any] = {
            "recommendation": verdict["recommendation"],
            "baseline_age_days": baseline_age,
            "regime_age_days": regime_age,
            "regime": regime,
            "regime_coverage": regime_coverage,
            "trained_ic": trained_ic,
            "live_ic": live_ic,
            "ic_ratio": ic_ratio,
            "stale_days": stale_days,
            "monitor_days": monitor_days,
        }

        return AgentSignal(
            agent_name=self.name,
            signal=verdict["signal"],
            confidence=verdict["confidence"],
            reasoning=verdict["reasoning"],
            metadata=metadata,
        )

    # ── Classification ────────────────────────────────────────────────────

    @staticmethod
    def _classify(
        *,
        baseline_age: float | None,
        regime_age: float | None,
        regime: str | None,
        regime_coverage: list[str],
        ic_ratio: float | None,
        stale_days: float,
        monitor_days: float,
    ) -> dict[str, Any]:
        """First-match condition ladder → (signal, confidence, reasoning, recommendation)."""
        # 1. Missing baseline pickle — worst case, no model to serve.
        if baseline_age is None:
            return _verdict(
                "bearish", 0.8, "retrain",
                "baseline model pickle missing",
            )

        # 2. Hard staleness OR heavy IC degradation.
        if baseline_age > stale_days:
            return _verdict(
                "bearish", 0.7, "retrain",
                f"baseline stale ({baseline_age:.0f}d > {stale_days:.0f}d)",
            )
        if ic_ratio is not None and ic_ratio < _IC_DROP_BEARISH:
            return _verdict(
                "bearish", 0.7, "retrain",
                f"live IC decayed ({ic_ratio:.0%} of trained)",
            )

        # 3. Regime coverage gap.
        if regime and regime not in regime_coverage:
            covered = ",".join(regime_coverage) or "none"
            return _verdict(
                "bearish", 0.6, "retrain",
                f"no dedicated model for regime '{regime}' (covered: {covered})",
            )

        # 4. Retrain window approaching.
        if (
            baseline_age > monitor_days
            or (regime_age is not None and regime_age > monitor_days)
            or (ic_ratio is not None and ic_ratio < _IC_DROP_NEUTRAL)
        ):
            if ic_ratio is not None and ic_ratio < _IC_DROP_NEUTRAL:
                why = f"IC decaying ({ic_ratio:.0%} of trained)"
            else:
                oldest = max(baseline_age or 0.0, regime_age or 0.0)
                why = f"retrain window approaching ({oldest:.0f}d)"
            return _verdict("neutral", 0.5, "monitor", why)

        # 5. Fresh & covered.
        return _verdict(
            "bullish", 0.8, "fresh",
            f"models fresh ({baseline_age:.0f}d) and regime '{regime or '?'}' covered",
        )

    # ── Alerts (throttled) ─────────────────────────────────────────────────

    def _maybe_alert(self, reason: str, now: float) -> None:
        cooldown = _env_float("KNOWLEDGE_ALERT_COOLDOWN", _DEFAULT_ALERT_COOLDOWN)
        if (now - self._last_alert_at) < cooldown:
            return
        try:
            from providers.alert import get_alert_channel
            channel = get_alert_channel()
            channel.send(
                f"ML knowledge stale — retrain recommended: {reason}",
                level="warning",
            )
            self._last_alert_at = now
        except Exception as exc:
            logger.debug("knowledge_agent: alert dispatch failed: %s", exc)


# ── Module helpers ─────────────────────────────────────────────────────────

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _safe_ratio(live: float | None, trained: float | None) -> float | None:
    """Return ``live / trained`` guarding against None / zero / opposite signs.

    When the trained IC was positive but live IC went negative, we clamp the
    ratio to 0.0 so the downstream ladder treats it as maximum decay.
    """
    if live is None or trained is None:
        return None
    if trained == 0:
        return None
    if (trained > 0 and live < 0) or (trained < 0 and live > 0):
        return 0.0
    ratio = live / trained
    return round(max(0.0, ratio), 4)


def _verdict(
    signal: str, confidence: float, recommendation: str, reasoning: str,
) -> dict[str, Any]:
    return {
        "signal": signal,
        "confidence": confidence,
        "recommendation": recommendation,
        "reasoning": reasoning,
    }


# ── CLI entrypoint ─────────────────────────────────────────────────────────

def main() -> int:
    """Print the AgentSignal as JSON and exit non-zero when a retrain is needed."""
    sig = KnowledgeAdaptionAgent().run({})
    payload = {
        "agent": sig.agent_name,
        "signal": sig.signal,
        "confidence": sig.confidence,
        "reasoning": sig.reasoning,
        "metadata": sig.metadata,
    }
    print(json.dumps(payload, indent=2, default=str))
    rec = (sig.metadata or {}).get("recommendation")
    return 1 if rec == "retrain" else 0


if __name__ == "__main__":
    sys.exit(main())
