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

Threat model (pickle path confinement)
--------------------------------------
``pickle.load`` executes arbitrary ``__reduce__`` code during deserialisation.
If an attacker can control ``LGBM_ALPHA_MODEL_PATH`` /
``LGBM_REGIME_MODELS_PATH`` they can redirect the agent (and
``strategies/ml_signal.py``) to a crafted pickle anywhere on disk and gain
code execution in the trading process.

Environment variables are therefore treated as **operator-trusted** — they
should only be set by the human who deploys the platform, never by
user-supplied request data or runtime config fetched from an untrusted
source. As defence-in-depth, every pickle load is confined to
``_ALLOWED_ROOTS`` (repo root + system temp dir for tests) via
``_safe_pickle_path`` / ``_confine_pickle_path``. Paths escaping those
roots are refused before ``pickle.load`` runs.

ENV vars
--------
    LGBM_ALPHA_MODEL_PATH       path to baseline pickle
                                (default: models/lgbm_alpha.pkl)
    LGBM_REGIME_MODELS_PATH     path to regime-models pickle
                                (default: models/lgbm_regime_models.pkl)
    KNOWLEDGE_STALE_DAYS        age > this → bearish / retrain (default 45)
    KNOWLEDGE_MONITOR_DAYS      age > this → neutral / monitor (default 35)
    KNOWLEDGE_ALERT_COOLDOWN    seconds between retrain alerts (default 86400)
    KNOWLEDGE_AUTO_RETRAIN      1 → on a retrain verdict, spawn
                                cron.monthly_ml_retrain in a detached
                                subprocess (opt-in; default off)
    KNOWLEDGE_RETRAIN_COOLDOWN  seconds between auto-retrain launches
                                (default 86400)
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess  # noqa: S404 — used only with hard-coded argv (sys.executable + module name)
import sys
import tempfile
import threading
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
_DEFAULT_RETRAIN_COOLDOWN = 24 * 3600.0
_AUTO_RETRAIN_STAMP = "retrain_fired_at"

# Fed to MetaAgent / ml_execution so they can discount conviction without
# changing direction.
_RECOMMENDATION_MULTIPLIER: dict[str, float] = {
    "fresh": 1.0,
    "monitor": 0.7,
    "retrain": 0.4,
}

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Pickle loads are confined to these roots. The repo root covers production
# (``models/*.pkl``); the system temp dir is needed so unit tests can stage
# fixture pickles under ``tmp_path``. Anything outside these roots is refused
# before ``pickle.load`` runs (see ``_confine_pickle_path``).
_ALLOWED_ROOTS: tuple[Path, ...] = (
    _REPO_ROOT.resolve(),
    Path(tempfile.gettempdir()).resolve(),
)


def recommendation_multiplier(recommendation: str) -> float:
    """Return the confidence / sizing multiplier for a recommendation tag."""
    return _RECOMMENDATION_MULTIPLIER.get(recommendation, 1.0)


def _confine_pickle_path(raw_path: str | Path) -> Path:
    """Resolve ``raw_path`` and require it under ``_ALLOWED_ROOTS``.

    Raises ``ValueError`` when the resolved path escapes every allowed root.
    Callers that load via ``pickle`` should invoke this immediately before
    opening the file so traversal tricks (``..``, symlinks) are neutralised.
    """
    resolved = Path(raw_path).expanduser().resolve()
    for root in _ALLOWED_ROOTS:
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    raise ValueError(
        f"pickle path {resolved} is outside allowed roots "
        f"{[str(r) for r in _ALLOWED_ROOTS]}; refusing to load"
    )


def _safe_pickle_path(env_var: str, default_rel: str) -> Path:
    """Resolve a model pickle path — env override wins, else repo-relative
    default — and confine it to ``_ALLOWED_ROOTS``.

    Raises ``ValueError`` on escape so the caller fails closed instead of
    silently loading from an attacker-controlled location.
    """
    override = os.environ.get(env_var)
    src = override if override else str(_REPO_ROOT / default_rel)
    return _confine_pickle_path(src)


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

    The path is re-confined to ``_ALLOWED_ROOTS`` at the load site as
    defence-in-depth; callers should still pass a path produced by
    ``_safe_pickle_path``.
    """
    try:
        safe_path = _confine_pickle_path(path)
    except ValueError as exc:
        logger.error("knowledge_agent: refusing regime-models pickle: %s", exc)
        return []
    if not safe_path.exists():
        return []
    with open(safe_path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301  — confined to _ALLOWED_ROOTS above
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


def _read_stamp(name: str) -> float:
    """Return ``last_fired_at`` for ``name`` from ``knowledge_stamps``.

    Silently returns ``0.0`` if the table is missing or the DB is
    unreachable — treats "no stamp" as "never fired" so the cooldown
    gate opens.
    """
    try:
        from data.db import get_connection
    except Exception:
        return 0.0
    try:
        conn = get_connection()
    except Exception:
        return 0.0
    try:
        row = conn.execute(
            "SELECT last_fired_at FROM knowledge_stamps WHERE name = ?",
            (name,),
        ).fetchone()
    except Exception:
        return 0.0
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if row is None:
        return 0.0
    try:
        value = row["last_fired_at"] if hasattr(row, "keys") else row[0]
    except Exception:
        return 0.0
    return float(value) if value is not None else 0.0


def _write_stamp(name: str, now: float) -> None:
    """Upsert ``(name, now)`` into ``knowledge_stamps``.

    Silently skips on any DB error — a missed stamp only risks one extra
    retrain launch, which is safer than crashing the hot path.
    """
    try:
        from data.db import get_connection
    except Exception:
        return
    try:
        conn = get_connection()
    except Exception:
        return
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_stamps "
                "(name, last_fired_at) VALUES (?, ?)",
                (name, now),
            )
    except Exception as exc:
        logger.debug("knowledge_agent: stamp write failed", error=str(exc))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _watch_retrain_subprocess(proc: subprocess.Popen) -> None:
    """Log the auto-retrain subprocess exit code off the agent's hot path."""
    try:
        rc = proc.wait()
        logger.info(
            "knowledge_agent: auto-retrain finished",
            pid=proc.pid, exit_code=rc,
        )
    except Exception as exc:
        logger.warning(
            "knowledge_agent: auto-retrain watcher failed", error=str(exc),
        )


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


def _resolve_plateau(context: dict, model_name: str = "lgbm_alpha") -> bool:
    """Resolve the ``plateau_detected`` flag.

    Precedence:
      1. ``context['plateau_detected']`` — explicit caller override (tests).
      2. Otherwise, read the last ``n`` ``test_ic_delta`` rows via
         ``analysis.retrain_roi.is_ic_plateau``.

    Fails open to ``False`` so a DB hiccup never spuriously demotes the
    agent's verdict.
    """
    if "plateau_detected" in context:
        return bool(context.get("plateau_detected"))
    try:
        from analysis.retrain_roi import is_ic_plateau
        return bool(is_ic_plateau(model_name, n=3))
    except Exception as exc:
        logger.debug("knowledge_agent: plateau lookup failed: %s", exc)
        return False


def _read_feature_stats(
    model_name: str,
) -> dict[str, dict[str, float]] | None:
    """Return the most recent training-time feature fingerprint for
    ``model_name`` from ``model_feature_stats``, or ``None`` when no
    rows exist or the DB is unreachable."""
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
            "SELECT MAX(trained_at) AS latest FROM model_feature_stats "
            "WHERE model_name = ?",
            (model_name,),
        ).fetchone()
        latest = None if row is None else (
            row["latest"] if hasattr(row, "keys") else row[0]
        )
        if latest is None:
            return None
        rows = conn.execute(
            "SELECT feature_name, mean, std, q10, q50, q90, n_samples "
            "FROM model_feature_stats "
            "WHERE model_name = ? AND trained_at = ?",
            (model_name, latest),
        ).fetchall()
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass
    if not rows:
        return None
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        name = r["feature_name"] if hasattr(r, "keys") else r[0]
        out[str(name)] = {
            "mean": float(r["mean"] if hasattr(r, "keys") else r[1]),
            "std": float(r["std"] if hasattr(r, "keys") else r[2]),
            "q10": float(r["q10"] if hasattr(r, "keys") else r[3]),
            "q50": float(r["q50"] if hasattr(r, "keys") else r[4]),
            "q90": float(r["q90"] if hasattr(r, "keys") else r[5]),
            "n_samples": int(r["n_samples"] if hasattr(r, "keys") else r[6]),
        }
    return out


def _resolve_drift(
    context: dict,
    model_name: str = "lgbm_alpha",
) -> dict | None:
    """Resolve the covariate-shift signal (#118).

    Precedence:
      1. ``context['drift']`` — full dict override (tests / advanced
         callers). Must already match the ``aggregate_drift`` schema.
      2. ``context['drift_score']`` — scalar shortcut (tests only); we
         synthesise a minimal dict with an empty ``drifted_features``.
      3. ``context['feature_frame']`` (a ``pd.DataFrame``) + stored
         training fingerprint → run ``feature_psi`` + ``aggregate_drift``.
      4. Otherwise → ``None`` (drift check is optional; fail-open).

    Returns the ``aggregate_drift`` result dict, or ``None`` when no
    signal is available."""
    if "drift" in context and isinstance(context["drift"], dict):
        return context["drift"]
    if "drift_score" in context:
        score = float(context["drift_score"])
        level = "retrain" if score >= 0.25 else (
            "monitor" if score >= 0.10 else "none"
        )
        return {"level": level, "max_psi": score, "drifted_features": []}

    frame = context.get("feature_frame")
    if frame is None:
        return None

    try:
        stats = _read_feature_stats(model_name)
        if not stats:
            return None
        from analysis.drift import aggregate_drift, feature_psi

        psi = feature_psi(stats, frame)
        if not psi:
            return None
        return aggregate_drift(psi)
    except Exception as exc:
        logger.debug("knowledge_agent: drift lookup failed: %s", exc)
        return None


def _resolve_at_risk(context: dict) -> bool:
    """Resolve the regime-boundary ``at_risk`` flag.

    Precedence:
      1. ``context['at_risk']`` — explicit caller override.
      2. If the caller also supplied ``context['regime']`` they are running
         offline / in tests, so we skip the live lookup and default to False
         (boundary detection requires live SPY/VIX).
      3. Otherwise, read the cached live regime classifier's flag.

    Fail-open to ``False`` on any error so the agent does not accidentally
    demote when the live feed is unavailable.
    """
    if "at_risk" in context:
        return bool(context.get("at_risk"))
    if "regime" in context:
        return False
    try:
        from analysis.regime import get_cached_live_regime
        return bool(get_cached_live_regime(use_llm=False).get("at_risk", False))
    except Exception as exc:
        logger.debug("knowledge_agent: live at_risk lookup failed: %s", exc)
        return False


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
    _popen = staticmethod(subprocess.Popen)
    _retrain_reader = staticmethod(_read_stamp)
    _retrain_writer = staticmethod(_write_stamp)

    def __init__(self) -> None:
        self._cache = _Cache()
        self._last_alert_at: float = 0.0
        self._last_launch_alert_at: float = 0.0

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

        try:
            baseline_path = _safe_pickle_path(
                "LGBM_ALPHA_MODEL_PATH", "models/lgbm_alpha.pkl"
            )
            regime_path = _safe_pickle_path(
                "LGBM_REGIME_MODELS_PATH", "models/lgbm_regime_models.pkl"
            )
        except ValueError as exc:
            logger.error("knowledge_agent: unsafe pickle path: %s", exc)
            return AgentSignal(
                agent_name=self.name,
                signal="bearish",
                confidence=0.8,
                reasoning=f"refusing to load pickle from untrusted path: {exc}",
                metadata={"recommendation": "retrain"},
            )

        baseline_age = _age_days(baseline_path, now)
        regime_age = _age_days(regime_path, now)
        regime = _resolve_regime(context)
        regime_coverage = self._cache.coverage(regime_path, self._coverage_reader)
        at_risk = _resolve_at_risk(context)
        plateau_detected = _resolve_plateau(context)
        drift = _resolve_drift(context)
        drift_level = (drift or {}).get("level") if drift else None
        drift_max_psi = (drift or {}).get("max_psi") if drift else None
        drifted_features = (drift or {}).get("drifted_features") or []

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
            at_risk=at_risk,
            plateau_detected=plateau_detected,
            drift_level=drift_level,
            drift_max_psi=drift_max_psi,
        )

        auto_retrain_meta: dict[str, Any] | None = None
        if verdict["recommendation"] == "retrain":
            self._maybe_alert(verdict["reasoning"], now)
            auto_retrain_meta = self._maybe_auto_retrain(verdict["reasoning"], now)

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
            "at_risk": at_risk,
            "plateau_detected": plateau_detected,
            "drift_level": drift_level,
            "drift_max_psi": drift_max_psi,
            "drifted_features": drifted_features,
        }
        if auto_retrain_meta is not None:
            metadata["auto_retrain"] = auto_retrain_meta

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
        at_risk: bool = False,
        plateau_detected: bool = False,
        drift_level: str | None = None,
        drift_max_psi: float | None = None,
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

        # 3. Covariate shift (#118) above the retrain threshold — the
        #    earliest leading indicator of lost edge. Trumps coverage
        #    gaps because shifted inputs invalidate every regime bucket.
        if drift_level == "retrain":
            psi_text = f" (PSI {drift_max_psi:.2f})" if drift_max_psi is not None else ""
            return _verdict(
                "bearish", 0.7, "retrain",
                f"covariate shift over retrain threshold{psi_text}",
            )

        # 4. Regime coverage gap.
        if regime and regime not in regime_coverage:
            covered = ",".join(regime_coverage) or "none"
            return _verdict(
                "bearish", 0.6, "retrain",
                f"no dedicated model for regime '{regime}' (covered: {covered})",
            )

        # 5. Retrain window approaching.
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

        # 6. Covariate shift approaching retrain threshold — softer rung
        #    so operators investigate before the shift hits 0.25.
        if drift_level == "monitor":
            psi_text = f" (PSI {drift_max_psi:.2f})" if drift_max_psi is not None else ""
            return _verdict(
                "neutral", 0.5, "monitor",
                f"covariate shift approaching retrain threshold{psi_text}",
            )

        # 7. Near a regime boundary — demote fresh to monitor so consumers
        #    downweight the signal proactively.
        if at_risk:
            return _verdict(
                "neutral", 0.5, "monitor",
                "regime near boundary — elevated coverage risk",
            )

        # 8. IC plateau — retraining is not paying off, feature drift suspected.
        #    Demote fresh to monitor so the operator investigates instead of
        #    blindly trusting the next retrain to recover the edge.
        if plateau_detected:
            return _verdict(
                "neutral", 0.5, "monitor",
                "IC plateau over 3 retrains — feature drift suspected",
            )

        # 9. Fresh & covered.
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
            logger.debug("knowledge_agent: alert dispatch failed", error=str(exc))

    # ── Opt-in auto-retrain (#119) ────────────────────────────────────────

    def _maybe_auto_retrain(
        self, reason: str, now: float,
    ) -> dict[str, Any] | None:
        """Fire a detached ``cron.monthly_ml_retrain`` subprocess.

        Opt-in via ``KNOWLEDGE_AUTO_RETRAIN`` (default off). Deduped across
        process restarts via the ``knowledge_stamps`` SQLite table
        (independent from the in-process alert cooldown so an alert can
        fire while the launch is still throttled). Returns a dict that
        the caller folds into ``AgentSignal.metadata`` so operators and
        tests can inspect what happened. Non-blocking — the retrain
        subprocess is detached and watched by a daemon thread.
        """
        if os.environ.get("KNOWLEDGE_AUTO_RETRAIN", "").strip().lower() not in (
            "1", "true", "yes", "on",
        ):
            return None

        cooldown = _env_float(
            "KNOWLEDGE_RETRAIN_COOLDOWN", _DEFAULT_RETRAIN_COOLDOWN,
        )
        last = self._retrain_reader(_AUTO_RETRAIN_STAMP)
        if (now - last) < cooldown:
            return {
                "auto_retrain": "throttled",
                "seconds_until_next": round(cooldown - (now - last), 1),
            }

        try:
            proc = self._popen(
                [sys.executable, "-m", "cron.monthly_ml_retrain"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=True,
            )
        except Exception as exc:
            logger.warning(
                "knowledge_agent: auto-retrain launch failed", error=str(exc),
            )
            return {"auto_retrain": "failed", "error": str(exc)}

        self._retrain_writer(_AUTO_RETRAIN_STAMP, now)
        threading.Thread(
            target=_watch_retrain_subprocess,
            args=(proc,),
            daemon=True,
        ).start()
        self._maybe_launch_alert(reason, proc.pid, now)
        logger.info(
            "knowledge_agent: auto-retrain launched",
            pid=proc.pid, reason=reason,
        )
        return {"auto_retrain": "launched", "pid": proc.pid}

    def _maybe_launch_alert(self, reason: str, pid: int, now: float) -> None:
        """Alert operators with a distinct subject for auto-retrain launches."""
        cooldown = _env_float(
            "KNOWLEDGE_ALERT_COOLDOWN", _DEFAULT_ALERT_COOLDOWN,
        )
        if (now - self._last_launch_alert_at) < cooldown:
            return
        try:
            from providers.alert import get_alert_channel
            channel = get_alert_channel()
            channel.send(
                f"ML knowledge auto-retrain launched (pid={pid}): {reason}",
                level="info",
            )
            self._last_launch_alert_at = now
        except Exception as exc:
            logger.debug(
                "knowledge_agent: launch alert dispatch failed", error=str(exc),
            )


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
