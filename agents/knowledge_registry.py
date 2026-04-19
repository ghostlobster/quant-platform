"""
agents/knowledge_registry.py — Model zoo registry for KnowledgeAdaptionAgent.

`KnowledgeAdaptionAgent` used to audit only the two LightGBM pickles
shipped by `strategies/ml_signal.py`. The repo carries several other
model families (bayesian, ridge, mlp, cnn, lstm, rf-long-short) that
were silently unchecked — a stale member could pollute the ensemble
blend without ever showing up in the agent's verdict.

This module provides the declarative surface the agent iterates over:

  1. :class:`ModelEntry` — the immutable per-model record: name, env
     var that overrides the artefact path, repo-relative default,
     `model_metadata.model_name` string, and staleness budget.

  2. :func:`build_default_registry` — collects every ``MODEL_ENTRY``
     constant exported by the strategy modules and returns the active
     list. Each strategy import is wrapped in a ``try / except`` so a
     missing optional dependency (e.g., torch for ``cnn_signal``)
     degrades to "that model is skipped" instead of breaking the whole
     audit.

Strategy modules should declare their entry as:

    from agents.knowledge_registry import ModelEntry

    MODEL_ENTRY = ModelEntry(
        name="bayesian_alpha",
        artefact_env="BAYES_ALPHA_MODEL_PATH",
        artefact_default="models/bayesian_alpha.pkl",
        metadata_name="bayesian_alpha",
    )

Tests can inject their own registry via
``KnowledgeAdaptionAgent(registry=[ModelEntry(...)])``.
"""
from __future__ import annotations

from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """Immutable description of one model family for the knowledge agent."""

    name: str
    """Short display label used in logs and metadata keys."""

    artefact_env: str
    """Env var that operators set to override the pickle/checkpoint path."""

    artefact_default: str
    """Repo-relative fallback when the env var is absent (e.g., ``models/x.pkl``)."""

    metadata_name: str
    """Must match ``model_metadata.model_name`` for IC lookups."""

    max_age_days: int = 45
    """Pickle mtime older than this → retrain verdict for this entry."""

    is_baseline: bool = False
    """The baseline owns the aux branches (regime coverage, IC ratio, drift).
    There should be exactly one baseline entry in the active registry."""


# Strategy module path → attribute name for each model entry.
_STRATEGY_MODULES: tuple[str, ...] = (
    "strategies.ml_signal",
    "strategies.linear_signal",
    "strategies.bayesian_signal",
    "strategies.mlp_signal",
    "strategies.rf_long_short",
    "strategies.cnn_signal",
    "strategies.dl_signal",
)


def build_default_registry() -> list[ModelEntry]:
    """Collect every strategy module's ``MODEL_ENTRY`` constant.

    Each module is imported in a ``try / except`` so a missing optional
    dependency (torch for the deep-learning signals, for instance) only
    drops the affected entry — the agent still audits every other
    family. Modules that have no ``MODEL_ENTRY`` attribute are skipped
    with a debug-level log line.

    Returns the entries in the order defined by ``_STRATEGY_MODULES``,
    which keeps the baseline (``lgbm_alpha``) first so the agent's
    fallback lookup never has to search the list.
    """
    import importlib

    entries: list[ModelEntry] = []
    for module_name in _STRATEGY_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(
                "knowledge_registry: skipping strategy import",
                module=module_name, error=str(exc),
            )
            continue
        entry = getattr(module, "MODEL_ENTRY", None)
        if isinstance(entry, ModelEntry):
            entries.append(entry)
        else:
            logger.debug(
                "knowledge_registry: module has no MODEL_ENTRY",
                module=module_name,
            )
    return entries


def worst_recommendation(recommendations: list[str]) -> str:
    """Return the most severe tag in ``recommendations``.

    Priority: ``retrain`` > ``monitor`` > ``fresh``. Empty list → ``fresh``.
    Unknown tags are treated as ``fresh`` so a typo never silently
    promotes the agent's verdict to a halt.
    """
    priority = {"retrain": 2, "monitor": 1, "fresh": 0}
    if not recommendations:
        return "fresh"
    return max(recommendations, key=lambda r: priority.get(r, 0))
