# Plan review — Issue #123: ModelEntry zoo registry for KnowledgeAdaptionAgent

- **Date:** 2026-04-18 (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** `/root/.claude/plans/issue-123-model-zoo-registry.md`
- **Overall:** minor

## Findings

### 1. Trading philosophy
aligned — The registry directly serves Pillar 1 (edge identification): stale zoo members silently polluting the ensemble blend is exactly the failure mode TRADING_PHILOSOPHY.md §1 guards against by requiring statistical edges to be systematic and tested. Pillar 3 (humility through testing) is reinforced by extending staleness auditing to every model family, not just the LightGBM baseline. The worst-case verdict propagation is consistent with §7 Step 4 ("has this setup been walk-forward tested?") — the agent enforces freshness as a proxy for ongoing validity. No §10 anti-patterns are introduced.

### 2. Codebase conventions
minor — Three small gaps relative to CLAUDE.md:

1. **Structlog missing from `agents/knowledge_registry.py`**: The plan describes a new module but does not mention adding `log = structlog.get_logger(__name__)`. Every module is expected to get its own logger (CLAUDE.md "Logging" section). The lazy-import loop in `build_default_registry` should log at DEBUG level when a strategy import is skipped due to a missing optional dependency (e.g., torch).
2. **Module docstring not called out**: CLAUDE.md "File Organization" requires a module docstring at the top of every file explaining purpose and relevant env vars. The plan lists env vars (`artefact_env`) at the `ModelEntry` field level but does not explicitly call for a module-level docstring in `agents/knowledge_registry.py`.
3. All other conventions are correctly followed: `UPPER_CASE` for `MODEL_ENTRY` constants, `snake_case` for helpers, env-var-driven paths (never hardcoded), no direct yfinance or raw SQLite usage, and no concrete adapter imports in business logic.

### 3. Test coverage & CI gates
aligned — The plan provides comprehensive test coverage: a new `tests/test_knowledge_registry.py` with four targeted tests, plus four additional cases in `tests/test_knowledge_agent.py`. The autouse fixture that restricts the default registry to `lgbm_alpha` is a sound backwards-compatibility mechanism that prevents churn across 35+ existing test calls. All tests are network-free unit tests (no `@pytest.mark.integration` required). The 76% floor is explicitly acknowledged in the acceptance checklist. Verification step 3 invokes `/pre-push` for the full CI mirror (ruff + bandit HIGH + pip-audit).

### 4. Risk & backtest sequence
n/a — This plan introduces no new trading strategy, no position sizing change, and no new backtest. The Backtest → Walk-Forward → Monte Carlo → Paper Trade → Live sequence (TRADING_PHILOSOPHY.md §5) is not applicable. The one downstream risk effect — a `retrain` verdict from a stale zoo member propagating through `MetaAgent`'s Kelly multiplier — is correctly identified in the "Risks & mitigations" table as an intentional, documented behaviour change, not a silent sizing escalation. Kelly fraction remains half-capped at ≤0.25 in `risk/portfolio_risk.py:kelly_fraction`; nothing in this plan bypasses that ceiling.

## Recommended edits

- Add `import structlog; log = structlog.get_logger(__name__)` to `agents/knowledge_registry.py` and emit a `log.debug("skipping strategy import", module=..., reason=str(exc))` inside the `try/except` block in `build_default_registry` so skipped entries are observable in production logs.
- Add a module docstring to `agents/knowledge_registry.py` listing purpose, the `DEFAULT_REGISTRY` constant, and the env vars each entry may override (e.g., `LGBM_ALPHA_MODEL_PATH`, `BAYES_ALPHA_MODEL_PATH`, etc.).
- Consider adding `.env.example` entries for all new `artefact_env` variables (`BAYES_ALPHA_MODEL_PATH`, `RIDGE_ALPHA_MODEL_PATH`, `MLP_ALPHA_MODEL_PATH`, `CNN_ALPHA_MODEL_PATH`, `DL_LSTM_MODEL_PATH`, `RF_LONG_SHORT_MODEL_PATH`) so operators know they are configurable. The plan updates `MAINTENANCE_AND_BROKERS.md` §11.7 but does not mention `.env.example`.

## Anti-patterns flagged

- None — no full-Kelly sizing, no walk-forward bypass, no unconfirmed signals, no direct yfinance imports, no hardcoded secrets, no over-fit backtest concerns. TRADING_PHILOSOPHY.md §10 rows are clean.
