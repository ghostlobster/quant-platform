# Plan review — Issue #118 — Covariate-shift detector (PSI/KS) feeding KnowledgeAdaptionAgent

- **Date:** 2026-04-18 (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** `/root/.claude/plans/issue-118-drift-detector.md`
- **Overall:** aligned

## Findings

### 1. Trading philosophy
aligned — The plan directly strengthens Pillar 3 (Humility Through Testing) by adding an early-warning leading indicator for model degradation, catching covariate shift before IC collapses and trades are damaged. It explicitly references AFML Ch 17 (structural breaks / PSI/KS) and frames the detector as a pre-live safeguard within the existing `KnowledgeAdaptionAgent` verdict ladder. The decision stack (§7) is not bypassed; the drift verdict feeds into existing retrain/monitor rungs rather than short-circuiting any stage. No anti-patterns from §10 are triggered — the plan introduces no new position-sizing logic, does not skip walk-forward, and does not add unconfirmed signal paths.

### 2. Codebase conventions
aligned — All conventions are respected:
- `analysis/drift.py` is a pure-function analysis module with no DB access, consistent with how `analysis/factor_ic.py` and `analysis/structural_breaks.py` are structured.
- DB writes go through `data/db.py:get_connection()` and use `INSERT OR REPLACE` (UPSERT) as required; the new `model_feature_stats` table is added via `init_db()` in `data/db.py`.
- The plan explicitly flags `_write_feature_stats` should mirror the `_write_metadata` pattern (structlog `log.warning` on exception, `with conn:` atomic block).
- `analysis/drift.py` functions carry no global state; thread-safety is explicitly addressed.
- No direct yfinance imports, no hardcoded secrets, no `print` statements are introduced.
- Naming follows snake_case / UPPER_CASE / `_leading_underscore` conventions throughout (`_DRIFT_PSI_MONITOR`, `_resolve_drift`, `_read_feature_stats`).
- One minor observation: the plan does not explicitly call out adding a module docstring to `analysis/drift.py` — CLAUDE.md requires "Module docstring at top of every file explaining purpose and relevant env vars." This is easily addressed at implementation time and does not constitute a gap in the plan design.

### 3. Test coverage & CI gates
aligned — The plan specifies 11 unit tests in `tests/test_drift.py`, 3 new cases in `tests/test_knowledge_agent.py`, and 1 new case in `tests/test_ml_signal.py` — 15 total new cases across three existing/new test files, all using synthetic numpy/pandas data with no network calls. Tests are deterministic via `np.random.default_rng(seed)`. The acceptance checklist explicitly calls out the 76% coverage floor. `@pytest.mark.integration` is not needed here (no live credentials required). The verification sequence (`ruff` → `pytest` targeted → `/pre-push`) mirrors the CI pipeline. No CI gates are skipped or weakened.

### 4. Risk & backtest sequence
n/a — This plan introduces no new trading strategy, no new position-sizing logic, and no new backtesting pipeline. The drift detector is a model-health diagnostic that feeds the existing `KnowledgeAdaptionAgent` verdict ladder (retrain/monitor). It does not touch Kelly fraction, VaR/CVaR budgets, ATR stops, or the Backtest → Walk-Forward → Monte Carlo → Paper Trade → Live sequence. The plan explicitly scopes out label-column drift and per-regime drift slices. No risk dimension anti-patterns are implicated.

## Recommended edits

- Add an explicit note in the plan (or implementation task list) to include a module docstring at the top of `analysis/drift.py` documenting its purpose and noting that it has no env-var dependencies — consistent with CLAUDE.md's "Module docstring at top of every file" convention.
- Consider noting in `MAINTENANCE_AND_BROKERS.md` §11.6 that the PSI thresholds (`_DRIFT_PSI_MONITOR=0.10`, `_DRIFT_PSI_RETRAIN=0.25`) are hard-coded constants and an env-var override ticket should be opened for operator tuning (the plan acknowledges this as out-of-scope but does not cross-reference a follow-up issue number).
- The `_resolve_drift` fallback is documented as "fail-open" (returns `None` when no feature frame is available). Confirm in the implementation that the `log.warning` is emitted when the DB read returns `None` but a `context["feature_frame"]` was supplied, so operators can detect misconfiguration silently dropping drift checks.

## Anti-patterns flagged

- none — No anti-patterns from TRADING_PHILOSOPHY.md §10 are present in this plan.
