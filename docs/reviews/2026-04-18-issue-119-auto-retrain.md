# Plan review — Issue #119: Opt-in auto-retrain trigger from KnowledgeAdaptionAgent

- **Date:** 2026-04-18 (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** `/root/.claude/plans/issue-119-auto-retrain.md`
- **Overall:** minor

## Findings

### 1. Trading philosophy
aligned — The plan does not introduce a new trading strategy, modify position sizing, or alter signal/confirmation/exit logic, so §1 pillars and §7 decision stack are not directly in scope. The auto-retrain mechanism supports Pillar 3 (Humility Through Testing) by tightening the feedback loop between a degraded ML model and retraining, reducing the window of reduced-Kelly trading. No trading decisions are automated by this change; the retrain is a model-refresh operation, not a trade execution. No §10 anti-patterns are triggered.

### 2. Codebase conventions
minor — Three small gaps noted:

1. **Logging style inconsistency.** The plan's `_maybe_auto_retrain` snippet uses `logger.warning("knowledge_agent: auto-retrain launch failed: %s", exc)` — the old `%s`-style positional format. The existing module and CLAUDE.md both mandate structlog keyword-argument style: `logger.warning("...", error=str(exc))`. The `_watch_retrain_subprocess` watcher correctly uses the keyword style (`pid=proc.pid, exit_code=rc`), so this is an oversight in one code block, not a pattern failure.

2. **`.env.example` not updated.** The plan adds two new env vars (`KNOWLEDGE_AUTO_RETRAIN`, `KNOWLEDGE_RETRAIN_COOLDOWN`) and documents them in the module docstring and `MAINTENANCE_AND_BROKERS.md`, but does not explicitly call out updating `.env.example`. CLAUDE.md requires all env vars to be present in `.env.example` with placeholder values.

3. **Alert channel access goes through `providers/alert.py`.** `_maybe_launch_alert` correctly uses `from providers.alert import get_alert_channel` — this is fully aligned with the DI convention. No direct adapter import. No concerns here.

All other conventions are correctly followed: `data/db.py:get_connection()` is used for `quant.db` access; no direct yfinance imports; no hardcoded secrets; structlog is the chosen logger (`utils/logger.py` wraps `structlog.get_logger`); injection points via class staticmethods preserve testability.

### 3. Test coverage & CI gates
aligned — Seven named unit tests are planned for `tests/test_knowledge_agent.py`, covering: disabled-by-default, single-fire within cooldown, refire after cooldown, non-retrain verdict, subprocess error resilience, cross-instance stamp persistence (via temp SQLite), and alert subject differentiation. An additional round-trip unit test for `_read_stamp`/`_write_stamp` is included. All tests monkeypatch `_popen`, `_retrain_reader`, and `_retrain_writer`, so no network or real subprocess is required. The plan explicitly acknowledges the 76% coverage floor and expects the new tests to increase coverage. No integration marker is needed for these tests. The plan mandates `/pre-push` (ruff + pytest 76% + bandit HIGH + pip-audit) before merging.

### 4. Risk & backtest sequence
n/a — This plan does not introduce a new strategy, modify sizing, alter stop logic, or touch VaR/CVaR budgets. The retrain operation itself (delegated to `cron.monthly_ml_retrain`) has its own established validation sequence. The 24h stamp cooldown prevents runaway retrains. Kelly fraction behaviour is unchanged; the plan's stated motivation is to shorten the period of reduced-Kelly trading caused by a stale model, which is consistent with risk-first thinking (§1 Pillar 2). The non-atomic stamp race condition is explicitly acknowledged in the risks table and deferred with a concrete remediation path noted.

## Recommended edits

- In `_maybe_auto_retrain`, change `logger.warning("knowledge_agent: auto-retrain launch failed: %s", exc)` to `logger.warning("knowledge_agent: auto-retrain launch failed", error=str(exc))` to conform to the structlog keyword-argument convention used throughout the module.
- Add `KNOWLEDGE_AUTO_RETRAIN=` and `KNOWLEDGE_RETRAIN_COOLDOWN=` (with blank/commented placeholder values) to `.env.example`, alongside the existing `KNOWLEDGE_ALERT_COOLDOWN` and `KNOWLEDGE_HEALTH_CRON` entries.
- Consider noting in `cron/README.md` the env var table (lines 209-212) should also include `KNOWLEDGE_AUTO_RETRAIN` and `KNOWLEDGE_RETRAIN_COOLDOWN` for operator discoverability — the plan only mentions adding a single cross-reference bullet, which may be insufficient for operators consulting the cron env-var table.

## Anti-patterns flagged

- none — No §10 anti-patterns apply. No new strategy is introduced; no position sizing is changed; no walk-forward is skipped; no direct yfinance import; no unconfirmed signal used to trade.
