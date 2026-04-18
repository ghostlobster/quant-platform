# Plan review — Issue #121: pages/model_health.py Streamlit tab

- **Date:** 2026-04-18 (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** `/root/.claude/plans/issue-121-model-health-page.md`
- **Overall:** minor

## Findings

### 1. Trading philosophy
aligned — This is a read-only observability dashboard. It surfaces the
`KnowledgeAdaptionAgent` verdict and the sizing multipliers (fresh=1.0,
monitor=0.7, retrain=0.4) that are directly downstream of Kelly-fraction
discipline (§6, §7 Step 3). The plan does not introduce any new trading
logic, signal, or sizing path, so §7's decision-stack order and the
three pillars (§1) are not at risk. The page makes the risk-first
thinking pillar more visible, not less.

### 2. Codebase conventions
minor — The plan is largely well-aligned: `data/db.py:get_connection()`
is explicitly required for all SQL (satisfying the CLAUDE.md "Common
Pitfalls" rule), `plotly` charts follow the repo standard, `st.cache_data`
wraps blocking helpers, and no OHLCV data is fetched (so `data/fetcher.py`
is irrelevant here). Two small gaps:

1. **No structlog logger declared.** The plan shows no `structlog.get_logger(__name__)` in `pages/model_health.py`. Every production-path module must have one per CLAUDE.md "Logging" section. The `try/except` error-handling blocks in each panel should log the exception via `log.warning(...)` before calling `st.warning()`, not just render a banner silently.

2. **Import of private helper.** The plan explicitly imports `_read_regime_coverage` and `_confine_pickle_path` from `agents/knowledge_agent.py` (leading-underscore names). The plan acknowledges this coupling but does not propose a resolution path. Per CLAUDE.md naming conventions, private helpers (`_leading_underscore`) are module-private. The plan's own risk table suggests noting this for future promotion to public — that note should appear in the page docstring as a `# TODO` so it is actionable, not just acknowledged in the plan prose.

### 3. Test coverage & CI gates
minor — Three smoke test cases in `tests/test_pages.py::TestPagesModelHealth`
are specified and cover the key fallback branches (empty DB, synthetic
metadata, live-IC warm-up). The plan invokes `/pre-push` as its final
verification step, which enforces the 76% coverage floor, ruff, bandit
HIGH, and pip-audit. However, the plan adds a new source module
(`pages/model_health.py`) and appends to `app.py` but names no
dedicated `tests/test_model_health.py` file — it folds its tests into
the existing `tests/test_pages.py`. CLAUDE.md states "one test file per
source module." Folding into `test_pages.py` is consistent with sibling
pages practice (the plan cites `_make_st_mock()` from there), but the
convention mismatch is worth calling out. No new external network calls
are introduced, so mock-requirement and integration-mark rules are
satisfied.

### 4. Risk & backtest sequence
n/a — This plan introduces no strategy, no sizing logic, no backtest
run, and no live-trading path. It is a pure display layer that reads
already-computed verdicts from the DB. The Kelly multiplier table it
renders is computed by `KnowledgeAdaptionAgent`, not by this page. The
backtest → walk-forward → Monte Carlo → paper → live sequence (§5) is
therefore not applicable.

## Recommended edits

- Add `import structlog` and `log = structlog.get_logger(__name__)` to `pages/model_health.py`; thread `log.warning("panel load failed", panel="inventory", exc_info=exc)` (or equivalent) into each `except` block before the `st.warning()` call.
- Replace the prose acknowledgement of the private-helper coupling with a `# TODO(public-api): promote _read_regime_coverage and _confine_pickle_path to public when refactoring agents/knowledge_agent.py` comment in the page's module docstring.
- Consider either adding a thin `tests/test_model_health.py` stub (even if it just imports the `TestPagesModelHealth` class to satisfy the one-file-per-module convention) or adding an explicit note in the plan acknowledging the deliberate deviation from CLAUDE.md convention.

## Anti-patterns flagged

- none — The plan does not introduce any of the §10 anti-patterns. No full-Kelly sizing, no unconfirmed signals, no direct yfinance imports, no hardcoded secrets, no VaR/CVaR bypass.
