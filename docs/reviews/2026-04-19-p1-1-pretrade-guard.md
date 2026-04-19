# Plan review — P1.1 Pre-trade risk guard + kill-switch

- **Date:** 2026-04-19 (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** /root/.claude/plans/p1-1-pretrade-guard.md
- **Overall:** minor

## Findings

### 1. Trading philosophy
aligned — The plan is a direct embodiment of Pillar 2 (Risk-First Thinking): every order must pass a deterministic gate before placement, enforcing `MAX_DAILY_LOSS_PCT`, `MAX_GROSS_EXPOSURE`, and `MAX_POSITION_PCT` limits. This operationalises §6's "VaR Budget → Daily loss limit constraint" and §7 Step 3 (Risk Sizing) at the execution layer. No anti-patterns from §10 are introduced. The kill-switch provides a manual circuit breaker consistent with the §8 flow chart's "Circuit Breaker: drawdown > 20% → halt new trades" principle. The plan correctly treats this as execution-layer enforcement, not a replacement for the upstream Backtest → Walk-Forward → Monte Carlo → Paper sequence, which is left intact.

### 2. Codebase conventions
minor — The plan is broadly well-aligned with CLAUDE.md conventions. The new module is placed in `risk/`, business logic codes against `providers/broker.BrokerProvider` Protocol, secrets are read from env vars via `GuardLimits.from_env()`, and structlog is used (`log.warning("pretrade_guard_reject", ...)`). One minor gap: the plan's "Reuse" section references `utils/logger.get_logger()` as the logging entry point, but CLAUDE.md mandates `structlog.get_logger(__name__)` directly (the `utils/` helper is a thin wrapper). This is cosmetic but should be confirmed consistent. Additionally, the adapter wiring imports `risk.pretrade_guard` directly into each concrete adapter — this is acceptable since `risk/` is internal business logic, not an external adapter, but implementers should confirm the import direction does not create a circular dependency through `providers/broker.BrokerProvider`.

### 3. Test coverage & CI gates
aligned — The plan specifies nine distinct unit tests in `tests/test_pretrade_guard.py`, covering all guard dimensions (blocklist, position pct, gross exposure, daily loss, order count, killswitch, decorator path, unset-limits pass-through, and `from_env` parsing). Tests use a `FakeBroker` fixture and a `tmp_path` SQLite journal — no network required. The verification section explicitly runs `--cov-fail-under=76` to confirm the coverage floor is maintained. The plan does not introduce integration-only code paths that would need `@pytest.mark.integration` gating. CI gates (ruff, bandit HIGH, pip-audit) are covered by the `/pre-push` invocation listed in the Verification section.

### 4. Risk & backtest sequence
aligned — This plan does not introduce a new strategy and does not bypass the Backtest → Walk-Forward → Monte Carlo → Paper → Live sequence; it adds an execution-layer safety net that operates after that sequence concludes. Kelly fraction is not touched (the existing half-Kelly cap at 25% in `risk/portfolio_risk.py` remains unchanged). VaR/CVaR budgets are honoured: `MAX_DAILY_LOSS_PCT` and `MAX_GROSS_EXPOSURE` map directly to §6's VaR Budget constraint and §7's Risk Sizing step. ATR-based stops are out of scope here (handled per §7 Step 3 upstream) and correctly deferred to P1.3/P1.7. One minor note: the plan does not explicitly mention that `MAX_DAILY_LOSS_PCT` is evaluated against CVaR-style realised+unrealised P&L — the design prose describes this correctly, but the implementation should confirm the unrealised component uses current market prices (not cost basis alone) to avoid understating exposure intraday.

## Recommended edits

- Confirm `utils/logger.get_logger()` is a transparent alias for `structlog.get_logger(__name__)` or replace the "Reuse" reference with the canonical `import structlog; log = structlog.get_logger(__name__)` pattern per CLAUDE.md.
- Add a note to the test plan asserting that `test_max_daily_loss_halts` uses current market prices for unrealised P&L (not just cost-basis fill), to make the intraday loss calculation unambiguous in CI.
- Consider a brief comment in `GuardLimits.from_env()` that `MAX_DAILY_LOSS_PCT` is a fraction (e.g. `0.05` = 5%), not a percentage integer, to prevent mis-configuration by operators.
- Verify that importing `risk.pretrade_guard` inside each broker adapter does not create a circular import via `providers/broker.BrokerProvider` (the Protocol is referenced for type hints in the guard). A `TYPE_CHECKING` guard may be needed.

## Anti-patterns flagged

- None. The plan does not introduce full-Kelly sizing, skip walk-forward validation, use unconfirmed signals, import yfinance directly, ignore VaR/CVaR budgets, or over-fit any backtest. All §10 anti-patterns are absent.
