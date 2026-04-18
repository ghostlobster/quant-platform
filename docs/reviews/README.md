# Plan reviews

Durable audit trail for implementation-plan reviews run by the
[`trading-philosophy-reviewer`](../../.claude/agents/trading-philosophy-reviewer.md)
sub-agent.

## Naming

```
YYYY-MM-DD-<slug>.md
```

- Date is UTC, matching the session wall-clock at review time.
- Slug is a short kebab-case summary of the plan title (first 5 words of the
  plan heading when no explicit title is provided).
- Collisions on the same day get a `-2`, `-3`, … suffix appended before `.md`.

## Record template

Every record follows the same four-dimension structure so reviews are easy
to diff over time:

```markdown
# Plan review — <title>

- **Date:** YYYY-MM-DD (UTC)
- **Reviewer:** trading-philosophy-reviewer
- **Target:** <path or short description>
- **Overall:** <aligned | minor | major>

## Findings

### 1. Trading philosophy
<verdict> — <1-3 sentence rationale; cite TRADING_PHILOSOPHY.md §N>

### 2. Codebase conventions
<verdict> — <rationale; cite CLAUDE.md section>

### 3. Test coverage & CI gates
<verdict> — <rationale>

### 4. Risk & backtest sequence
<verdict> — <rationale>

## Recommended edits

- <bullet list of concrete changes the plan author should apply>

## Anti-patterns flagged

- <bullet list referencing TRADING_PHILOSOPHY.md §10 rows, or "none">
```

## Workflow

1. Draft a plan (e.g., in `/root/.claude/plans/<slug>.md`).
2. Run `/review-plan <plan-path>`. The slash command dispatches the
   sub-agent, which writes a record here and returns a 5-line summary.
3. Address any `major` findings in the plan before implementation. `minor`
   items should be acknowledged but are non-blocking.

The reviewer is advisory. It never edits plan or source files and never
runs the CI gates — that is what the `/pre-push` skill does. Reviews are
kept in this directory permanently so future sessions can see the
historical gate decisions.
