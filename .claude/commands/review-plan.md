---
description: Audit a draft implementation plan against TRADING_PHILOSOPHY.md and CLAUDE.md. Delegates to the `trading-philosophy-reviewer` subagent, which writes a durable record under docs/reviews/ and returns a compact verdict. Advisory — never blocks.
argument-hint: <plan-path>
tools: []
---

Delegate to the `trading-philosophy-reviewer` subagent. Pass the plan path
through verbatim — the subagent handles reading, review, and record-writing.

## Arguments

- `<plan-path>`: absolute path to the plan file (e.g.,
  `/root/.claude/plans/<slug>.md`), a PR diff file, or a file containing an
  inline plan.

If the argument is missing, print the usage line and stop; **do not**
dispatch to the subagent:

```
/review-plan <plan-path>
```

## Dispatch

Launch the `trading-philosophy-reviewer` subagent with a self-contained prompt:

> Review the plan at `<plan-path>` against `TRADING_PHILOSOPHY.md` and
> `CLAUDE.md`. Write the record under `docs/reviews/` per your definition
> and return the 5-line summary. Include the full record only if the user
> asks for it.

The subagent handles all file reads, dimension checks, record-writing, and
error reporting. See `.claude/agents/trading-philosophy-reviewer.md` for the
full interface.

## After the subagent returns

Forward the 5-line summary to the user verbatim, plus the record path so
the user can open it. If the user then asks for "full review" or "raw
record", re-invoke the subagent with an explicit full-output request — do
not attempt to reconstruct the payload from memory.

## Notes

- The reviewer is advisory. It never edits plan or source files and never
  runs the CI gates (that is `/pre-push`'s job).
- Records are committed to the repo under `docs/reviews/` so every plan has
  a durable audit trail. The directory is kept via a `README.md`.
- Invocation stays explicit — there is intentionally no settings hook that
  auto-dispatches this command. Sessions that want an auto-review can call
  `/review-plan` themselves after writing a plan.
