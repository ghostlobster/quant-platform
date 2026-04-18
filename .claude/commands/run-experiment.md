---
description: Run a single phase of an ML experiment (ic, tune, or wf) via the experiment-tracker subagent and return a compact summary. Detailed artifacts remain in data/wf_history.db.
argument-hint: ic|tune|wf <TICKER> --ticket <REF> [--feature NAME]
tools: []
---

Delegate to the `experiment-tracker` subagent with the parsed arguments. This
keeps verbose Optuna trial logs and per-fold walk-forward tables out of the
main conversation — the subagent returns only the compact summary block
documented in its definition.

## Arguments

- `<phase>` (required): one of `ic`, `tune`, `wf`.
- `<ticker>` (required): symbol matching `^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$` (same
  regex as `/run-agent`).
- `--ticket <REF>` (required): ticket reference (`#123`, `QP-45`, `issue-87`).
  The subagent enforces this too, but reject early at the command layer for a
  cleaner error.
- `--feature NAME` (optional, `ic` phase only): which feature in
  `data/features.py` to measure.

If any required argument is missing or malformed, print the usage line and
stop; **do not** dispatch to the subagent:

```
/run-experiment <ic|tune|wf> <TICKER> --ticket <REF> [--feature NAME]
```

## Dispatch

Launch the `experiment-tracker` subagent with a self-contained prompt:

> Run the `<phase>` phase on ticker `<TICKER>`, ticket `<REF>`
> (feature=`<NAME>` if provided). Return the compact summary block documented
> in your definition. Do not modify feature, strategy, or backtester code.

The subagent handles module dispatch, artifact recording in
`data/wf_history.db`, and failure reporting. See
`.claude/agents/experiment-tracker.md` for the full interface.

## After the subagent returns

Forward the summary block to the user verbatim. If the user asks for
per-trial detail or per-fold tables, point them at `data/wf_history.db`
(the subagent printed the `run_id`) rather than re-running the phase.

## Notes

- For an end-to-end experiment (feature add → IC → wire → tune → backtest),
  use the `/ml-experiment` skill instead; it chains the phases and handles
  the edit steps the subagent deliberately does not perform.
- `python -c:*` is pre-approved in `.claude/settings.json`, so the subagent
  runs without permission prompts.
