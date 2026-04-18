---
description: Manually run a cron job (cron.daily_ml_execute or cron.monthly_ml_retrain) locally with optional ticker override, and summarise the structured log output. Delegates to the `cron-runner` subagent so training/trade logs stay out of the main context.
argument-hint: daily|monthly [TICKER1,TICKER2,...]
---

Delegate to the `cron-runner` subagent with the parsed arguments. This keeps
verbose training/trade logs out of the main conversation — the subagent
returns only the parsed summary of the final structured log line.

## Arguments

- First arg (required): `daily` or `monthly`.
- Second arg (optional): comma-separated ticker list (forwarded as
  `WF_TICKERS` for the subprocess only; never mutates the user's shell).

Reject anything else with:

```
/run-cron daily|monthly [TICKER1,TICKER2,...]
```

## Dispatch

Launch the `cron-runner` subagent with a self-contained prompt:

> Run `<mode>` (tickers=<value or "default">). Perform the documented
> pre-flight checks, surface any env vars that differ from defaults, and
> return the compact summary block.

The subagent enforces the "never switch `BROKER_PROVIDER` away from paper
without confirmation" guardrail and handles failure reporting. See
`.claude/agents/cron-runner.md` for the full interface.

## After the subagent returns

Forward the summary block to the user. If the subagent reports a missing
model, suggest running `/run-cron monthly` first with the same tickers.

## Guardrails (enforced by the subagent)

- Does not switch `BROKER_PROVIDER` away from `paper` without explicit
  confirmation — a mis-set broker could place real orders.
- Does not modify anything under `cron/` or `strategies/`.
- Does not commit, push, or tag anything.
