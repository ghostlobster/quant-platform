---
description: Invoke a specialist trading agent (regime, risk, sentiment, screener, execution, knowledge) or the meta-agent with a ticker context, and display the AgentSignal. Delegates to the `trading-agent-runner` subagent so large JSON payloads stay out of the main context.
argument-hint: <agent_name> <ticker>
---

Delegate to the `trading-agent-runner` subagent with the parsed arguments.
This keeps verbose agent output (full reasoning, metadata dumps) out of the
main conversation — the subagent returns only a 3-line summary unless the
user explicitly asks for raw JSON.

## Arguments

- `<agent_name>`: one of `regime`, `risk`, `sentiment`, `screener`,
  `execution`, `knowledge`, `meta` (case-insensitive).
- `<ticker>`: symbol matching `^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$` — same regex as
  `agents/meta_agent.py:25`.

If either argument is missing or malformed, print the usage line and stop;
**do not** dispatch to the subagent:

```
/run-agent <regime|risk|sentiment|screener|execution|knowledge|meta> <TICKER>
```

## Dispatch

Launch the `trading-agent-runner` subagent with a self-contained prompt:

> Run the `<agent_name>` agent on ticker `<TICKER>`. Return the compact
> 3-line summary documented in your definition. Include raw JSON only if the
> user explicitly asked for it.

The subagent handles module dispatch, execution, and failure reporting. See
`.claude/agents/trading-agent-runner.md` for the full interface.

## After the subagent returns

Forward the 3-line summary to the user verbatim. If the user then asks for
"full output" or "raw JSON", re-invoke the subagent with the explicit
raw-JSON request — do not attempt to reconstruct the payload.

## Notes

- `MetaAgent` reads `AGENT_WEIGHTS` and `AGENT_LLM_ARBITER` from the
  environment (`agents/meta_agent.py:9-11`). Do not override them from this
  command.
- `python -c:*` is pre-approved in `.claude/settings.json`, so the subagent
  runs without permission prompts.
