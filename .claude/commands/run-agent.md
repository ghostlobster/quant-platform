---
description: Invoke a specialist trading agent (regime, risk, sentiment, screener, execution) or the meta-agent with a ticker context, and display the AgentSignal.
argument-hint: <agent_name> <ticker>
---

Invoke one of the trading agents defined in `agents/` with a minimal context and print the resulting `AgentSignal` (see `agents/base.py:15-48`).

## Arguments

- `<agent_name>`: one of `regime`, `risk`, `sentiment`, `screener`, `execution`, `meta` (case-insensitive).
- `<ticker>`: symbol, validated against `^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$` — same regex as `agents/meta_agent.py:25`.

Reject unknown agent names with a list of valid options. Reject malformed tickers with the regex.

## Dispatch

| agent_name | Class | Module |
|---|---|---|
| `regime` | `RegimeAgent` | `agents.regime_agent` |
| `risk` | `RiskAgent` | `agents.risk_agent` |
| `sentiment` | `SentimentAgent` | `agents.sentiment_agent` |
| `screener` | `ScreenerAgent` | `agents.screener_agent` |
| `execution` | `ExecutionAgent` | `agents.execution_agent` |
| `meta` | `MetaAgent` | `agents.meta_agent` |

## Execution

Run a one-shot Python invocation. For specialists:

```
python -c "
import json
from agents.<module> import <Class>
agent = <Class>()
sig = agent.run({'ticker': '<TICKER>'})
print(json.dumps({
    'agent': sig.agent_name,
    'signal': sig.signal,
    'confidence': sig.confidence,
    'reasoning': sig.reasoning,
    'metadata': sig.metadata,
}, indent=2, default=str))
"
```

For `meta`, `MetaAgent.run()` already returns a dict (see `agents/meta_agent.py:136-150`) — dump it directly:

```
python -c "
import json
from agents.meta_agent import MetaAgent
result = MetaAgent().run({'ticker': '<TICKER>'})
print(json.dumps(result, indent=2, default=str))
"
```

## Output

Show the JSON result as a formatted block. Do not edit files. If the agent raises, show the exception class and message; suggest checking the agent's dependencies (e.g., sentiment needs an LLM provider configured via `LLM_PROVIDER`).

## Notes

- `MetaAgent` reads `AGENT_WEIGHTS` and `AGENT_LLM_ARBITER` from the environment (`agents/meta_agent.py:9-11`). Do not override them from this command.
- `python -c:*` is pre-approved in `.claude/settings.json`, so this runs without a permission prompt.
