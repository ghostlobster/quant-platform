---
description: Manually run a cron job (cron.daily_ml_execute or cron.monthly_ml_retrain) locally with optional ticker override, and summarise the structured log output.
argument-hint: daily|monthly [TICKER1,TICKER2,...]
---

Manually trigger one of the ML cron jobs. Mirrors the manual-invocation lines documented in `cron/daily_ml_execute.py:9-11` and `cron/monthly_ml_retrain.py:7-11`.

## Arguments

- First arg (required): `daily` or `monthly`.
- Second arg (optional): comma-separated ticker list. If present, export `WF_TICKERS=<arg>` **for the subprocess only** — do not mutate the user's shell.

Reject any other first arg with a usage line: `/run-cron daily|monthly [TICKER1,TICKER2,...]`.

## Dispatch

| First arg | Command | Script |
|---|---|---|
| `daily` | `python -m cron.daily_ml_execute` | `cron/daily_ml_execute.py` |
| `monthly` | `python -m cron.monthly_ml_retrain` | `cron/monthly_ml_retrain.py` |

Both are pre-approved in `.claude/settings.json`.

## Pre-flight for `daily`

Before invoking, check that a trained model exists — otherwise `cron/daily_ml_execute.py:59-64` will exit 1:

```
python -c "import os; p=os.environ.get('LGBM_ALPHA_MODEL_PATH','models/lgbm_alpha.pkl'); import sys; sys.exit(0 if os.path.exists(p) else 1)"
```

If the model is missing, stop and tell the user to run `/run-cron monthly` first (optionally with the same tickers).

## Env vars to surface

`daily` also reads `ML_SCORE_THRESHOLD` (default 0.3), `ML_MAX_POSITIONS` (default 5), `BROKER_PROVIDER` (default `paper`) — all from `cron/daily_ml_execute.py:17-22`. Before running, print the current values and ask the user whether to override. Default to the existing env; never silently switch to a live broker.

`monthly` reads `ML_TRAIN_PERIOD` (default `2y`) from `cron/monthly_ml_retrain.py:14-17`. Surface the same way.

## Run

Stream output to the user. On success, parse the final structured log line and summarise:

- **daily**: extract `n_actions`, `n_scores`, and the `actions` list from the `daily_ml_execute: complete` log (fields match `cron/daily_ml_execute.py:75-79`). Show a one-screen summary.
- **monthly**: extract `train_ic`, `test_ic`, `train_icir`, `test_icir`, `n_train`, `n_test` from the `ml_retrain: complete` log (fields match `cron/monthly_ml_retrain.py:55-63`). Show them in a small table.

## On failure

If exit code is non-zero, show the last 20 lines of stderr and propose the next step:

- `lightgbm is not installed` → `pip install lightgbm`.
- `no trained baseline model found` → run `/run-cron monthly` first.
- `runtime error` during retraining → check `WF_TICKERS` and network; do not retry silently.

## Guardrails

- Do not switch `BROKER_PROVIDER` away from `paper` without explicit user confirmation — a mis-set broker could place real orders.
- Do not modify anything under `cron/` or `strategies/` from this command.
- Do not commit, push, or tag anything.
