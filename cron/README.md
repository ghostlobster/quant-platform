# Monthly Walk-Forward Cron

Runs a walk-forward backtest across a configurable set of tickers on the first of each month and persists results to `data/wf_history.db`.

## Manual run

```bash
# From project root, with the venv active:
python -m cron.monthly_wf
```

Override tickers via environment variable:

```bash
WF_TICKERS="SPY,QQQ,GLD" python -m cron.monthly_wf
```

## Crontab entry

Run at 06:00 on the 1st of every month:

```cron
0 6 1 * * cd /home/<user>/projects/quant-platform && /home/<user>/projects/quant-platform/.venv/bin/python -m cron.monthly_wf >> /var/log/quant-platform/monthly_wf.log 2>&1
```

Add with:

```bash
crontab -e
```

## Output

Results are upserted into `data/wf_history.db`, table `wf_results`:

| Column            | Type    | Description                             |
|-------------------|---------|-----------------------------------------|
| id                | INTEGER | Auto-increment PK                       |
| run_date          | TEXT    | ISO date (YYYY-MM-DD)                   |
| ticker            | TEXT    | Ticker symbol                           |
| consistency_score | REAL    | Fraction of windows with positive return |
| total_return      | REAL    | Average return across windows           |
| n_windows         | INTEGER | Number of walk-forward windows run      |

Re-running on the same date **overwrites** existing rows for that date (upsert on `run_date, ticker`).

## Environment variables

| Variable    | Default                      | Description              |
|-------------|------------------------------|--------------------------|
| WF_TICKERS  | SPY,QQQ,AAPL,MSFT,TSLA       | Comma-separated tickers  |
| LOG_LEVEL   | INFO                         | Logging verbosity        |

---

# Monthly ML Model Retraining Cron

Retrains the LightGBM alpha model on the 1st of each month and saves a fresh
checkpoint to `models/lgbm_alpha.pkl` (or the path set by `LGBM_ALPHA_MODEL_PATH`).

## Manual run

```bash
python -m cron.monthly_ml_retrain
```

Override tickers or training period:

```bash
WF_TICKERS="AAPL,MSFT,GOOG" ML_TRAIN_PERIOD="1y" python -m cron.monthly_ml_retrain
```

## Crontab entry

Run at 07:00 on the 1st of every month (after walk-forward at 06:00):

```cron
0 7 1 * * cd /home/<user>/projects/quant-platform && /home/<user>/projects/quant-platform/.venv/bin/python -m cron.monthly_ml_retrain >> /var/log/quant-platform/monthly_ml_retrain.log 2>&1
```

## Output

On success the job logs `ml_retrain: complete` with the following metrics:

| Field        | Description                              |
|--------------|------------------------------------------|
| train_ic     | Spearman IC on training split            |
| test_ic      | Spearman IC on held-out test split       |
| train_icir   | IC / IC-std on training split            |
| test_icir    | IC / IC-std on test split                |
| n_train      | Number of training samples               |
| n_test       | Number of test samples                   |

The checkpoint is also recorded in `quant.db` → `model_metadata` table.

## Environment variables

| Variable              | Default                    | Description                        |
|-----------------------|----------------------------|------------------------------------|
| WF_TICKERS            | SPY,QQQ,AAPL,MSFT,TSLA    | Comma-separated tickers to train on|
| ML_TRAIN_PERIOD       | 2y                         | yfinance period for training data  |
| LGBM_ALPHA_MODEL_PATH | models/lgbm_alpha.pkl      | Path to save the model checkpoint  |

## Live IC backfill (#115)

`analysis/live_ic.py` persists every scored (ticker, model) row to the
`live_predictions` SQLite table at order time. A daily
`live_ic_backfill_job` (registered alongside `knowledge_health_job` in
`scheduler/alerts.py`) pulls realized forward-returns via
`data/fetcher.fetch_ohlcv` once each row's horizon has expired, then
`rolling_live_ic("lgbm_alpha")` feeds the IC back into
`KnowledgeAdaptionAgent` so the IC-degradation branch can actually fire.

Override the daily cadence with `LIVE_IC_BACKFILL_CRON` (default
`"30 4 * * *"`). Disable the writer by setting
`KNOWLEDGE_RECORD_PREDICTIONS=0` (useful in unit tests and dry-runs).

## Opt-in auto-trigger from KnowledgeAdaptionAgent (#119)

Setting `KNOWLEDGE_AUTO_RETRAIN=1` in the environment of whichever
process runs `KnowledgeAdaptionAgent` (the `#116` hourly APScheduler
job, the `MetaAgent` vote path, the `daily_ml_execute` circuit
breaker, or a one-off CLI invocation) makes that agent spawn
`python -m cron.monthly_ml_retrain` as a detached subprocess on the
first `retrain` verdict. A SQLite stamp in `knowledge_stamps` dedups
launches to **at most once per 24h** (tune via `KNOWLEDGE_RETRAIN_COOLDOWN`).
Off by default. See `MAINTENANCE_AND_BROKERS.md` for operator notes.

---

# Daily ML Signal Execution Cron

Scores the configured ticker universe with the trained LightGBM alpha model and
routes the resulting orders through the broker provider (paper by default).
Intended to run after the US market close on weekdays.

## Manual run

```bash
python -m cron.daily_ml_execute
```

Override threshold or cap positions:

```bash
ML_SCORE_THRESHOLD=0.4 ML_MAX_POSITIONS=3 python -m cron.daily_ml_execute
```

## Crontab entry

Run at 16:05 ET (post-close) on weekdays:

```cron
5 16 * * 1-5 cd /home/<user>/projects/quant-platform && /home/<user>/projects/quant-platform/.venv/bin/python -m cron.daily_ml_execute >> /var/log/quant-platform/daily_ml_execute.log 2>&1
```

## Environment variables

| Variable                | Default                    | Description                                           |
|-------------------------|----------------------------|-------------------------------------------------------|
| WF_TICKERS              | SPY,QQQ,AAPL,MSFT,TSLA    | Comma-separated tickers to score                       |
| ML_SCORE_THRESHOLD      | 0.3                        | Minimum \|score\| required to act                      |
| ML_MAX_POSITIONS        | 5                          | Maximum simultaneous long positions                    |
| BROKER_PROVIDER         | paper                      | paper / alpaca / ibkr / schwab (routed via providers/) |
| LGBM_ALPHA_MODEL_PATH   | models/lgbm_alpha.pkl      | Path to the trained model checkpoint                   |
| KNOWLEDGE_GATE_ENFORCE  | (unset)                    | When `1`, same as passing `--enforce-knowledge-gate`   |

Position sizes are computed as `equity × Kelly × regime_mult × |score|`, where
Kelly uses the priors `ML_KELLY_WIN_RATE` (0.55), `ML_KELLY_AVG_WIN` (0.03),
`ML_KELLY_AVG_LOSS` (0.02). Regime multiplier halves size in `high_vol`.

## Pre-trade circuit breaker

Optional flag that refuses to place orders when `KnowledgeAdaptionAgent`
returns a `retrain` verdict — intended for production so stale models do
not silently bleed P&L:

```bash
python -m cron.daily_ml_execute --enforce-knowledge-gate
# or equivalently
KNOWLEDGE_GATE_ENFORCE=1 python -m cron.daily_ml_execute
```

The CLI flag wins over the env var when both are set (use
`--no-enforce-knowledge-gate` to explicitly disable). Verdicts:

| Verdict    | Behaviour                                                |
|------------|----------------------------------------------------------|
| `fresh`    | Proceeds normally.                                       |
| `monitor`  | Proceeds normally; agent logs the reason.                |
| `retrain`  | **Exits with code 2** before any order is placed.        |

Exit codes emitted by the job: `0` success, `1` fatal error (missing
lightgbm, no trained model, unexpected exception), `2` knowledge gate
tripped. The agent's own 24h alert cooldown dedupes alerts so the cron
does not spam on every run.

---

# Knowledge Health Job (push-based verdict)

Runs `KnowledgeAdaptionAgent().run({})` on a cron schedule so staleness is
detected even on quiet days (no votes, no trade flow). Exposed as
`scheduler.alerts.knowledge_health_job` and scheduled via
`scheduler.alerts.start_knowledge_health_scheduler` (APScheduler
`BackgroundScheduler`). The agent's own 24h alert cooldown dedupes
retrain alerts, so scheduling hourly does **not** spam operators.

## Opt-in in the app

The Streamlit bootstrap starts the scheduler only when
`ENABLE_KNOWLEDGE_HEALTH_JOB=1`, so dev sessions do not get surprise
background jobs:

```bash
ENABLE_KNOWLEDGE_HEALTH_JOB=1 streamlit run app.py
```

## Manual run (one-shot)

```bash
python -c "from scheduler.alerts import knowledge_health_job; knowledge_health_job()"
# or, via the slash command
/run-cron knowledge-health
```

The Claude Code `/run-cron knowledge-health` dispatches to the
`cron-runner` subagent which emits the compact verdict summary.

## Environment variables

| Variable                      | Default         | Description                                        |
|-------------------------------|-----------------|----------------------------------------------------|
| `ENABLE_KNOWLEDGE_HEALTH_JOB` | (unset)         | `1` → app bootstrap starts the scheduler           |
| `KNOWLEDGE_HEALTH_ENABLED`    | `1`             | Kill-switch — set `0` to skip without re-deploy     |
| `KNOWLEDGE_HEALTH_CRON`       | `0 * * * *`     | Crontab-style schedule (top of every hour)         |
| `KNOWLEDGE_ALERT_COOLDOWN`    | `86400`         | Seconds between retrain alerts (inherited)         |

## Verdict → log level

| Recommendation | Log level | Structured payload                                      |
|----------------|-----------|---------------------------------------------------------|
| `fresh`        | INFO      | `recommendation, signal, confidence, reasoning`         |
| `monitor`      | INFO      | same                                                    |
| `retrain`      | WARNING   | same (agent fires its own alert with 24h cooldown)      |
