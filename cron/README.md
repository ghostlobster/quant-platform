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
