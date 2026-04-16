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
