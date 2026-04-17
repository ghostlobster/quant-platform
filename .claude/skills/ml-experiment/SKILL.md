---
name: ml-experiment
description: Guide the user end-to-end through a new ML signal or feature experiment — feature engineering in data/features.py, IC measurement via analysis/factor_ic.py, wiring into strategies/ml_signal.py or ensemble_signal.py, Optuna tuning via strategies/ml_tuning.py, and a backtest via backtester/walk_forward.py. Use when the user wants to experiment with a new alpha, add a feature, try a new ML signal, or tune an existing model.
---

# ml-experiment — scaffold a new ML signal experiment

The platform already has a full ML pipeline. This skill does **not** re-implement any of it — it chains the existing modules and emits a concise experiment report. Always reuse; never duplicate.

## Pipeline modules (reuse these, do not rewrite)

| Stage | Module | Key API |
|---|---|---|
| Features | `data/features.py` | add a function here; the feature list is assembled inside the module |
| IC / ICIR | `analysis/factor_ic.py` | `compute_ic(df, feature, horizon)` |
| Baseline model | `strategies/ml_signal.py` | `MLSignal().train(tickers, period)`, `MLSignal().predict(tickers, period)` — LightGBM + regime-conditioning |
| Ensemble | `strategies/ensemble_signal.py` | use when stacking multiple signals |
| Tuning | `strategies/ml_tuning.py` | `run_optuna(...)`, `load_best_params(model_name)` — persisted params are picked up automatically by `cron/monthly_ml_retrain.py` |
| Walk-forward backtest | `backtester/walk_forward.py` | `run_walk_forward(...)` |
| Tests | `tests/test_ml_signal.py`, `tests/test_features.py` | mock `fetch_ohlcv`; no network |

## Steps

Walk the user through these, one at a time. At each step, print what you changed and what command you ran.

1. **Clarify scope.** Ask: what feature or signal idea, and on which ticker universe? Default universe is `$WF_TICKERS` or `SPY,QQQ,AAPL,MSFT,TSLA` (matches `cron/daily_ml_execute.py:35`).
2. **Add the feature.** Edit `data/features.py` to add the feature function; register it in the existing feature list. Add or extend a unit test in `tests/test_features.py` that mocks OHLCV — do not hit network.
3. **Run the feature test.** `pytest tests/test_features.py -v`.
4. **Measure IC.** Compute `compute_ic` on the configured universe across sensible horizons (1d / 5d / 20d). Convention from `IMPLEMENTATION_SUMMARY.md`: if `|IC| < 0.02` and ICIR is small, reject the feature and stop. Report the IC table.
5. **Wire into a signal.** If a pure feature, add to `MLSignal`. If combining signals, extend `ensemble_signal.py`. Do **not** modify `cron/monthly_ml_retrain.py` or `cron/daily_ml_execute.py` — they consume whatever `MLSignal` and `load_best_params` return.
6. **Tune.** Call `strategies.ml_tuning.run_optuna(...)` for N trials (ask; default 30) with purged CV. Persist winners so `monthly_ml_retrain` picks them up on the next schedule.
7. **Walk-forward backtest.** `run_walk_forward` on the configured universe; report Sharpe, Sortino, max drawdown, turnover.
8. **Report.** Emit a one-screen experiment report:

   ```
   Experiment: <feature/signal name>
   Universe:   <tickers>
   IC (5d):    <value>   ICIR: <value>
   Tuned params: <json>
   Backtest:   Sharpe=<>, Sortino=<>, MaxDD=<>, Turnover=<>
   Recommendation: <promote to production / further work / reject>
   ```

## Guardrails

- Never modify anything under `cron/`. The experiment ends at "tuned params saved + backtest reported"; the cron jobs pick up tuned params on their next run.
- Never hit the network in tests. Mock `data.fetcher.fetch_ohlcv` as other tests in `tests/` do.
- Keep ruff / pytest / coverage green — after each step, propose running the `pre-push` skill if the user wants to validate before moving on.
- Never commit or push from this skill.

## Verification before declaring done

Run the `pre-push` skill (or tell the user to). A passing `pre-push` + a backtest report with Sharpe above the current live baseline is the bar for promoting the experiment.
