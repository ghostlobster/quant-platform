---
name: ml-experiment
description: Guide the user end-to-end through a new ML signal or feature experiment — feature engineering in data/features.py, IC measurement via analysis/factor_ic.py, wiring into strategies/ml_signal.py or ensemble_signal.py, Optuna tuning via strategies/ml_tuning.py, and a backtest via backtester/walk_forward.py. Use when the user wants to experiment with a new alpha, add a feature, try a new ML signal, or tune an existing model.
---

# ml-experiment — scaffold a new ML signal experiment

The platform already has a full ML pipeline. This skill does **not** re-implement any of it — it chains the existing modules via the `experiment-tracker` subagent and emits a concise experiment report. Always reuse; never duplicate.

## Pipeline modules (reuse these, do not rewrite)

| Stage | Module | Key API |
|---|---|---|
| Features | `data/features.py` | add a function here; the feature list is assembled inside the module |
| IC / ICIR | `analysis/factor_ic.py` | measured via the `experiment-tracker` subagent (`ic` phase) |
| Baseline model | `strategies/ml_signal.py` | `MLSignal().train(tickers, period)`, `MLSignal().predict(tickers, period)` — LightGBM + regime-conditioning |
| Ensemble | `strategies/ensemble_signal.py` | use when stacking multiple signals |
| Tuning | `strategies/ml_tuning.py` | measured via the `experiment-tracker` subagent (`tune` phase); persisted winners are picked up automatically by `cron/monthly_ml_retrain.py` |
| Walk-forward backtest | `backtester/walk_forward.py` | run via the `experiment-tracker` subagent (`wf` phase) |
| Tests | `tests/test_ml_signal.py`, `tests/test_features.py` | mock `fetch_ohlcv`; no network |

## Delegation — keep compute off the main context

Long-running phases (IC, Optuna tuning, walk-forward) MUST run inside the
`experiment-tracker` subagent (see `.claude/agents/experiment-tracker.md`), not
as inline `python -c` calls from the main context. The subagent returns a
compact summary block; raw trial logs and per-fold tables stay in
`data/wf_history.db`. Never paste those back into the main conversation.

Collect the user's ticket ref (`#123`, `QP-45`, `issue-87`) up front in step 1
and forward it to every subagent dispatch — the subagent rejects invocations
without one.

## Steps

Walk the user through these, one at a time. At each step, print what you changed and what command you ran.

1. **Clarify scope and ticket.** Ask: what feature or signal idea, on which ticker universe, and what ticket ref? Default universe is `$WF_TICKERS` or `SPY,QQQ,AAPL,MSFT,TSLA` (matches `cron/daily_ml_execute.py:35`). Stop if no ticket is provided.
2. **Add the feature.** Edit `data/features.py` to add the feature function; register it in the existing feature list. Add or extend a unit test in `tests/test_features.py` that mocks OHLCV — do not hit network.
3. **Run the feature test.** `pytest tests/test_features.py -v`.
4. **Measure IC (delegated).** Dispatch the `experiment-tracker` subagent with phase=`ic`, the ticker, the ticket ref, and `--feature <name>`. Forward the returned summary block verbatim. Convention from `IMPLEMENTATION_SUMMARY.md`: if `|IC| < 0.02` and ICIR is small, reject the feature and stop.
5. **Wire into a signal.** If a pure feature, add to `MLSignal`. If combining signals, extend `ensemble_signal.py`. Do **not** modify `cron/monthly_ml_retrain.py` or `cron/daily_ml_execute.py` — they consume whatever `MLSignal` and `load_best_params` return.
6. **Tune (delegated).** Dispatch the `experiment-tracker` subagent with phase=`tune`, the ticker, the ticket ref (ask for trial count; default 30). The subagent persists winners via `strategies/ml_tuning.py` so `monthly_ml_retrain` picks them up on the next schedule. Forward the summary block.
7. **Walk-forward backtest (delegated).** Dispatch the `experiment-tracker` subagent with phase=`wf`, the ticker, the ticket ref. Forward the summary block (OOS Sharpe, OOS max drawdown, hit rate, folds).
8. **Report.** Combine the three returned summary blocks into a one-screen experiment report:

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
