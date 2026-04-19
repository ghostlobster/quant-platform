# Broker Integration & Maintenance Guide
## Quant Trading Platform — Individual Trader Reference

> **Platform context:** This document is specific to the quant-platform codebase at `~/projects/quant-platform/`. All file paths, module references, and CLI commands refer to that project.

---

## Table of Contents

1. [Broker Landscape](#1-broker-landscape)
2. [Broker Comparison Matrix](#2-broker-comparison-matrix)
3. [Adding a New Broker](#3-adding-a-new-broker)
4. [Maintenance Playbook](#4-maintenance-playbook)
5. [Enhancement Roadmap](#5-enhancement-roadmap)
6. [Priority Summary Table](#6-priority-summary-table)

---

## 1. Broker Landscape

### 1a. Currently Implemented

#### Alpaca Markets (`broker/alpaca_bridge.py`)
The platform's live/paper execution layer. Alpaca provides commission-free equity and crypto trading with a clean REST + WebSocket API.

**What's wired up:**
- `get_account()` — equity, buying power, portfolio value
- `get_positions()` — all open positions with current P&L
- `place_market_order(ticker, qty, side)` — market orders only
- `cancel_all_orders()` — emergency flatten
- `is_market_open()` — calendar check before order routing

**Safe-mode behaviour:** When `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` are absent from `.env`, every method returns a no-op result instead of raising. This lets you run full backtests and paper simulations without credentials.

**Switching paper → live:** Set `ALPACA_BASE_URL=https://api.alpaca.markets` in `.env` (paper default is `https://paper-api.alpaca.markets`). No code change required.

---

#### Internal Paper Trader (`broker/paper_trader.py`)
SQLite-backed paper trading engine that runs entirely offline. Useful for strategy validation before touching Alpaca.

**Key functions:**
```
paper_buy(ticker, qty, price)
paper_sell(ticker, qty, price)
get_paper_portfolio()
get_paper_trades()
```

The DB path is controlled by `PAPER_DB_PATH` env var (default `paper_trades.db`). Override in tests with `monkeypatch`.

---

### 1b. Recommended Additions

#### Tradier (`broker/tradier_bridge.py` — not yet built)
Best choice if you trade **options**. Offers $0 commission on equities and a flat $0.35/contract on options. REST API is straightforward and well-documented.

**Key capabilities:**
- Options chains, Greeks, expiry calendars
- Streaming quotes via WebSocket
- Paper trading sandbox (free account)

**Python integration:** `requests` only — no dedicated SDK needed. OAuth bearer token, refresh daily.

---

#### Interactive Brokers — IBKR (`broker/ibkr_bridge.py` — not yet built)
The institutional-grade choice. Best for: large position sizes, international markets, margin accounts, futures.

**Integration path:** `ib_insync` (community library, async-friendly):
```bash
pip install ib_insync
```
Requires TWS or IB Gateway running locally. Connect via `IB().connect('127.0.0.1', 7497, clientId=1)`.

**Caution:** IBKR's API is stateful and session-based. Reconnection logic and heartbeat handling add meaningful complexity compared to Alpaca's stateless REST.

---

#### Charles Schwab (`broker/schwab_bridge.py` — not yet built)
Acquired TD Ameritrade's thinkorswim infrastructure. Solid choice for retail traders who want broad instrument coverage (ETFs, options, bonds) under one roof.

**Integration path:** `schwab-py` (community SDK):
```bash
pip install schwab-py
```
OAuth2 flow — tokens persist to a local JSON file. Supports streaming quotes and options chains.

---

#### Tastytrade (`broker/tastytrade_bridge.py` — not yet built)
Derivatives-first broker with the lowest options commissions in the space ($1/contract cap). Excellent API documentation.

**Integration path:** `tastytrade` SDK:
```bash
pip install tastytrade
```
REST + WebSocket. Particularly good for covered calls, spreads, and defined-risk options strategies that complement the existing momentum and pairs signals.

---

#### ccxt — Crypto (`broker/ccxt_bridge.py` — not yet built)
Single unified interface to 100+ crypto exchanges (Binance, Coinbase, Kraken, Bybit, OKX, etc.).

**Integration path:**
```bash
pip install ccxt
```
Supports spot, perpetual futures, and options on supported exchanges. Pairs naturally with the existing momentum scorer and cointegration pairs engine for crypto pairs like BTC/ETH.

---

## 2. Broker Comparison Matrix

| Broker | Asset Classes | Commission | API Style | Complexity | Status |
|---|---|---|---|---|---|
| **Alpaca** | Equities, Crypto | $0 | REST / WS | Low | ✅ Implemented |
| **Paper Trader** | Equities (simulated) | $0 | Internal | None | ✅ Implemented |
| **Tradier** | Equities, Options | $0 eq / $0.35 opt | REST / WS | Low | 🔲 Recommended next |
| **Tastytrade** | Equities, Options, Futures | $1/contract cap | REST / WS | Low–Medium | 🔲 Options focus |
| **Schwab** | Equities, Options, ETFs, Bonds | $0 eq | OAuth REST / WS | Medium | 🔲 Broad coverage |
| **IBKR** | Everything (global) | Tiered | TWS socket (ib_insync) | High | 🔲 Pro/institutional |
| **ccxt** | Crypto (100+ exchanges) | Exchange-dependent | REST / WS | Medium | 🔲 Crypto expansion |

---

## 3. Adding a New Broker

All broker modules should follow the same interface contract established by `alpaca_bridge.py`. Create `broker/<name>_bridge.py` implementing:

```python
def get_account() -> dict:          # cash, equity, buying_power
def get_positions() -> list[dict]:  # ticker, qty, avg_entry, unrealised_pnl
def place_market_order(ticker: str, qty: int, side: str) -> dict:
def cancel_all_orders() -> None:
def is_market_open() -> bool:
```

This keeps `app.py` and the execution model (`broker/execution.py`) broker-agnostic. The execution cost model can wrap any bridge transparently.

---

## 4. Maintenance Playbook

### 4a. Daily (5–10 minutes)

| Task | How to do it in this platform |
|---|---|
| Check alert log | `tail -100 logs/quant_platform.log` — look for CRITICAL or WARNING |
| Verify supervisord health | `supervisorctl status` — both `quant-streamlit` and `quant-alerts` should show `RUNNING` |
| Scan screener output | Open the Screener tab in Streamlit; flag any tickers near your buy/sell thresholds |
| Check open positions | Alpaca tab → Positions panel; compare against expected from last session |
| Review overnight news sentiment | `data/sentiment.py get_ticker_sentiment(ticker)` for any open positions |

---

### 4b. Weekly (30–60 minutes)

| Task | How to do it in this platform |
|---|---|
| Re-run screener on full watchlist | Streamlit Screener tab with your full ticker list |
| Check correlation matrix | Risk tab → Correlation Heatmap; flag pairs > 0.80 (concentration risk) |
| Review Sortino and Calmar | Risk tab → Metrics panel; Sortino < 0.5 or Calmar < 0.3 warrants investigation |
| Inspect paper trade log | `SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT 50;` on `paper_trades.db` |
| Reconcile paper vs. live | Compare paper portfolio returns against Alpaca live account performance |
| Check for library updates | `pip list --outdated` — review before upgrading in prod |

---

### 4c. Monthly (2–3 hours)

| Task | How to do it in this platform |
|---|---|
| Re-run walk-forward backtest | Backtester tab → Walk-Forward; compare consistency score to last month's baseline |
| Run Monte Carlo stress test | Backtester tab → Monte Carlo; check 5th-percentile path against your drawdown tolerance |
| Update pip freeze | `pip freeze > requirements.txt` after any library changes |
| Rotate API credentials | Generate new Alpaca API key pair; update `.env`; restart supervisord |
| SQLite maintenance | `sqlite3 paper_trades.db "VACUUM; ANALYZE;"` — keeps query performance tight |
| Audit `.env` | Confirm no stale keys; verify `MAX_DRAWDOWN_PCT` and `ALPACA_BASE_URL` are correct |
| Review Kelly fractions | `risk/kelly.py kelly_from_backtest()` with last 30 days of trades — adjust position sizing |

---

### 4d. Quarterly (half-day)

| Task | How to do it in this platform |
|---|---|
| Re-optimise strategy parameters | Re-run walk-forward with wider parameter grid; update default thresholds in config |
| Re-run Markowitz frontier | Risk tab → Efficient Frontier with refreshed price data; rebalance if allocation drifted > 5% |
| Stress-test VaR assumptions | Compare `historical_var()` vs. `parametric_var()` vs. `conditional_var()`; check for divergence |
| Review cointegration pairs | Re-run `pairs.test_cointegration()` on your pairs universe; drop broken pairs |
| Full log archive | Rotate `logs/quant_platform.log`; archive to cold storage; verify supervisord log rotation |
| Performance attribution | Break down returns by strategy (momentum vs. pairs vs. SMA crossover) — cut underperformers |
| Regime review | Manually classify last quarter: trending / mean-reverting / choppy; note which strategies won |

---

### 4e. Infrastructure Hygiene

| Item | Action |
|---|---|
| `.env` backup | Encrypt with `gpg -c .env` and store outside the repo (never commit plaintext keys) |
| `paper_trades.db` backup | Weekly `cp paper_trades.db backups/paper_trades_$(date +%Y%m%d).db` |
| `requirements.txt` pin | Keep pinned to exact versions in prod; use a separate `requirements-dev.txt` for looser dev deps |
| Supervisord watchdog | Add a cron job: `supervisorctl status | grep -v RUNNING && supervisorctl restart all` |
| Disk space | `du -sh logs/ *.db` monthly — log files grow silently |
| Test suite | Run `pytest --tb=short -q` before any production code change |

---

### 4f. The Two Most-Neglected Items

These are the maintenance tasks most individual quant traders skip — and the ones that cause the most problems long-term:

**1. Trading Journal**
Without a journal, you cannot distinguish luck from edge. After every trade (paper or live), log:
- Entry rationale (which signal fired, what regime you thought it was)
- Exit rationale (target hit, stop hit, time exit)
- Post-trade review: was the signal correct? Was the sizing right?

Currently the platform has no trading journal module. This is the highest-value addition you can make next (see Enhancement Roadmap §5).

**2. Market Regime Detector**
All strategies in this platform — momentum, SMA crossover, mean reversion, pairs — have regimes where they work and regimes where they destroy capital. Without a regime classifier running in real-time, you're flying blind on *when* to use each strategy.

A simple two-state regime detector (trending vs. mean-reverting) using VIX level + SPY 200-day SMA relationship would immediately let the strategy selector route signals to the right model.

### 4g. Live Broker Testing (Alpaca paper)

Tracking issue: [#78](https://github.com/ghostlobster/quant-platform/issues/78). Cron unit tests run with a fake broker; this runbook is the manual loop for confirming the real `BROKER_PROVIDER=alpaca` path end-to-end against a paper account.

**Required environment**

```bash
export BROKER_PROVIDER=alpaca
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
export ALPACA_PAPER=true                  # never run this against a live account
export WF_TICKERS="AAPL,MSFT"             # optional — defaults to the cron's universe
export ML_MAX_POSITIONS=2                 # optional
export ML_SCORE_THRESHOLD=0.3             # optional
```

**Procedure**

```bash
python -m cron.monthly_ml_retrain   # warm up the LightGBM checkpoint
python -m cron.daily_ml_execute     # place paper orders
```

**Verification (Alpaca dashboard)**

- Orders fill at market and appear in the paper account.
- Position sizes honour `Kelly × regime × |score|` from `strategies/ml_execution.py`.
- A repeat run leaves the portfolio unchanged when scores stay within `[-threshold, threshold]` (idempotency).

**Automated portion**

The integration test scaffolded under [`tests/test_alpaca_smoke.py`](tests/test_alpaca_smoke.py) covers the broker plumbing automatically when credentials are present:

```bash
pytest tests/test_alpaca_smoke.py -m integration -v
```

It is always skipped without `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`, so it is invisible in the default `pytest -m "not integration"` CI run. The `ci.yml` integration job picks it up when those secrets are configured at the repo level.

**Closing the ticket**

The screenshot acceptance criterion in #78 still requires manual verification via the Alpaca dashboard — record the fills there, attach to the PR or issue, and link to the green integration-job run.

---

## 5. Enhancement Roadmap

### Priority 1 — Foundation (1–2 weeks each)

#### P1-A: Market Regime Detector
**File:** `analysis/regime.py` (new)

Classify the current market environment to gate which strategies are activated.

```python
# Proposed interface
def detect_regime(spy_prices: pd.Series, vix_level: float) -> str:
    # Returns: 'trending_bull' | 'trending_bear' | 'mean_reverting' | 'high_vol'
```

**Logic:**
- SPY above 200d SMA + VIX < 20 → `trending_bull` → favour momentum, SMA crossover
- SPY below 200d SMA + VIX < 20 → `trending_bear` → favour short momentum, pairs
- VIX 20–30 → `mean_reverting` → favour RSI mean reversion, pairs
- VIX > 30 → `high_vol` → reduce all position sizes; Kelly fraction halved automatically

**Integration point:** `app.py` Strategy tab header — show current regime badge before signal output.

---

#### P1-B: Trading Journal Module
**File:** `journal/trading_journal.py` (new)

SQLite-backed journal that auto-captures every paper and live trade, and prompts for post-trade notes.

```python
def log_entry(ticker, side, qty, price, signal_source, regime, notes="") -> None:
def log_exit(ticker, side, qty, price, pnl, exit_reason, notes="") -> None:
def get_journal(start_date, end_date) -> pd.DataFrame:
def win_rate_by_signal_source() -> pd.DataFrame:   # which signals actually work
def avg_pnl_by_regime() -> pd.DataFrame:            # regime attribution
```

**Integration point:** Hook into `paper_trader.py` `paper_buy()` / `paper_sell()` to auto-log. Add a Journal tab in Streamlit with filters by date, ticker, signal, regime.

---

### Priority 2 — Broker Expansion (1 week each)

#### P2-A: Tradier Options Bridge (`broker/tradier_bridge.py`)
Options are the most capital-efficient way to express directional and volatility views. Tradier's flat-rate pricing makes it viable for individual traders.

**Minimum viable scope:**
- Fetch options chain for a ticker (calls + puts, all expiries)
- Place single-leg market order (buy call, buy put, sell covered call)
- Track open options positions with delta, theta, days-to-expiry

---

#### P2-B: ccxt Crypto Bridge (`broker/ccxt_bridge.py`)
The existing momentum scorer and cointegration pairs engine work on price series — they're exchange-agnostic. Wrapping ccxt gives you 100+ crypto markets for free.

**Minimum viable scope:**
- Fetch OHLCV for any ccxt-supported pair
- Place spot market orders on Binance or Coinbase
- Map ccxt position format to the standard internal dict

---

### Priority 3 — Strategy Enhancements (2–4 weeks)

#### P3-A: Options Greeks Module (`analysis/greeks.py`)
Once Tradier is connected, add a Greeks calculator for position-level risk:

- Delta (directional exposure)
- Theta (daily time decay in $)
- Vega (volatility sensitivity)
- Portfolio-level delta and theta aggregation

Use the Black-Scholes closed form (pure numpy, no external deps — same pattern as `risk/var.py`).

---

#### P3-B: Real-Time Data Upgrade (`data/realtime.py`)
Currently all data is fetched via `yfinance` (batch, delayed). For intraday strategies, upgrade to:

- **Alpaca WebSocket** (already authenticated) for live equity quotes — free with existing credentials
- **Crypto:** ccxt WebSocket streams for bid/ask
- **Integration point:** Feed live prices into the screener's signal engine for intraday signal generation

---

#### P3-C: Portfolio Rebalancing Automation (`strategies/rebalancer.py`)
The Markowitz frontier already computes the optimal weights. Add a rebalancing engine:

```python
def compute_rebalance_trades(
    current_positions: dict,      # {ticker: market_value}
    target_weights: dict,         # from get_max_sharpe_portfolio()
    total_equity: float,
    min_trade_value: float = 500  # ignore tiny rebalancing trades
) -> list[RebalanceTrade]:
```

**Integration point:** Efficient Frontier tab — "Generate Rebalance Orders" button that outputs the trade list for review before any execution.

---

#### P3-D: Alerting Upgrade (`alerts/channels.py`)
Currently alerts write to log file only. Add push notification channels:

- **Telegram Bot:** `python-telegram-bot` — free, instant, no spam filters
- **Email:** `smtplib` with Gmail app password — zero new dependencies
- **Webhook:** Generic HTTP POST — connects to Slack, Discord, PagerDuty, n8n

Each channel registered as a plugin so you can mix-and-match without touching alert logic.

---

### Priority 4 — Infrastructure (ongoing)

#### P4-A: Docker Deployment (`docker-compose.yml`)
Replace `supervisord` with Docker Compose for portability and reproducibility:

```yaml
services:
  streamlit: { build: ., command: streamlit run app.py }
  alerts:    { build: ., command: python alerts/engine.py }
  db:        { image: sqlite-web }  # optional web UI for paper_trades.db
```

Eliminates "works on my machine" issues when moving to a VPS.

---

#### P4-B: Automated Walk-Forward Cron (`cron/monthly_wf.py`)
Run walk-forward backtests automatically on the 1st of each month for all active tickers. Save results to a time-series SQLite table so you can track consistency score drift over time.

```bash
# crontab entry
0 6 1 * * /path/to/venv/bin/python /path/to/quant-platform/cron/monthly_wf.py
```

---

#### P4-C: Code Coverage to 80%+ (`tests/`)
Current coverage: 23%. The gap is almost entirely `app.py` (0%) and the newer `strategies/` modules. Approach:

- Add `pytest-mock` fixtures for Streamlit components to unit-test page render functions
- Add integration tests for `momentum.py` and `pairs.py` using synthetic OHLCV fixtures
- Use `coverage run -m pytest && coverage html` to get a browsable HTML report

---

## 6. Priority Summary Table

| ID | Enhancement | Effort | Impact | Priority |
|---|---|---|---|---|
| P1-A | Market Regime Detector | 1 week | 🔴 Critical — prevents strategy misuse | **Do first** |
| P1-B | Trading Journal | 1 week | 🔴 Critical — without it you can't learn | **Do first** |
| P2-A | Tradier Options Bridge | 1 week | 🟠 High — opens options strategies | Next |
| P2-B | ccxt Crypto Bridge | 1 week | 🟠 High — expands asset universe | Next |
| P3-A | Options Greeks Module | 2 weeks | 🟠 High — required for options risk management | After P2-A |
| P3-B | Real-Time Data Feed | 2 weeks | 🟡 Medium — enables intraday signals | After P2 |
| P3-C | Portfolio Rebalancer | 1 week | 🟡 Medium — closes the Markowitz loop | Anytime |
| P3-D | Alerting Channels | 3 days | 🟡 Medium — quality-of-life | Anytime |
| P4-A | Docker Deployment | 3 days | 🟢 Low — portability | When deploying to VPS |
| P4-B | Automated Walk-Forward Cron | 2 days | 🟢 Low — saves manual effort | After walk-forward stable |
| P4-C | Test Coverage to 80% | 2 weeks | 🟢 Low — reliability safety net | Ongoing |

---

## 11. Knowledge-Gating Operations

The `KnowledgeAdaptionAgent` (`agents/knowledge_agent.py`) watches the
freshness and IC of the LightGBM alpha model and produces a verdict in
`{fresh, monitor, retrain}`. Three optional operator controls escalate
the agent from advisory to automated:

### 11.1 Pre-trade circuit breaker (#120)

```
python -m cron.daily_ml_execute --enforce-knowledge-gate
KNOWLEDGE_GATE_ENFORCE=1  python -m cron.daily_ml_execute
```

On a `retrain` verdict the cron exits with code **2** before any order
is placed. `fresh` / `monitor` proceed unchanged. The CLI flag wins
over the env var when both are set.

### 11.2 Scheduled knowledge health check (#116)

`scheduler/alerts.py` hosts a `knowledge_health_job` registered on the
APScheduler `BackgroundScheduler`. Runs hourly by default (override
with `KNOWLEDGE_HEALTH_CRON`, disable with `KNOWLEDGE_HEALTH_ENABLED=0`).
Ensures stale models are flagged even on quiet days when no trade flow
would otherwise invoke the agent.

### 11.4 Live-IC pipeline (#115)

Without real-time IC feedback the agent's IC-degradation branch never
fires: mtime-based staleness and regime coverage are the only active
gates. The live-IC pipeline closes that loop.

**Writer.** `strategies/ml_execution.py::execute_ml_signals` and
`cron/daily_ml_execute.py` both call
`analysis.live_ic.record_predictions(scores, "lgbm_alpha", horizon_d=5)`
**before** filtering by threshold — recording only traded names would
bias the IC toward high-conviction positions. Both writers are wrapped
in `try / except` so a DB outage can never break trading.

**Storage.** New table `live_predictions (ts, ticker, model_name, score,
horizon_d, realized)` in `quant.db`, with an index on
`(model_name, ts DESC)` so rolling-IC reads stay O(window). Idempotent
`INSERT OR REPLACE`; the PK dedups re-runs at the same epoch.

**Backfill.** `analysis.live_ic.backfill_realized(model_name="lgbm_alpha")`
is registered in `scheduler/alerts.py` as `live_ic_backfill_job`, running
at 04:30 UTC daily by default (override with `LIVE_IC_BACKFILL_CRON`).
One `fetch_ohlcv(ticker, period="3mo")` per distinct ticker in the
candidate set — the fetcher's cache dedups across runs. Bounded by
`max_rows=1000` so a catch-up after downtime can't exhaust memory.

**Estimator.** `analysis.live_ic.rolling_live_ic("lgbm_alpha")` returns
Spearman rank-IC over the last 60 realized rows (warm-up floor 30).
5-minute TTL cache, invalidated on `backfill_realized`. The value is
threaded into `KnowledgeAdaptionAgent().run({"regime": ..., "live_ic":
...})` via `_knowledge_gate` in `strategies/ml_execution.py`.

**Operator controls.**
- `KNOWLEDGE_RECORD_PREDICTIONS=0` disables the writer (defaults on).
- `LIVE_IC_BACKFILL_CRON` changes the backfill cadence.
- `KNOWLEDGE_HEALTH_ENABLED=0` disables the APScheduler loop entirely
  (kills both the health-check job and the backfill job).

**Runtime risks and mitigations.**

| Risk | Mitigation |
|---|---|
| yfinance rate limits on backfill | `fetch_ohlcv` caches via `data/price_cache`; backfill groups by ticker so each symbol is fetched at most once per run. |
| `live_predictions` grows unbounded | Out of scope for #115 — open a retention ticket if it becomes a pain point. Rolling-IC reads use `ORDER BY ts DESC LIMIT window` so query time stays constant regardless of table size. |
| Realized return spans a missing trading day | `_realized_return` uses `searchsorted` with `side="left"` so weekends/holidays pull the next available close. Returns `None` when either anchor is past the fetched window. |
| Cache returns stale IC after backfill | `backfill_realized` calls `_invalidate_ic_cache(model_name)` before returning. |
| Writer import failure during cold-start | Both call sites wrap the import in `try/except`; trading proceeds without recording, and the next invocation retries. |

### 11.5 Model Health dashboard (#121)

Read-only Streamlit tab — `streamlit run app.py` → **🩺 Model Health**
— that surfaces the state `KnowledgeAdaptionAgent` reports before every
trade. Reads `model_metadata`, `live_predictions` (via
`analysis.live_ic.rolling_live_ic`), and the LightGBM regime-models
pickle. No writes, no broker calls.

Four panels:
1. **Inventory** — latest `model_metadata` row per model plus the
   agent's current verdict and Kelly multiplier.
2. **Live vs trained IC** — Plotly line per model; falls back to a
   warm-up notice when `live_predictions` has fewer than 30 realized
   rows.
3. **Regime coverage** — matrix of `REGIME_STATES × model`. ✅ covered,
   ❌ missing, ℹ︎ baseline (pooled).
4. **Retrain history** — last 10 `model_metadata` rows per model with
   `test_ic` and optional `test_ic_delta` (#122) sparklines.

Cached reads (`@st.cache_data`): 60 s for the SQL helpers, 300 s for
the `KnowledgeAdaptionAgent().run({})` call (matches the agent's own
pickle-read TTL).

### 11.6 Covariate-shift detector (#118)

At training time `MLSignal.train()` persists a compact per-feature
fingerprint (mean, std, q10/50/90, n_samples) to the new
`model_feature_stats` table — one row per
`(model_name, trained_at, feature_name)`, `INSERT OR REPLACE`.

Drift is then available via `analysis.drift`:

- `summarize_features(frame, feature_cols)` — the fingerprint writer;
  drops columns with fewer than 30 non-NaN rows.
- `feature_psi(training_stats, live_frame, feature_cols)` — Population
  Stability Index per feature, computed by reconstructing 10 bin edges
  from the quantile anchors + ±3σ tails. An approximation (the
  fingerprint does not carry the raw training sample) — tests pin the
  approximation error below 10 % on synthetic Gaussian shifts.
- `kolmogorov_smirnov(train, live)` — thin wrapper around
  `scipy.stats.ks_2samp` for callers that still have both raw samples.
- `aggregate_drift(psi_scores)` — collapses a per-feature PSI dict to
  `{"level": "none"|"monitor"|"retrain", "max_psi": float|None,
  "drifted_features": list[str]}`.

`KnowledgeAdaptionAgent` consumes `context["drift"]` (full dict),
`context["drift_score"]` (scalar shortcut), or `context["feature_frame"]`
(raw live DataFrame — the agent reads the stored fingerprint and runs
PSI). Defaults: `_DRIFT_PSI_MONITOR=0.10`, `_DRIFT_PSI_RETRAIN=0.25`
(AFML / ML4T Ch 17 conventions). Drift verdicts slot into the ladder
**after** hard staleness and IC-collapse but **before** regime-coverage
gaps — shifted inputs invalidate every regime bucket, so they trump a
single missing regime.

`metadata["drift_level"]`, `metadata["drift_max_psi"]`, and
`metadata["drifted_features"]` are surfaced on every agent run so the
Model Health tab (#121) can display them.

### 11.7 Model zoo registry (#123)

`KnowledgeAdaptionAgent` used to audit only the two LightGBM pickles.
The repo ships several other model families (bayesian, ridge, mlp,
cnn, lstm, rf-long-short); stale members silently polluted the
ensemble blend.

`agents/knowledge_registry.py::ModelEntry` is the immutable per-model
record — name, env var that overrides the artefact path, repo-relative
default, `model_metadata.model_name` string, and staleness budget.
Each strategy module declares its own `MODEL_ENTRY` constant so the
registry stays declarative (see
`strategies/ml_signal.py:MODEL_ENTRY` for the baseline template).

`build_default_registry()` collects every strategy's entry at runtime;
each import is wrapped in `try / except` so a missing optional
dependency (torch for `cnn_signal` / `dl_signal`, for example) drops
that single entry rather than crashing the whole audit.

The agent loops over the registry, writes a per-model verdict into
`metadata["per_model"][name]`, and reports the **worst** verdict
(`retrain` > `monitor` > `fresh`) as the top-level `recommendation`
so `MetaAgent`'s existing multiplier still works. Adding a new family
is two lines: declare `MODEL_ENTRY` in the strategy module and list
the env var in `.env.example`.

Operators can opt into a narrower audit surface by constructing the
agent with an explicit registry (`KnowledgeAdaptionAgent(registry=[…])`),
which is also what the test suite uses.

### 11.3 Opt-in auto-retrain trigger (#119)

**Default: off.** Set `KNOWLEDGE_AUTO_RETRAIN=1` in the environment of
whichever process instantiates `KnowledgeAdaptionAgent` (the
`knowledge_health_job`, the `MetaAgent` vote path, the
`daily_ml_execute` circuit breaker, or a manual CLI run). When enabled
and the verdict is `retrain`, the agent spawns

```
python -m cron.monthly_ml_retrain
```

as a **detached subprocess** (`Popen(..., start_new_session=True)`).
The agent's hot path returns immediately — a daemon watcher thread
logs the subprocess exit code via structlog, but nothing in the agent
call graph blocks on the retrain.

Launches are deduped by a SQLite row in `knowledge_stamps`
(`name='retrain_fired_at'`), independent from the in-process alert
cooldown. Default cooldown is 24h; override via
`KNOWLEDGE_RETRAIN_COOLDOWN` (seconds).

The launch emits an alert with a distinct subject (`"ML knowledge
auto-retrain launched (pid=…): …"`) so operators can distinguish it
from the existing stale-model alert (`"ML knowledge stale — retrain
recommended: …"`).

**Runtime risks and mitigations:**

| Risk | Mitigation |
|---|---|
| Retrain subprocess uses `sys.executable` | Runs in the same venv as the caller — operators must ensure `lightgbm` is installed in that venv. |
| Overlap with monthly cron (`0 7 1 * *`) | Safe — both write atomically via `MLSignal().train()`, and the 24h stamp prevents the agent from piling up. |
| Retrain failure goes unnoticed | Watcher thread logs exit code at INFO level; production should aggregate structlog output. Non-zero exits do **not** currently trigger an alert (deferred). |
| Stamp never cleared | Harmless — an old timestamp only makes the cooldown gate open sooner on the next retrain condition. |
| Double-fire across two simultaneous `run()` calls | Extremely unlikely (microsecond window between stamp read and write). If it ever bites, upgrade the write to `INSERT ... ON CONFLICT DO UPDATE WHERE last_fired_at < excluded.last_fired_at`. |

**Disabling:** unset `KNOWLEDGE_AUTO_RETRAIN` and restart the process.
The stamp row remains but is ignored.

---

*Document generated: 2026-04-11 | Platform: `~/projects/quant-platform/` | See also: `TRADING_PHILOSOPHY.md`, `Quant_Living_Roadmap.docx`*
