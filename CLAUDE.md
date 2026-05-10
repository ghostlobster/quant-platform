# CLAUDE.md — AI Assistant Guide for quant-platform

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

## Project Overview

A production-ready quantitative trading and analytics platform built with **Python 3.11** and **Streamlit**. It supports multi-broker live/paper trading, ML-driven alpha generation, backtesting, options analytics, portfolio risk, and market screening — all behind a single web dashboard.

**Entry point:** `streamlit run app.py` → http://localhost:8501

---

## Repository Structure

```
quant-platform/
├── app.py                  # Streamlit entry point; bootstraps all subsystems
├── config.py               # Loads .env vars, configures structlog
├── requirements.txt        # 73 pinned Python dependencies
├── requirements-dev.txt    # Dev-only deps (mypy, stubs)
├── ruff.toml               # Linter config (line-length 100, E/F/W/I rules)
├── setup.cfg               # mypy config (Phase-1: providers/, risk/, bus/, journal/)
├── pytest.ini              # Test discovery config (60s timeout, strict markers)
├── .coveragerc             # Coverage config (76%+ floor, branch coverage)
├── .gitleaks.toml          # Secret-scan allowlist for pre-commit + CI
├── .pre-commit-config.yaml # gitleaks + linting hooks
├── Dockerfile              # python:3.11-slim, healthcheck on port 8501
├── docker-compose.yml      # streamlit + alerts + prometheus + grafana
├── run.sh                  # Local dev launcher (venv + streamlit)
│
├── adapters/               # Pluggable adapters (implement provider protocols)
│   ├── broker/             # Broker adapters (Alpaca, IBKR, Schwab, Paper)
│   ├── market_data/        # Market data adapters (alpaca, yfinance, polygon, mock)
│   ├── alert/              # Alert channel adapters (email, slack, no-op)
│   ├── llm/                # LLM adapters (Anthropic, OpenAI, Ollama, mock)
│   ├── sentiment/          # Sentiment adapters (VADER, StockTwits, mock)
│   ├── tsdb/               # Time-series DB adapters (SQLite, DuckDB, TimescaleDB)
│   ├── feature_store/      # Feature store adapters (memory, Redis)
│   ├── execution_algo/     # Execution algorithm adapters (market, TWAP, VWAP)
│   ├── macro/              # Macro data adapters (FRED, mock)
│   ├── auth/               # Auth adapters (GitHub, Google)
│   ├── model_registry/     # Model registry adapters (MLflow, mock)
│   └── options_flow/       # Options flow adapters (Unusual Whales, ThetaData, mock)
│
├── agents/                 # AI/rule-based specialist trading agents
├── alerts/                 # Notification channels: Telegram, Email, Slack, Webhook
├── analysis/               # Quant analytics: Greeks, risk metrics, regime, IC, drift
├── audit/                  # Structured audit trail logger
├── auth/                   # OAuth token management + RBAC session state
├── backtester/             # Event-driven backtester, walk-forward, Monte Carlo
├── broker/                 # Direct broker integrations + paper trading engine
├── bus/                    # Pub-sub event bus for inter-module communication
├── cron/                   # Scheduled jobs (daily execution, monthly retrain, WF)
├── data/                   # Data fetching, caching, feature engineering (SQLite-backed)
├── deploy/                 # supervisord configs and deployment helpers
├── docs/                   # Decision records and plan-review audit logs
├── journal/                # Trading journal (entry/exit metadata, analytics)
├── monitoring/             # Prometheus metrics exporter + sidecar
├── pages/                  # One Streamlit tab per file + shared sidebar
├── providers/              # Protocol definitions + factory functions (DI layer)
├── risk/                   # VaR, CVaR, Kelly, HRP, Markowitz, pre-trade guards
├── scheduler/              # APScheduler alert + knowledge-health engine
├── screener/               # Equity screening by momentum / factor criteria
├── scripts/                # CI helper scripts (coverage, silent-skip, e2e perf)
├── strategies/             # Technical indicators, ML signals, execution logic
├── tests/                  # 39 test files, 76%+ coverage enforced in CI
└── utils/                  # Logging helpers
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| UI | Streamlit 1.57.0 |
| Data | pandas 3.0.2, numpy 2.4.4, yfinance 1.3.0 |
| Charts | plotly 6.7.0 |
| Indicators | ta 0.11.0 |
| ML | lightgbm 4.0+, scikit-learn 1.3+, scipy 1.11+, gensim 4.3+ |
| Crypto | ccxt 4.0+ (100+ exchanges) |
| Brokers | Alpaca, IBKR (IB Gateway), Schwab, Tradier |
| Scheduling | APScheduler 3.11.2 |
| Database | SQLite (WAL mode) via stdlib `sqlite3`, DuckDB 1.5.2 |
| Logging | structlog 24.0+ |
| Observability | Prometheus metrics, Grafana dashboards |
| Linting | ruff |
| Type checking | mypy (Phase-1: providers/, risk/, bus/, journal/) |
| Security | bandit, pip-audit, gitleaks |
| Testing | pytest 9.0.3, pytest-cov, pytest-xdist, Hypothesis, freezegun |
| CI/CD | GitHub Actions |
| Deploy | Docker + docker-compose, Kubernetes Helm charts |

---

## Development Workflow

### Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # mypy + stubs
cp .env.example .env                  # fill in API keys
pip install pre-commit                # secret-scan + lint hooks (#249)
pre-commit install                    # one-time per clone
bash run.sh                           # starts Streamlit on :8501
```

The `pre-commit install` step wires `gitleaks` into every `git commit`
to catch leaked API keys / tokens before they enter the local history.
CI runs the same check via `gitleaks/gitleaks-action` so devs without
the local hook still get caught at PR time, but the pre-commit gate is
the cheaper layer (it fires before the commit lands, not after the push
is rejected). False positives go in `.gitleaks.toml`.

### Running Tests

```bash
# Unit tests only (fast, no external services required)
pytest tests/ -m "not integration and not e2e"

# With coverage report
pytest tests/ -m "not integration and not e2e" --cov=. --cov-report=term-missing

# E2E chain tests (SQLite + paper trader, no network)
pytest tests/ -m "e2e"

# Integration tests (require live credentials)
pytest tests/ -m "integration"
```

**CI enforces `--cov-fail-under=76` (branch coverage).** The excellent-test gate (#215) additionally requires ≥ 85% coverage on every source file a PR modifies.

For a single-command CI-mirror run (ruff + pytest 76% + bandit HIGH + pip-audit) from inside Claude Code, invoke the `/pre-push` skill.

### Linting

```bash
ruff check .          # check for issues
ruff check . --fix    # auto-fix where possible
```

Config: `ruff.toml` — line-length 100, rules E/F/W/I, E501 ignored.

### Type Checking

```bash
python -m mypy    # config-driven via setup.cfg
```

Phase-1 scope: `providers/`, `risk/`, `bus/`, `journal/` with `--strict-optional`. Catches `Optional[X]` passed where `X` is required at PR time.

### Security Scanning

```bash
bandit -r . -ll --exclude ./.git,./tests
pip-audit -r requirements.txt
```

Bandit fails CI only on HIGH severity findings. `PYSEC-2022-42969` is allowlisted in pip-audit.

### Docker

```bash
docker-compose up            # streamlit (:8501) + alerts + prometheus + grafana
docker-compose up streamlit  # UI only
```

---

## CI/CD Pipelines

All pipelines live in `.github/workflows/`.

| Workflow | Trigger | Jobs |
|---|---|---|
| `ci.yml` | PR → main, push to main | lint → typecheck → security → test → e2e → merge-gate |
| `build.yml` | push to main | full test suite → Docker build → validate compose |
| `release.yml` | tag `v*.*.*` | tests → Docker push to GHCR → GitHub Release + changelog |

**Job details for `ci.yml`:**

| Job | What it does |
|---|---|
| `lint` | `ruff check .` |
| `typecheck` | `mypy` on Phase-1 modules (`providers/`, `risk/`, `bus/`, `journal/`) |
| `security` | gitleaks secret scan, bandit HIGH gate, pip-audit, SARIF upload |
| `test` | Unit tests (`-m "not integration and not e2e"`), 76% branch-coverage floor, xdist parallel, excellent-test gate (85% per changed file) |
| `e2e` | E2E chain tests (`-m "e2e"`), per-test ≤ 3 s, total ≤ 30 s, per-module 40% floor |
| `merge-gate` | Single required check; fails if any upstream job did not succeed |

**Never skip CI.** Fix lint/test failures rather than using `--no-verify` or bypass flags.

---

## Architecture Patterns

### Dependency Injection via Providers

The `providers/` directory defines `Protocol` classes and factory functions. Concrete implementations live in `adapters/`. Switch implementations with env vars — no code changes required.

```python
from providers.market_data import get_market_data

provider = get_market_data()          # reads MARKET_DATA_PROVIDER env var
bars = provider.get_bars("AAPL", "1Day", "2024-01-01", "2024-12-31")
```

**All providers and their env var selectors:**

| Provider | Env Var | Options |
|---|---|---|
| Market data | `MARKET_DATA_PROVIDER` | `alpaca`, `yfinance` (default), `polygon`, `mock` |
| Broker | `BROKER_PROVIDER` | `alpaca`, `ibkr`, `schwab`, `tradier`, `ccxt`, `paper` |
| Alerts | `ALERT_PROVIDER` | `telegram`, `email`, `slack`, `webhook` |
| LLM | `LLM_PROVIDER` | `anthropic`, `openai`, `ollama`, `mock` |
| Sentiment | `SENTIMENT_PROVIDER` | `vader`, `stocktwits`, `mock` |
| TSDB | `TSDB_PROVIDER` | `sqlite`, `duckdb`, `timescaledb` |
| Feature store | `FEATURE_STORE_PROVIDER` | `memory`, `redis` |
| Execution algo | `EXECUTION_ALGO_PROVIDER` | `market`, `twap`, `vwap` |
| Macro data | `MACRO_PROVIDER` | `fred`, `mock` |
| Auth | `AUTH_PROVIDER` | `github`, `google` |
| Model registry | `MODEL_REGISTRY_PROVIDER` | `mlflow`, `mock` |
| Options flow | `OPTIONS_FLOW_PROVIDER` | `unusual_whales`, `thetadata`, `mock` |

Always code against the Protocol interface, never import a concrete adapter directly in business logic.

### Data Fetching & Caching

`data/fetcher.py` is the single source of OHLCV data. It checks `data/price_cache` in SQLite before hitting yfinance.

```python
from data.fetcher import fetch_ohlcv

df = fetch_ohlcv("AAPL", "6mo")   # returns pandas DataFrame
```

Cache TTLs: intraday 1h, short-term 4h, historical 24h. Data is stored as JSON in SQLite.

### ML Feature Pipeline

`data/features.py` builds the cross-sectional MultiIndex feature matrix consumed by all ML signals.

```python
from data.features import build_feature_matrix

# Returns MultiIndex (date, ticker) DataFrame with lag returns, rolling stats, volume
features = build_feature_matrix(tickers=["AAPL", "MSFT"], lookback=252)
```

For stationary features, use `data/frac_diff.py` (fixed-width FFD with ADF sweep) to fractionally differentiate price series without discarding memory.

### Database Access

Three SQLite databases:
- `quant.db` — main app state (watchlist, paper trading, portfolio history, price cache)
- `journal_trades.db` — trading journal records
- `data/wf_history.db` — walk-forward backtest results

Always use `data/db.py:get_connection()` for `quant.db` — it handles thread-local connections, WAL mode, and foreign keys.

```python
from data.db import get_connection

conn = get_connection()
conn.execute("SELECT * FROM watchlist")
```

Use UPSERT patterns (`INSERT OR REPLACE`) for cache/state tables.

### Event Bus

Cross-module communication goes through `bus/event_bus.py` — never import pages from strategies or strategies from risk. Publish events; subscribe in the consumer.

```python
from bus.event_bus import EventBus
from bus.events import SignalEvent

bus = EventBus.get()
bus.publish(SignalEvent(ticker="AAPL", score=0.72, regime="trending_bull"))
```

### Logging

Every module should get its own logger via structlog:

```python
import structlog
log = structlog.get_logger(__name__)

log.info("fetching data", ticker="AAPL", period="6mo")
log.warning("cache miss", reason="expired")
```

`LOG_FORMAT=json` produces structured JSON logs (for production). Default is console with colors.  
`LOG_LEVEL` controls verbosity (default `INFO`).

### Streamlit Pages

Each tab in `app.py` corresponds to a file in `pages/`. Each page module exposes a single `render()` function:

```python
# pages/mypage.py
def render() -> None:
    import streamlit as st
    st.title("My Page")
    ...
```

Shared sidebar state (ticker, period, overlays) lives in `pages/shared.py:render_sidebar()`. Access it via `st.session_state` after calling `render_sidebar()`.

---

## Key Module Reference

### Data & Features

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `data/fetcher.py` | OHLCV data with caching | `fetch_ohlcv(ticker, period)` |
| `data/db.py` | SQLite connection factory | `get_connection()`, `init_db()` |
| `data/watchlist.py` | User ticker watchlists | `get_watchlist()`, `add_ticker()` |
| `data/realtime.py` | Real-time price feed (WS + polling fallback) | `RealtimeFeed` class |
| `data/features.py` | Cross-sectional MultiIndex feature matrix | `build_feature_matrix(tickers, lookback)` |
| `data/frac_diff.py` | Fractional differentiation (FFD + ADF sweep) | `frac_diff_ffd(series, d)` |

### Analysis & Quant

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `analysis/greeks.py` | Black-Scholes Greeks + portfolio aggregation | `delta()`, `gamma()`, etc. |
| `analysis/risk_metrics.py` | Sharpe, Sortino, Calmar, max drawdown | `sharpe()`, `sortino()`, `max_drawdown()` |
| `analysis/regime.py` | 4-state market regime classifier | `classify_regime(df)` |
| `analysis/factor_ic.py` | Information Coefficient / ICIR evaluation | `compute_ic(predictions, returns)` |
| `analysis/triple_barrier.py` | López de Prado triple-barrier labeling | `get_labels(prices, events, pt_sl)` |
| `analysis/live_ic.py` | Real-time IC tracking from executed predictions | `LiveICTracker` class |
| `analysis/drift.py` | Covariate-shift detector (PSI) | `compute_psi(reference, current)` |
| `analysis/deflated_sharpe.py` | Probability of Backtest Overfitting | `deflated_sharpe_ratio(...)` |

### Strategies & Signals

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `strategies/indicators.py` | SMA, EMA, RSI, MACD, BB, ATR | Function per indicator |
| `strategies/ml_signal.py` | LightGBM baseline + regime-conditioned models | `MLSignal.fit()`, `.predict()` |
| `strategies/linear_signal.py` | Ridge regression alternative | `LinearSignal.fit()`, `.predict()` |
| `strategies/ensemble_signal.py` | Weighted blend of heterogeneous signals | `EnsembleSignal.predict()` |
| `strategies/meta_label.py` | RandomForest confidence wrapper (AFML Ch 3) | `MetaLabeler.fit()`, `.predict_proba()` |
| `strategies/ml_execution.py` | Kelly × regime × score position sizing | `compute_position_size(signal, regime, kelly)` |
| `strategies/ml_tuning.py` | Optuna Bayesian HPO with purged CV | `tune_hyperparams(signal_cls, features, labels)` |
| `strategies/momentum.py` | Cross-sectional momentum scoring | `momentum_score(df)` |
| `strategies/pairs.py` | Cointegration-based pairs trading | `find_pairs(universe)` |
| `strategies/sentiment_signal.py` | NLP-scored headline signals | `SentimentSignal.predict()` |
| `strategies/rebalancer.py` | Markowitz-based portfolio rebalancing | `rebalance(positions, target_weights)` |

### Risk & Portfolio

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `risk/var.py` | VaR/CVaR (historical, parametric, filtered) | `var()`, `cvar()` |
| `risk/kelly.py` | Half-Kelly position sizing (capped at 25%) | `kelly_fraction(win_rate, payoff_ratio)` |
| `risk/markowitz.py` | Efficient frontier optimization | `optimize(returns, target="max_sharpe")` |
| `risk/hrp.py` | Hierarchical Risk Parity allocation | `hrp_allocate(returns)` |
| `risk/correlation.py` | Correlation matrix + concentration detection | `correlation_matrix(returns)` |
| `risk/pretrade_guard.py` | Pre-trade risk gates + killswitch | `PreTradeGuard.check(order, account)` |
| `risk/options_sizing.py` | Options position sizing (Greeks-based) | `size_options_position(greeks, account)` |

### Backtesting

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `backtester/engine.py` | Vectorized backtester | `run_signal_backtest(signals, prices)` → `BacktestResult` |
| `backtester/walk_forward.py` | Purged walk-forward CV with embargo gap | `run_walk_forward(signal_cls, features, labels)` |
| `backtester/monte_carlo.py` | Bootstrap + synthetic path stress testing | `simulate(df, n_paths)` |
| `backtester/combinatorial_cv.py` | Combinatorial parameter tuning validation | `combinatorial_cv(...)` |

### Agents

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `agents/meta_agent.py` | Ensemble vote over all specialized agents | `MetaAgent.run(ticker)` → `AgentSignal` |
| `agents/regime_agent.py` | Regime classifier + strategy router | `RegimeAgent.run(ticker)` |
| `agents/risk_agent.py` | Risk constraint evaluator | `RiskAgent.run(account)` |
| `agents/knowledge_agent.py` | Model staleness / IC / drift verdict | `KnowledgeAgent.run(model_id)` → `fresh\|monitor\|retrain` |
| `agents/sentiment_agent.py` | Sentiment-based trading signal | `SentimentAgent.run(ticker)` |
| `agents/screener_agent.py` | Equity screening by momentum/factors | `ScreenerAgent.run(universe)` |
| `agents/execution_agent.py` | Order execution optimizer | `ExecutionAgent.run(order)` |
| `agents/knowledge_registry.py` | Model zoo registry for multi-family auditing | `KnowledgeRegistry` class |

### Infrastructure

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `broker/paper_trader.py` | Offline paper trading | `buy()`, `sell()`, `get_positions()` |
| `scheduler/alerts.py` | APScheduler alert + knowledge-health engine | `start_scheduler()` |
| `journal/trading_journal.py` | Trade recording & analytics | `log_trade()`, `get_trades()` |
| `bus/event_bus.py` | Pub-sub event dispatcher | `EventBus.get()`, `.publish()`, `.subscribe()` |
| `audit/logger.py` | Structured audit trail | `audit_log(action, context)` |
| `monitoring/metrics.py` | Prometheus metrics exporter | `record_signal()`, `record_trade()` |
| `providers/market_data.py` | Market data DI factory | `get_market_data()` |
| `screener/screener.py` | Factor/momentum stock screening | `screen(criteria)` |

---

## Agents Architecture

The `agents/` package implements a decision-agent layer on top of the quant modules. Each specialist agent follows the `AgentBase` protocol (`agents/base.py`) and returns a typed `AgentSignal`.

```python
from agents.meta_agent import MetaAgent

signal = MetaAgent().run("AAPL")
# signal.verdict: "buy" | "sell" | "hold"
# signal.confidence: float 0–1
# signal.regime: "trending_bull" | "trending_bear" | "mean_revert" | "high_vol"
```

The **knowledge agent** (`agents/knowledge_agent.py`) audits model health by combining live IC, PSI drift score, and days-since-retrain into a verdict: `fresh` | `monitor` | `retrain`. The **knowledge registry** (`agents/knowledge_registry.py`) tracks all deployed model families and is consumed by `pages/model_health.py`.

---

## ML Alpha Pipeline

The full ML signal lifecycle:

```
data/features.py          →  feature matrix (MultiIndex date×ticker)
analysis/factor_ic.py     →  IC/ICIR evaluation per feature
analysis/triple_barrier.py →  target labels
strategies/ml_signal.py   →  LightGBM model (regime-conditioned)
strategies/ml_tuning.py   →  Optuna HPO (purged CV, no leakage)
backtester/walk_forward.py →  out-of-sample validation
strategies/ml_execution.py →  position sizing (Kelly × regime × |score|)
cron/monthly_ml_retrain.py →  scheduled retraining (1st of month)
cron/daily_ml_execute.py  →  daily scoring + order submission (16:05 ET)
```

Use the `/ml-experiment` skill to run this pipeline end-to-end for a new signal. Use `/run-experiment` to run a single phase (ic, tune, or wf) via the experiment-tracker subagent.

**Key invariants:**
- Always use purged walk-forward CV (`backtester/walk_forward.py`) — never `train_test_split`
- Labels come from `analysis/triple_barrier.py` only
- Position sizing goes through `strategies/ml_execution.py` — never size directly from raw scores
- Model staleness is gated by `agents/knowledge_agent.py` before live execution

---

## Code Conventions

### Naming
- `snake_case` for functions, variables, modules, file names
- `PascalCase` for classes
- `UPPER_CASE` for module-level constants
- `_leading_underscore` for module-private helpers
- Prefer `ticker` over `symbol` throughout the codebase

### File Organization
- One primary class or responsibility per file
- Imports ordered: stdlib → third-party → local (PEP 8)
- `from __future__ import annotations` at top of files using forward references
- Module docstring at top of every file explaining purpose and relevant env vars

### Error Handling
- Return empty/default values when optional credentials are absent — don't raise at module import time
- Validate at system boundaries (user input, external API responses)
- Use graceful fallbacks (e.g., realtime feed falls back from Alpaca WS to yfinance polling)
- Don't add defensive error handling for internal code paths that can't fail

### Type Hints
- Use type hints on all public function signatures
- Use `Protocol` for interfaces (not ABC)
- Prefer `list[str]` over `List[str]` (Python 3.9+ style)
- mypy is enforced in CI for Phase-1 modules (`providers/`, `risk/`, `bus/`, `journal/`)

### Security
- **Never hardcode secrets** — all credentials loaded from `.env` via `os.getenv()`
- **Never log secrets** — sanitize before logging
- `.env` is in `.gitignore`; commit only `.env.example` with placeholder values
- Default broker URLs always point to paper/sandbox endpoints

---

## Environment Variables

Copy `.env.example` to `.env` and populate the relevant keys. Key variables:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `console` | `console` or `json` |
| `MARKET_DATA_PROVIDER` | `yfinance` | `alpaca`, `yfinance`, `polygon`, `mock` |
| `ALPACA_API_KEY` | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | — | Alpaca secret |
| `ALPACA_BASE_URL` | paper URL | Override for live trading |
| `ALPACA_PAPER` | `true` | Set `false` only for live trading |
| `POLYGON_API_KEY` | — | Polygon.io API key (historical backfill) |
| `CCXT_EXCHANGE` | — | Exchange ID (e.g., `binance`) |
| `TELEGRAM_BOT_TOKEN` | — | Telegram alert bot |
| `TELEGRAM_CHAT_ID` | — | Target Telegram chat |
| `EMAIL_SMTP_HOST` | — | SMTP server for email alerts |
| `PAPER_STARTING_CASH` | `100000` | Paper trading starting balance |
| `WF_TICKERS` | — | Comma-separated tickers for walk-forward cron |
| `MAX_DRAWDOWN_PCT` | — | Alert threshold for drawdown |
| `MAX_GROSS_EXPOSURE` | — | Pre-trade guard: max gross notional exposure |
| `MAX_DAILY_LOSS_PCT` | — | Pre-trade guard: max daily loss % before killswitch |
| `KILLSWITCH_FILE` | — | Path to killswitch sentinel file |
| `MLFLOW_URI` | — | MLflow tracking server URI |
| `DUCKDB_PATH` | — | Path for DuckDB time-series cache |
| `KNOWLEDGE_AUTO_RETRAIN` | `false` | Auto-trigger retrain on `retrain` verdict |
| `LIVE_IC_BACKFILL_CRON` | — | Cron expression for live-IC backfill job |

---

## Testing Conventions

- Test files: `tests/test_<module_name>.py`
- One test file per source module
- Mock all external APIs (yfinance, broker APIs) — tests must not require network
- Use `@pytest.mark.integration` for tests that need live credentials (excluded from CI unit run)
- Use `@pytest.mark.e2e` for full-chain tests (run in the separate `e2e` CI job)
- `conftest.py` sets `OPENBLAS_NUM_THREADS=1` to prevent BLAS thread contention in CI
- Fixtures for DB setup should use in-memory SQLite (`:memory:`) or temp files
- Aim to keep unit tests fast (< 1s each); e2e tests have a hard 3 s per-test budget

### Bug-regression discipline (#230)

Every fixed bug ships with a permanent regression test. Tag the test
with the issue number using a `# regression test for #NNN` comment
on the line above the test function so reviewers can find it during
code review:

```python
# regression test for #183 — PreTradeGuard read `equity` only,
# missing the paper broker's `total_value` key
def test_guard_accepts_paper_broker_equity():
    ...
```

The PR description must call out either the regression test OR (in
rare cases — e.g. the bug is environmental and impossible to
reproduce in a test) explicitly explain why a regression test
isn't possible. The PR template at
[`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md)
carries the checkbox.

### Negative-test discipline (#231)

Every public function on a critical path (`broker/`, `journal/`,
`risk/`, `audit/`, `bus/`) ships **at least one happy-path test
AND at least one failure-mode test** (raise, return None, empty
input, etc.). The Phase-1 e2e injection fixtures
(`inject_broker_failure`, `inject_journal_failure`,
`trip_killswitch` in `tests/conftest.py`) are the unit-suite
equivalent — pull them in with the same fixture-based pattern
where possible.

### Synthetic-data factories (#239)

[`tests/factories.py`](tests/factories.py) is the **single source
of truth** for synthetic test data. New tests pull the helper that fits:

```python
from tests.factories import make_ohlcv, make_returns, make_prices, make_feature_matrix

df  = make_ohlcv(n=60, seed=7)              # OHLCV DataFrame (capitalised columns)
r   = make_returns(n=252, sigma=0.02)       # daily-return series
p   = make_prices(n=200, last=110.0)        # constant-price for SMA tests
fmx = make_feature_matrix(n=100, tickers=["AAPL", "MSFT"], seed=42)  # MultiIndex features
```

Each factory is fully deterministic given its `seed` argument so
tests don't depend on the determinism trio (#227) for stability.
Legacy in-file helpers in older `tests/test_*.py` will migrate
over time; new tests should reach for `tests.factories` first.

### Running a Specific Test

```bash
pytest tests/test_greeks.py -v
pytest tests/test_backtester.py::test_sma_crossover -v
```

---

## Adding New Features

### New Broker Integration

1. Create `broker/<name>_bridge.py` implementing the broker interface
2. Create `adapters/broker/<name>_adapter.py` wrapping it with the provider Protocol
3. Register in `providers/broker.py` factory
4. Add env vars to `.env.example` and document in `MAINTENANCE_AND_BROKERS.md`
5. Write `tests/test_<name>_bridge.py` with mocked API responses

### New Streamlit Tab

1. Create `pages/<name>.py` with a `render() -> None` function
2. Import and add to `app.py` tab list
3. Use `st.session_state` for state shared with sidebar

### New Strategy/Indicator

1. Add indicator function to `strategies/indicators.py` (or new file for complex strategies)
2. Wire into `backtester/engine.py` if it needs backtesting support
3. Add tests in `tests/test_indicators.py` or a new test file

### New ML Signal

1. Add features to `data/features.py`
2. Measure IC with `analysis/factor_ic.py`
3. Implement signal class in `strategies/` following `MLSignal` interface
4. Tune with `strategies/ml_tuning.py` (Optuna + purged CV)
5. Validate with `backtester/walk_forward.py`
6. Wire sizing through `strategies/ml_execution.py`
7. Register model in `agents/knowledge_registry.py`
8. Use `/ml-experiment` or `/run-experiment` skills for guided execution

### New Alert Channel

1. Implement in `alerts/channels.py`
2. Create adapter in `adapters/alert/`
3. Register in `providers/alert.py`

---

## Plan Review Workflow

Before implementing a non-trivial plan, run the plan through the
[`trading-philosophy-reviewer`](.claude/agents/trading-philosophy-reviewer.md)
sub-agent. It audits the draft against `TRADING_PHILOSOPHY.md` (three
pillars, decision stack §7, anti-patterns §10) and the codebase
conventions above (DI via `providers/`, `data/fetcher.py` for OHLCV,
`data/db.py:get_connection()` for `quant.db`, structlog, no hardcoded
secrets), and writes a durable record under `docs/reviews/`.

**Invocation (explicit, no settings hook):**

```
/review-plan /root/.claude/plans/<slug>.md
```

The slash command dispatches the sub-agent, which writes
`docs/reviews/YYYY-MM-DD-<slug>.md` and returns a 5-line summary
(overall verdict + per-dimension verdict). The reviewer is **advisory** —
it never hard-blocks and never edits plan or source files. It also does
not run `ruff`/`pytest`/`bandit`/`pip-audit`; that is the `/pre-push`
skill's job. Address `major` findings before implementation; `minor`
items should be acknowledged but are non-blocking.

---

## Branch Discipline — new implementation starts a new branch

Every new implementation begins on its own feature branch off `origin/main`.
This is enforced by the [`new-branch`](.claude/skills/new-branch/SKILL.md)
skill, which refuses to start work on `main`, on a stale feature branch, or
with a dirty working tree.

**Invocation:**

```
/new-branch <issue-number | roadmap-id | slug>
```

Naming convention — `claude/<identifier>-<kebab-slug>`:
- Issue → `claude/issue-139-pretrade-risk-guard`
- Roadmap ticket → `claude/p1-1-pretrade-risk-guard`
- Free-form → `claude/<slug>-<rand4>`

The skill only runs read/write git — it never edits code, never pushes,
never commits. Together with `/pre-push` at the end of the cycle, it
brackets every implementation. Claude is expected to invoke `/new-branch`
as its first action whenever the user asks to start a new ticket or
implementation.

---

## Release Process

```bash
# Ensure all tests pass on main
git tag v1.2.3
git push origin v1.2.3
```

The `release.yml` workflow automatically:
1. Runs full test suite
2. Builds and pushes Docker image to `ghcr.io/ghostlobster/quant-platform`
3. Creates a GitHub Release with auto-generated changelog

---

## Key Documentation Files

| File | Contents |
|---|---|
| `README.md` | Quick start, feature list, release process |
| `PLAN.md` | Architecture overview, build roadmap, security checklist |
| `TRADING_PHILOSOPHY.md` | Trading indicators, risk management, decision framework, anti-patterns |
| `IMPLEMENTATION_SUMMARY.md` | Feature progress tracker (P1–P5), coverage status |
| `MAINTENANCE_AND_BROKERS.md` | Broker landscape, integration guide, maintenance playbook |
| `ML_BOOK_MAP.md` | AFML/ML4T chapter cross-reference with implementation map |
| `cron/README.md` | Monthly walk-forward and daily execution cron setup |
| `deploy/README.md` | Deployment notes (Docker, Kubernetes Helm, supervisord) |

---

## Common Pitfalls to Avoid

- **Do not** import concrete adapters directly in business logic — always go through `providers/`
- **Do not** call `yfinance` directly outside `data/fetcher.py` — use `fetch_ohlcv()` to benefit from caching
- **Do not** open raw SQLite connections in page/strategy code — use `data/db.py:get_connection()` for `quant.db`
- **Do not** hardcode ticker lists — read from the watchlist or `WF_TICKERS` env var
- **Do not** use `st.experimental_*` APIs — prefer stable Streamlit APIs
- **Do not** store sensitive data in `st.session_state` across sessions
- **Do not** add blocking I/O in Streamlit render functions without spinner context (`st.spinner`)
- **Do not** skip tests or lower the coverage threshold — fix the underlying issue instead
- **Do not** use `train_test_split` for ML model validation — use purged walk-forward CV (`backtester/walk_forward.py`)
- **Do not** size positions directly from raw ML scores — route through `strategies/ml_execution.py`
- **Do not** publish live orders without passing through `risk/pretrade_guard.py`
- **Do not** cross-import between `pages/`, `strategies/`, and `risk/` — use `bus/event_bus.py` for decoupled communication
- **Do not** call agent methods from Streamlit render functions synchronously on large universes — use `st.spinner` and cache results in `st.session_state`
