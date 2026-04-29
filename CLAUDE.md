# CLAUDE.md — AI Assistant Guide for quant-platform

This file provides context for AI assistants (Claude, Copilot, etc.) working in this repository.

## Project Overview

A production-ready quantitative trading and analytics platform built with **Python 3.11** and **Streamlit**. It supports multi-broker live/paper trading, backtesting, options analytics, portfolio risk, and market screening — all behind a single web dashboard.

**Entry point:** `streamlit run app.py` → http://localhost:8501

---

## Repository Structure

```
quant-platform/
├── app.py                  # Streamlit entry point; bootstraps all subsystems
├── config.py               # Loads .env vars, configures structlog
├── requirements.txt        # 63 pinned Python dependencies
├── ruff.toml               # Linter config (line-length 100, E/F/W/I rules)
├── pytest.ini              # Test discovery config
├── Dockerfile              # python:3.11-slim, healthcheck on port 8501
├── docker-compose.yml      # streamlit service + alerts daemon
├── run.sh                  # Local dev launcher (venv + streamlit)
│
├── adapters/               # Pluggable adapters (implement provider protocols)
│   ├── broker/             # Broker adapters (Alpaca, IBKR, Schwab, Tradier, CCXT)
│   ├── market_data/        # Market data adapters (alpaca, yfinance, mock)
│   ├── alert/              # Alert channel adapters
│   ├── llm/                # LLM adapters
│   └── tsdb/               # Time-series DB adapters
│
├── alerts/                 # Notification channels: Telegram, Email, Slack, Webhook
├── analysis/               # Quant analytics: Greeks, risk metrics, regime detection
├── backtester/             # Event-driven backtester, walk-forward, Monte Carlo
├── broker/                 # Direct broker integrations + paper trading engine
├── cron/                   # Scheduled jobs (monthly walk-forward runner)
├── data/                   # Data fetching, caching, watchlist (SQLite-backed)
├── deploy/                 # supervisord configs and deployment helpers
├── journal/                # Trading journal (entry/exit metadata, analytics)
├── pages/                  # One Streamlit tab per file + shared sidebar
├── providers/              # Protocol definitions + factory functions (DI layer)
├── risk/                   # VaR, CVaR, Kelly criterion, Markowitz optimization
├── scheduler/              # APScheduler alert engine
├── screener/               # Equity screening by momentum / factor criteria
├── strategies/             # Technical indicators and trading strategy logic
├── tests/                  # 39 test files, 76%+ coverage enforced in CI
└── utils/                  # Logging helpers
```

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| UI | Streamlit 1.56.0 |
| Data | pandas 3.0.2, numpy 2.4.4, yfinance 1.2.1 |
| Charts | plotly 6.6.0 |
| Indicators | ta 0.11.0 |
| Crypto | ccxt 4.0+ (100+ exchanges) |
| Brokers | Alpaca, IBKR (IB Gateway), Schwab, Tradier |
| Scheduling | APScheduler 3.11.2 |
| Database | SQLite (WAL mode) via stdlib `sqlite3` |
| Logging | structlog 24.0+ |
| Linting | ruff |
| Security | bandit, pip-audit |
| Testing | pytest 9.0.3, pytest-cov |
| CI/CD | GitHub Actions |
| Deploy | Docker + docker-compose |

---

## Development Workflow

### Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in API keys
bash run.sh                 # starts Streamlit on :8501
```

### Running Tests

```bash
# Unit tests only (fast, no external services required)
pytest tests/ -m "not integration"

# With coverage report
pytest tests/ -m "not integration" --cov=. --cov-report=term-missing

# Integration tests (require live credentials)
pytest tests/ -m "integration"
```

**CI enforces `--cov-fail-under=76`.** Keep coverage above this threshold when adding code.

For a single-command CI-mirror run (ruff + pytest 76% + bandit HIGH + pip-audit) from inside Claude Code, invoke the `/pre-push` skill.

### Linting

```bash
ruff check .          # check for issues
ruff check . --fix    # auto-fix where possible
```

Config: `ruff.toml` — line-length 100, rules E/F/W/I, E501 ignored.

### Security Scanning

```bash
bandit -r . -ll --exclude ./.git,./tests
pip-audit -r requirements.txt
```

Bandit fails CI only on HIGH severity findings. `PYSEC-2022-42969` is allowlisted in pip-audit.

### Docker

```bash
docker-compose up          # streamlit (:8501) + alerts daemon
docker-compose up streamlit  # UI only
```

---

## CI/CD Pipelines

All pipelines live in `.github/workflows/`.

| Workflow | Trigger | Steps |
|---|---|---|
| `ci.yml` | PR → main, push to main | lint → security → unit tests → integration tests → coverage comment on PR |
| `build.yml` | push to main | full test suite → Docker build → validate compose |
| `release.yml` | tag `v*.*.*` | tests → Docker push to GHCR → GitHub Release + changelog |

**Never skip CI.** Fix lint/test failures rather than using `--no-verify` or bypass flags.

---

## Architecture Patterns

### Dependency Injection via Providers

The `providers/` directory defines `Protocol` classes and factory functions. Concrete implementations live in `adapters/`. Switch implementations with env vars — no code changes required.

```python
# providers/market_data.py
from providers.market_data import get_market_data

provider = get_market_data()          # reads MARKET_DATA_PROVIDER env var
bars = provider.get_bars("AAPL", "1Day", "2024-01-01", "2024-12-31")
```

**Providers and their env var selectors:**

| Provider | Env Var | Options |
|---|---|---|
| Market data | `MARKET_DATA_PROVIDER` | `alpaca`, `yfinance` (default), `mock` |
| Broker | `BROKER_PROVIDER` | `alpaca`, `ibkr`, `schwab`, `tradier`, `ccxt`, `paper` |
| Alerts | `ALERT_PROVIDER` | `telegram`, `email`, `slack`, `webhook` |
| LLM | `LLM_PROVIDER` | various |

Always code against the Protocol interface, never import a concrete adapter directly in business logic.

### Data Fetching & Caching

`data/fetcher.py` is the single source of OHLCV data. It checks `data/price_cache` in SQLite before hitting yfinance.

```python
from data.fetcher import fetch_ohlcv

df = fetch_ohlcv("AAPL", "6mo")   # returns pandas DataFrame
```

Cache TTLs: intraday 1h, short-term 4h, historical 24h. Data is stored as JSON in SQLite.

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

| Module | Purpose | Key Entrypoint |
|---|---|---|
| `data/fetcher.py` | OHLCV data with caching | `fetch_ohlcv(ticker, period)` |
| `data/db.py` | SQLite connection factory | `get_connection()`, `init_db()` |
| `data/watchlist.py` | User ticker watchlists | `get_watchlist()`, `add_ticker()` |
| `data/realtime.py` | Real-time price feed | `RealtimeFeed` class |
| `strategies/indicators.py` | SMA, EMA, RSI, MACD, BB | Function per indicator |
| `analysis/greeks.py` | Black-Scholes Greeks | `delta()`, `gamma()`, etc. |
| `analysis/risk_metrics.py` | VaR, CVaR, Sharpe, Sortino | `var()`, `cvar()`, `sharpe()` |
| `analysis/regime.py` | 4-state market regime classifier | `classify_regime(df)` |
| `risk/portfolio_risk.py` | Kelly, Markowitz optimization | `kelly_fraction()`, `optimize()` |
| `backtester/engine.py` | Event-driven backtester | `run_backtest(strategy, df)` → `BacktestResult` |
| `backtester/walk_forward.py` | Walk-forward validation | `run_walk_forward(...)` |
| `backtester/monte_carlo.py` | Bootstrap simulation | `simulate(df, n_paths)` |
| `broker/paper_trader.py` | Offline paper trading | `buy()`, `sell()`, `get_positions()` |
| `scheduler/alerts.py` | APScheduler alert engine | `start_scheduler()` |
| `screener/` | Factor/momentum stock screening | `screen(criteria)` |
| `journal/trading_journal.py` | Trade recording & analytics | `log_trade()`, `get_trades()` |
| `providers/market_data.py` | Market data DI factory | `get_market_data()` |

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
| `MARKET_DATA_PROVIDER` | `yfinance` | `alpaca`, `yfinance`, `mock` |
| `ALPACA_API_KEY` | — | Alpaca API key |
| `ALPACA_SECRET_KEY` | — | Alpaca secret |
| `ALPACA_BASE_URL` | paper URL | Override for live trading |
| `ALPACA_PAPER` | `true` | Set `false` only for live trading |
| `CCXT_EXCHANGE` | — | Exchange ID (e.g., `binance`) |
| `TELEGRAM_BOT_TOKEN` | — | Telegram alert bot |
| `TELEGRAM_CHAT_ID` | — | Target Telegram chat |
| `EMAIL_SMTP_HOST` | — | SMTP server for email alerts |
| `PAPER_STARTING_CASH` | `100000` | Paper trading starting balance |
| `WF_TICKERS` | — | Comma-separated tickers for walk-forward cron |
| `MAX_DRAWDOWN_PCT` | — | Alert threshold for drawdown |

---

## Testing Conventions

- Test files: `tests/test_<module_name>.py`
- One test file per source module
- Mock all external APIs (yfinance, broker APIs) — tests must not require network
- Use `@pytest.mark.integration` for tests that need live credentials; these are excluded from CI unit test run
- `conftest.py` sets `OPENBLAS_NUM_THREADS=1` to prevent BLAS thread contention in CI
- Fixtures for DB setup should use in-memory SQLite (`:memory:`) or temp files
- Aim to keep unit tests fast (< 1s each)

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
of truth** for synthetic OHLCV / returns / prices / feature
matrices. New tests pull the helper that fits:

```python
from tests.factories import make_ohlcv, make_returns, make_prices

df = make_ohlcv(n=60, seed=7)         # canonical capitalised-column OHLCV
r  = make_returns(n=252, sigma=0.02)  # daily-return series
p  = make_prices(n=200, last=110.0)   # constant-price for SMA tests
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
| `IMPLEMENTATION_SUMMARY.md` | Feature progress tracker (P1–P4), coverage status |
| `MAINTENANCE_AND_BROKERS.md` | Broker landscape, integration guide, maintenance playbook |
| `cron/README.md` | Monthly walk-forward cron setup |
| `deploy/README.md` | Deployment notes |

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
