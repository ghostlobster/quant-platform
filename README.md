# Quant Platform

A multi-feature quantitative finance dashboard built with Streamlit (Python 3.11).

## Quick Start

```bash
git clone <repo-url>
cd quant-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # add your API keys
pip install pre-commit            # one-time per clone
pre-commit install                # wires gitleaks into git commit
bash run.sh
```

The app is served at **http://localhost:8501**.

## Configuration

All runtime configuration is read from environment variables. Copy
`.env.example` to `.env` and fill in the values you need —
`.env.example` is the canonical source of truth and groups variables by
purpose:

- **Brokers** — Alpaca, Tradier, IBKR, Schwab, CCXT
- **Market data** — Polygon, TSDB choice (sqlite vs duckdb), backfill
- **Model registry** — MLflow path/URI, per-model overrides
- **Alerts** — Telegram, Email (SMTP), Slack, Webhook
- **Scheduler / cron** — walk-forward tickers, drawdown thresholds, paper cash
- **Risk guards** — pre-trade caps, daily loss %, gross exposure, killswitch
- **Paper promotion** — minimum days / Sharpe before promoting to live
- **Event bus** — Redis Streams (optional)
- **App settings** — `LOG_LEVEL`, `LOG_FORMAT`, `APP_ENV`, executor backend

The variables most users touch first:

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | `development` or `production` |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_FORMAT` | `console` | `console` or `json` (use `json` in prod) |
| `MARKET_DATA_PROVIDER` | `yfinance` | `alpaca`, `yfinance`, `mock` |
| `BROKER_PROVIDER` | `paper` | `alpaca`, `ibkr`, `schwab`, `tradier`, `ccxt`, `paper` |
| `ALERT_PROVIDER` | — | `telegram`, `email`, `slack`, `webhook` |
| `PAPER_STARTING_CASH` | `100000` | Paper trading starting balance |
| `WF_TICKERS` | — | Comma-separated tickers for walk-forward cron |
| `MAX_DRAWDOWN_PCT` | — | Alert threshold for drawdown |

Default broker URLs always point at paper/sandbox endpoints. Never commit
`.env` — only `.env.example` (with placeholders) is checked in.

## Running the App

### Local

```bash
bash run.sh                       # streamlit on :8501
```

### Docker

```bash
docker-compose up                 # full stack (streamlit + alerts + metrics)
docker-compose up streamlit       # UI only
```

`docker-compose.yml` exposes:

| Port | Service |
|---|---|
| 8501 | Streamlit dashboard |
| 9090 | metrics sidecar |
| 9091 | Prometheus |
| 3000 | Grafana |
| 5000 | MLflow |

## Testing

```bash
# Fast unit tests (no network, no live credentials)
pytest tests/ -m "not integration and not e2e"

# With coverage report
pytest tests/ -m "not integration and not e2e" --cov=. --cov-report=term-missing

# Integration tests (require live broker / market-data credentials)
pytest tests/ -m "integration"

# End-to-end tests
pytest tests/ -m "e2e"

# A single test
pytest tests/test_greeks.py::test_delta_call -v
```

CI enforces `--cov-fail-under=76` (line + branch combined) — keep coverage
above this threshold when adding code. Use `@pytest.mark.integration` for
tests that need live credentials; they are excluded from the default unit
run. Test config lives in `pytest.ini` and `.coveragerc`.

## Linting

```bash
ruff check .                      # report violations
ruff check . --fix                # auto-fix where possible
```

Configuration: `ruff.toml` — line-length 100, rules `E/F/W/I`, `E501` ignored.

## Security Scanning

```bash
bandit -r . -ll --exclude ./.git,./tests
pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969
```

CI fails only on **bandit HIGH** findings. `PYSEC-2022-42969` is allowlisted
for `pip-audit`.

`pre-commit install` (from Quick Start) wires `gitleaks` to scan staged
diffs for secrets before every `git commit`. CI re-runs `gitleaks` on every
PR via `gitleaks/gitleaks-action` so contributors without the local hook
are still caught at PR time. False positives go in `.gitleaks.toml`.

## CI Gate

Every PR must pass the four merge-gate checks defined in
`.github/workflows/ci.yml`:

1. `ruff check .`
2. `bandit -r . -ll --exclude ./.git,./tests` (fail on HIGH)
3. `pip-audit -r requirements.txt --ignore-vuln PYSEC-2022-42969`
4. `pytest tests/ -m "not integration and not e2e" --cov=. --cov-fail-under=76`

To run all four locally before pushing, invoke the `/pre-push` skill from
inside Claude Code — it mirrors the CI commands against your working tree
and reports pass/fail per stage.

## Features

1. **Portfolio Tracker** — monitor holdings, P&L, and allocation by asset class
2. **Options Pricing** — Black-Scholes and binomial model calculators with Greeks
3. **Risk Analytics** — VaR, CVaR, Sharpe/Sortino ratios, drawdown analysis
4. **Backtesting Engine** — test strategies on historical data with performance metrics
5. **Technical Analysis** — candlestick charts, RSI, MACD, Bollinger Bands, moving averages
6. **Market Scanner** — screen equities by momentum, value, or custom factor criteria
7. **Correlation Matrix** — heatmap of asset correlations across configurable windows
8. **Monte Carlo Simulator** — price-path simulation for portfolio and options pricing
9. **News Sentiment** — NLP-scored headlines aggregated by ticker
10. **Auto-refresh** — configurable sidebar timer (1 min / 5 min / 15 min / 30 min) to keep live data current

## Documentation

| File | Contents |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | AI-assistant guide & deeper conventions |
| [`PLAN.md`](PLAN.md) | Architecture overview & build roadmap |
| [`TRADING_PHILOSOPHY.md`](TRADING_PHILOSOPHY.md) | Three pillars, decision stack, anti-patterns |
| [`IMPLEMENTATION_SUMMARY.md`](IMPLEMENTATION_SUMMARY.md) | Feature progress (P1–P4) |
| [`MAINTENANCE_AND_BROKERS.md`](MAINTENANCE_AND_BROKERS.md) | Broker landscape & maintenance playbook |
| [`cron/README.md`](cron/README.md) | Monthly walk-forward cron setup |
| [`deploy/README.md`](deploy/README.md) | Deployment notes |

## Making a Release

1. Ensure all tests pass on `main`
2. Tag the release: `git tag v1.0.0 && git push origin v1.0.0`
3. The Release workflow runs automatically:
   - Runs full test suite
   - Builds and pushes Docker image to `ghcr.io/ghostlobster/quant-platform`
   - Creates a GitHub Release with auto-generated changelog
