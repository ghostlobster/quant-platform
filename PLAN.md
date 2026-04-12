# Quant Trading Platform — Personal Use
**Budget:** Free / Open-Source only  
**Target:** macOS local environment  
**Last updated:** 2026-04-08

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                              │
│   Dashboard │ Screener │ Backtest Runner │ Live Monitor          │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐ ┌──────────┐ ┌───────────────┐
   │  Data Layer │ │ Strategy │ │  Broker Layer  │
   │  (yfinance/ │ │  Engine  │ │  (Alpaca free  │
   │  CCXT free) │ │(vectorbt)│ │  paper trading)│
   └──────┬──────┘ └────┬─────┘ └───────┬───────┘
          │             │               │
          └─────────────┴───────────────┘
                         │
                  ┌──────▼──────┐
                  │  SQLite DB  │
                  │  (local)    │
                  └─────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| **Data Ingestion** | Pull OHLCV data via yfinance (free, no key needed) |
| **Strategy Engine** | Define and backtest strategies using vectorbt |
| **Backtester** | Vectorbt for fast vectorized backtesting |
| **Paper Broker** | Alpaca Markets free paper-trading API (no real money) |
| **Database** | SQLite for storing signals, trades, backtest results |
| **UI** | Streamlit dashboard (local web app) |
| **Scheduler** | APScheduler for periodic data fetches & signal generation |
| **Notifications** | Optional: local desktop notifications via plyer |

---

## 2. Tech Stack (All Free / Open-Source)

| Tool | Version | Purpose | Cost |
|------|---------|---------|------|
| Python | 3.11+ | Core language | Free |
| yfinance | latest | Market data (Yahoo Finance) | Free |
| vectorbt | latest | Vectorized backtesting | Free |
| pandas / numpy | latest | Data manipulation | Free |
| Streamlit | latest | Web UI (local) | Free |
| Plotly | latest | Interactive charts | Free |
| SQLite (sqlite3) | built-in | Local data storage | Free |
| python-dotenv | latest | Env var management | Free |
| APScheduler | latest | Job scheduling | Free |
| alpaca-trade-api | latest | Paper trading (free tier) | Free |
| ta-lib / pandas-ta | latest | Technical indicators | Free |
| plyer | latest | Desktop notifications | Free |
| pytest | latest | Testing | Free |

---

## 3. Step-by-Step Build Order

### Step 1 — Project Scaffold *(current step)*
- Create folder structure
- Set up virtualenv + install deps
- `.env.example`, `.gitignore`
- Basic Streamlit app: title + AAPL price chart from yfinance

### Step 2 — Data Layer
- `data/fetcher.py`: fetch OHLCV data with caching to SQLite
- Support multiple tickers and date ranges
- Streamlit page: data browser with ticker input

### Step 3 — Technical Indicators
- `strategies/indicators.py`: SMA, EMA, RSI, MACD, Bollinger Bands via pandas-ta
- Streamlit page: indicator overlay on price chart

### Step 4 — Strategy Framework
- `strategies/base.py`: abstract Strategy class
- `strategies/sma_crossover.py`: first concrete strategy (SMA cross)
- `strategies/rsi_mean_reversion.py`: second strategy

### Step 5 — Backtesting Engine
- `backtester/engine.py`: vectorbt wrapper
- Run backtests, generate equity curves, stats
- Streamlit page: backtest runner + results (Sharpe, max drawdown, etc.)

### Step 6 — Stock Screener
- `screener/screener.py`: scan universe of tickers for signals
- Streamlit page: screener results table with filters

### Step 7 — Paper Trading Integration
- `broker/alpaca_paper.py`: connect Alpaca paper API (free, no real money)
- Submit orders, track positions
- Streamlit page: live positions + P&L

### Step 8 — Scheduler & Alerts
- `scheduler/jobs.py`: APScheduler for periodic scans (market hours)
- Desktop notifications for new signals

### Step 9 — Portfolio Analytics
- Correlation matrix, sector exposure, drawdown analysis
- Streamlit page: portfolio dashboard

### Step 10 — Hardening & Polish
- Unit tests (pytest)
- Logging to file
- README with setup instructions
- Export backtest results to CSV/PDF

---

## 4. Cost Breakdown

| Item | Cost | Notes |
|------|------|-------|
| Python + all libraries | **Free** | All open-source |
| yfinance market data | **Free** | Yahoo Finance scraper; rate limits apply |
| Alpaca paper trading | **Free** | Paper account, no real money |
| SQLite database | **Free** | Built into Python |
| Streamlit (local) | **Free** | Running locally, not on cloud |
| Streamlit Cloud (optional) | $0–$20/mo | Only if you want remote access |
| Alpaca live trading | **Free** (no commission) | Only if/when you graduate to live; ask me first |
| VPS hosting (optional) | ~$5/mo | Only if you want 24/7 scheduling; ask me first |

**Current estimated cost: $0**

---

## 5. Security Checklist

- [ ] All API keys in `.env` file — never in source code
- [ ] `.env` listed in `.gitignore`
- [ ] `.env.example` committed with placeholder values only
- [ ] Alpaca paper API keys (not live) used during development
- [ ] Live trading keys never stored — only paper keys in `.env`
- [ ] SQLite database file in `.gitignore`
- [ ] No credentials logged to console or log files
- [ ] Secrets loaded via `python-dotenv` at runtime only
- [ ] Review all third-party packages for supply chain risk before installing
- [ ] Ask before enabling any live trading feature

---

## 6. Folder Structure

```
quant-platform/
├── PLAN.md
├── README.md
├── .env.example          # Template — commit this
├── .env                  # Real secrets — NEVER commit
├── .gitignore
├── requirements.txt
├── app.py                # Streamlit entry point
├── config.py             # App configuration (reads .env)
├── data/
│   ├── __init__.py
│   ├── fetcher.py        # yfinance wrapper + caching
│   └── db.py             # SQLite helpers
├── strategies/
│   ├── __init__.py
│   ├── base.py           # Abstract strategy
│   ├── indicators.py     # Technical indicators
│   ├── sma_crossover.py
│   └── rsi_mean_reversion.py
├── backtester/
│   ├── __init__.py
│   └── engine.py         # vectorbt wrapper
├── screener/
│   ├── __init__.py
│   └── screener.py
├── broker/
│   ├── __init__.py
│   └── alpaca_paper.py   # Paper trading only
├── scheduler/
│   ├── __init__.py
│   └── jobs.py
├── ui/
│   ├── pages/
│   │   ├── 01_Data_Browser.py
│   │   ├── 02_Indicators.py
│   │   ├── 03_Backtester.py
│   │   ├── 04_Screener.py
│   │   ├── 05_Paper_Trading.py
│   │   └── 06_Portfolio.py
│   └── components/       # Reusable UI widgets
├── tests/
│   └── test_data.py
└── logs/
    └── .gitkeep
```
