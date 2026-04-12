# Quant Platform

A multi-feature quantitative finance dashboard built with Streamlit.

## Setup

```bash
git clone <repo-url>
cd quant-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
bash run.sh
```

The app will be available at **http://localhost:8501**.

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
