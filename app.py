"""
Quant Platform — Main Streamlit Entry Point
Run with: streamlit run app.py
"""
import streamlit as st

import config
from broker.paper_trader import init_paper_tables
from data.db import init_db
from journal.trading_journal import init_journal_table
from pages import (
    alerts,
    backtest,
    chart,
    efficient_frontier,
    journal_tab,
    portfolio,
    screener,
    shared,
)
from scheduler.alerts import init_alerts_table

# ── App bootstrap ─────────────────────────────────────────────────────────────
config.configure_logging()
init_db()
init_paper_tables()
init_alerts_table()
init_journal_table()

st.set_page_config(
    page_title="Quant Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

shared.render_sidebar()

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Chart", "🔬 Backtest", "🔍 Screener", "💼 Portfolio", "🔔 Alerts", "📓 Journal", "📐 Efficient Frontier"
])
with tab1:
    chart.render()
with tab2:
    backtest.render()
with tab3:
    screener.render()
with tab4:
    portfolio.render()
with tab5:
    alerts.render()
with tab6:
    journal_tab.render()
with tab7:
    efficient_frontier.render()
