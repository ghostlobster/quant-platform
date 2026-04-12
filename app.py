"""
Quant Platform — Main Streamlit Entry Point
Run with: streamlit run app.py
"""
import logging

import streamlit as st

from data.db import init_db
from broker.paper_trader import init_paper_tables
from scheduler.alerts import init_alerts_table
from journal.trading_journal import init_journal_table
from pages import shared, chart, backtest, screener, portfolio, alerts
from pages import journal_tab, efficient_frontier

# ── App bootstrap ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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
with tab1: chart.render()
with tab2: backtest.render()
with tab3: screener.render()
with tab4: portfolio.render()
with tab5: alerts.render()
with tab6: journal_tab.render()
with tab7: efficient_frontier.render()
