"""
Quant Platform — Main Streamlit Entry Point
Run with: streamlit run app.py
"""
import os as _bootstrap_os

import streamlit as st
import structlog as _bootstrap_structlog

import config
from auth.session import is_authenticated
from broker.paper_trader import init_paper_tables
from data.db import init_db
from journal.trading_journal import init_journal_table
from pages import (
    alerts,
    backtest,
    chart,
    efficient_frontier,
    greeks,
    journal_tab,
    ml_signals,
    model_health,
    portfolio,
    screener,
    shared,
)
from pages import (
    login as _login_page,
)
from scheduler.alerts import init_alerts_table, start_knowledge_health_scheduler

# ── App bootstrap ─────────────────────────────────────────────────────────────
config.configure_logging()
init_db()
init_paper_tables()
init_alerts_table()
init_journal_table()

# Opt-in hourly knowledge health check (#116). Requires the operator to set
# ENABLE_KNOWLEDGE_HEALTH_JOB=1 so dev sessions don't get surprise background
# jobs. KNOWLEDGE_HEALTH_CRON / KNOWLEDGE_HEALTH_ENABLED fine-tune the schedule.
if _bootstrap_os.environ.get("ENABLE_KNOWLEDGE_HEALTH_JOB", "").strip().lower() in (
    "1", "true", "yes", "on",
):
    try:
        start_knowledge_health_scheduler()
    except Exception as _exc:
        _bootstrap_structlog.get_logger(__name__).warning(
            "knowledge_health_job: bootstrap failed", error=str(_exc),
        )

st.set_page_config(
    page_title="Quant Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not is_authenticated():
    _login_page.render()
    st.stop()

shared.render_sidebar()

domain_choice = st.session_state.get("_nav_domain", "🌐 All Features")

if domain_choice == "📊 Market Research":
    tab1, tab2 = st.tabs(["📈 Chart", "🔍 Screener"])
    with tab1:
        chart.render()
    with tab2:
        screener.render()

elif domain_choice == "🤖 Quantitative ML":
    tab1, tab2, tab3 = st.tabs(["🤖 ML Signals", "🧮 Greeks", "📐 Efficient Frontier"])
    with tab1:
        ml_signals.render()
    with tab2:
        greeks.render()
    with tab3:
        efficient_frontier.render()

elif domain_choice == "🔬 Backtest & Model":
    tab1, tab2 = st.tabs(["🔬 Backtest", "🩺 Model Health"])
    with tab1:
        backtest.render()
    with tab2:
        model_health.render()

elif domain_choice == "💼 Execution & Portfolio":
    tab1, tab2, tab3 = st.tabs(["💼 Portfolio", "📓 Journal", "🔔 Alerts"])
    with tab1:
        portfolio.render()
    with tab2:
        journal_tab.render()
    with tab3:
        alerts.render()

else:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📈 Chart", "🔬 Backtest", "🔍 Screener", "💼 Portfolio",
        "🔔 Alerts", "📓 Journal", "📐 Efficient Frontier", "🧮 Greeks",
        "🤖 ML Signals", "🩺 Model Health",
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
    with tab8:
        greeks.render()
    with tab9:
        ml_signals.render()
    with tab10:
        model_health.render()

