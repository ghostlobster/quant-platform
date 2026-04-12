"""
Shared sidebar renderer and utilities used across all page modules.

Call render_sidebar() once in app.py.  Widget keys are deliberately chosen
so that page render() functions can read values directly from st.session_state:

  active_ticker  — current ticker symbol
  _period        — period code, e.g. "6mo"  (derived from selectbox)
  _period_label  — human label, e.g. "6 Months"
  _chart_type    — "Candlestick" or "Line"
  _show_ema      — bool
  _show_bb       — bool
  _show_rsi      — bool
  _show_macd     — bool
  _show_signals  — bool
"""
import re
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh

from data.watchlist import add_ticker, get_watchlist, remove_ticker

# ── Ticker validation regex (also imported by screener.py) ───────────────────
_TICKER_RE = re.compile(r"^[A-Z]{1,5}(-[A-Z]+)?$")

_PERIOD_OPTIONS: dict[str, str] = {
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
}


def set_ticker(sym: str) -> None:
    """Update both session state keys so sidebar input stays in sync."""
    sym = sym.upper().strip()
    st.session_state["active_ticker"]   = sym
    st.session_state["_sidebar_ticker"] = sym


def _on_sidebar_ticker_change() -> None:
    val = st.session_state.get("_sidebar_ticker", "AAPL").upper().strip()
    if val and not _TICKER_RE.match(val):
        st.session_state["_ticker_error"] = f"Invalid ticker: '{val}'"
        return
    st.session_state.pop("_ticker_error", None)
    st.session_state["active_ticker"] = val or "AAPL"


def render_sidebar() -> dict:
    """Render the full sidebar and return a dict of the selected values."""
    # ── Session state: active ticker (shared across tabs) ─────────────────────
    if "active_ticker" not in st.session_state:
        st.session_state["active_ticker"] = "AAPL"

    if "_sidebar_ticker" not in st.session_state:
        st.session_state["_sidebar_ticker"] = st.session_state["active_ticker"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.title("📈 Quant Platform")
    st.sidebar.markdown("*Personal Trading Dashboard*")
    st.sidebar.divider()

    # ── Section: Chart Controls ───────────────────────────────────────────────
    st.sidebar.markdown("### 📊 Chart Controls")

    st.sidebar.text_input(
        "Ticker",
        key="_sidebar_ticker",
        on_change=_on_sidebar_ticker_change,
    )
    ticker = st.session_state["active_ticker"]

    # Use explicit key so chart.py can read _period_label from session_state
    st.sidebar.selectbox(
        "Period", list(_PERIOD_OPTIONS.keys()), index=2, key="_period_label"
    )
    selected_period_label = st.session_state["_period_label"]
    # Derive the period code and store it too
    st.session_state["_period"] = _PERIOD_OPTIONS[selected_period_label]

    st.sidebar.radio("Chart Type", ["Candlestick", "Line"], key="_chart_type")

    st.sidebar.divider()

    # ── Section: Overlays ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 🔍 Overlays")
    st.sidebar.checkbox("EMA 20 / 50",     value=True, key="_show_ema")
    st.sidebar.checkbox("Bollinger Bands", value=True, key="_show_bb")
    st.sidebar.checkbox("RSI panel",       value=True, key="_show_rsi")
    st.sidebar.checkbox("MACD panel",      value=True, key="_show_macd")
    st.sidebar.checkbox("Signal summary",  value=True, key="_show_signals")

    st.sidebar.divider()

    # ── Section: Watchlist ────────────────────────────────────────────────────
    st.sidebar.markdown("### 👁️ Watchlist")

    new_ticker_col, add_btn_col = st.sidebar.columns([3, 1])
    _raw_ticker = new_ticker_col.text_input(
        "Add ticker", placeholder="e.g. NVDA", label_visibility="collapsed"
    )
    new_ticker_input = (_raw_ticker if isinstance(_raw_ticker, str) else "").upper().strip()
    if add_btn_col.button("＋", help="Add to watchlist"):
        if new_ticker_input:
            if not _TICKER_RE.match(new_ticker_input):
                st.sidebar.error(f"Invalid ticker: '{new_ticker_input}'")
            else:
                added = add_ticker(new_ticker_input)
                st.sidebar.success(
                    f"{new_ticker_input} added" if added else f"{new_ticker_input} already tracked"
                )
                st.rerun()

    watchlist = get_watchlist()
    for wt in watchlist:
        col_t, col_r = st.sidebar.columns([4, 1])
        col_t.markdown(f"**{wt}**")
        if col_r.button("✕", key=f"rm_{wt}", help=f"Remove {wt}"):
            remove_ticker(wt)
            st.rerun()

    st.sidebar.divider()

    # ── Section: Auto-refresh ─────────────────────────────────────────────────
    st.sidebar.markdown("### ⚡ Auto-refresh")
    autorefresh_on = st.sidebar.toggle("Enable auto-refresh", value=False, key="autorefresh_toggle")
    refresh_interval_label = st.sidebar.selectbox(
        "Interval",
        ["1 min", "5 min", "15 min", "30 min"],
        index=1,
        disabled=not autorefresh_on,
        key="refresh_interval_select",
    )
    _interval_map = {"1 min": 60_000, "5 min": 300_000, "15 min": 900_000, "30 min": 1_800_000}
    if autorefresh_on:
        st_autorefresh(interval=_interval_map[refresh_interval_label], key="auto_refresh_counter")

    st.sidebar.divider()
    st.sidebar.caption("© Quant Platform · All 10 steps complete ✓")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Auto-refresh")
    refresh_interval = st.sidebar.selectbox("Interval", [0, 60, 300, 900, 1800], format_func=lambda x: "Off" if x == 0 else f"{x//60} min")
    if refresh_interval > 0:
        st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

    # ── Header ────────────────────────────────────────────────────────────────
    _now = datetime.now()
    _header_col, _ts_col = st.columns([3, 1])
    _header_col.title("Quant Platform — Live")
    _ts_col.markdown(
        f"<div style='text-align:right; padding-top:18px; color:#888; font-size:0.85rem;'>"
        f"Last updated<br><b>{_now.strftime('%Y-%m-%d %H:%M:%S')}</b></div>",
        unsafe_allow_html=True,
    )
    st.caption("Data via Yahoo Finance (SQLite-cached)")

    return {
        "ticker":                ticker,
        "period":                st.session_state["_period"],
        "selected_period_label": selected_period_label,
        "chart_type":            st.session_state["_chart_type"],
        "show_ema":              st.session_state["_show_ema"],
        "show_bb":               st.session_state["_show_bb"],
        "show_rsi":              st.session_state["_show_rsi"],
        "show_macd":             st.session_state["_show_macd"],
        "show_signals":          st.session_state["_show_signals"],
        "watchlist":             watchlist,
    }
