"""
pages/screener.py — Stock screener tab.
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

from pages.shared import set_ticker
from screener.screener import UNIVERSE, run_screen

_DEFAULT_REFRESH = 30


@st.cache_resource
def get_feed():
    from data.realtime import create_feed
    return create_feed(mode='auto')


def _poll_seconds() -> int:
    try:
        return int(os.getenv("REALTIME_POLL_SECONDS", str(_DEFAULT_REFRESH)))
    except (ValueError, TypeError):
        return _DEFAULT_REFRESH


def render() -> None:
    st.subheader("Stock Screener")
    st.caption(f"Universe: {len(UNIVERSE)} tickers across Technology, Finance, Healthcare, Energy, Consumer, Industrials")

    # ── Filter controls ───────────────────────────────────────────────────────
    with st.expander("Filters", expanded=True):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            st.markdown("**RSI (14)**")
            rsi_filter = st.radio(
                "RSI filter", ["Any", "Oversold (< 30)", "Overbought (> 70)", "Custom range"],
                key="sc_rsi_filter", label_visibility="collapsed",
            )
            sc_rsi_min, sc_rsi_max = None, None
            if rsi_filter == "Oversold (< 30)":
                sc_rsi_max = 30.0
            elif rsi_filter == "Overbought (> 70)":
                sc_rsi_min = 70.0
            elif rsi_filter == "Custom range":
                rsi_range = st.slider("RSI range", 0, 100, (20, 80), key="sc_rsi_range")
                sc_rsi_min, sc_rsi_max = float(rsi_range[0]), float(rsi_range[1])

        with fc2:
            st.markdown("**Price & Volume**")
            sc_price_min = st.number_input("Min price ($)", min_value=0.0, value=0.0,
                                           step=10.0, key="sc_price_min") or None
            sc_price_max_raw = st.number_input("Max price ($)", min_value=0.0, value=0.0,
                                               step=100.0, key="sc_price_max",
                                               help="0 = no limit")
            sc_price_max = sc_price_max_raw if sc_price_max_raw > 0 else None
            sc_vol_spike = st.number_input("Min volume ratio (×avg)", min_value=0.0,
                                           value=0.0, step=0.5, key="sc_vol_spike",
                                           help="e.g. 1.5 = at least 1.5× the 20-day avg. 0 = no filter")
            sc_vol_spike_min = sc_vol_spike if sc_vol_spike > 0 else None

        with fc3:
            st.markdown("**Trend & Change**")
            trend_option = st.selectbox(
                "SMA50 trend", ["Any", "Above SMA50 (uptrend)", "Below SMA50 (downtrend)"],
                key="sc_trend",
            )
            sc_trend = (None if trend_option == "Any"
                        else "above_sma50" if "Above" in trend_option
                        else "below_sma50")

            sc_change_days = st.selectbox("Price change look-back",
                                          [1, 3, 5, 10, 20], index=2, key="sc_change_days")
            change_pct_range = st.slider(
                f"{sc_change_days}-day change (%)", -30, 30, (-30, 30), key="sc_change_range"
            )
            sc_change_min = float(change_pct_range[0]) if change_pct_range[0] > -30 else None
            sc_change_max = float(change_pct_range[1]) if change_pct_range[1] < 30 else None

    # ── Sector filter ─────────────────────────────────────────────────────────
    all_sectors = sorted({item["sector"] for item in UNIVERSE})
    selected_sectors = st.multiselect(
        "Sectors", all_sectors, default=all_sectors, key="sc_sectors"
    )
    tickers_in_scope = [
        item["ticker"] for item in UNIVERSE
        if item["sector"] in selected_sectors
    ]

    # ── Live Quotes ───────────────────────────────────────────────────────────
    poll_secs = _poll_seconds()
    feed = get_feed()
    if tickers_in_scope:
        feed.subscribe(tickers_in_scope)

    @st.fragment(run_every=poll_secs)
    def _live_quotes() -> None:
        st.markdown("#### Live Quotes")
        st.caption(
            f"Auto-refreshes every **{poll_secs}s** "
            f"(`REALTIME_POLL_SECONDS={poll_secs}`)"
        )

        if not tickers_in_scope:
            st.info("Select at least one sector to see live quotes.")
            return

        rows = []
        for ticker in tickers_in_scope:
            q = feed.get_quote(ticker)
            if q is None:
                rows.append({
                    "Ticker": ticker,
                    "Bid": None,
                    "Ask": None,
                    "Last": None,
                    "Volume": None,
                    "Timestamp": "Fetching…",
                })
            else:
                rows.append({
                    "Ticker": ticker,
                    "Bid": q.bid,
                    "Ask": q.ask,
                    "Last": q.last,
                    "Volume": q.volume,
                    "Timestamp": q.timestamp,
                })

        quotes_df = pd.DataFrame(rows)
        n_ready = int(quotes_df["Last"].notna().sum())
        n_total = len(quotes_df)

        if n_ready == 0:
            st.info("Feed initialising — quotes arriving shortly…")
        else:
            st.caption(f"{n_ready} / {n_total} tickers have live data")

        fmt = {col: "${:.2f}" for col in ("Bid", "Ask", "Last") if col in quotes_df.columns}
        if "Volume" in quotes_df.columns:
            fmt["Volume"] = "{:,.0f}"

        def _style_last(val) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "color: #555"
            return ""

        styled = (
            quotes_df.style
            .map(_style_last, subset=["Last"])
            .format(fmt, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    _live_quotes()

    st.divider()

    # ── Run button ────────────────────────────────────────────────────────────
    run_screen_btn = st.button("🔍 Run Screen", type="primary", key="sc_run")

    st.divider()

    # ── Results ───────────────────────────────────────────────────────────────
    if run_screen_btn:
        if not tickers_in_scope:
            st.warning("No tickers selected — choose at least one sector.")
        else:
            progress_bar = st.progress(0, text="Fetching market data…")
            with st.spinner(f"Scanning {len(tickers_in_scope)} tickers…"):
                screen_df = run_screen(
                    tickers=tickers_in_scope,
                    change_days=int(sc_change_days),
                    rsi_min=sc_rsi_min,
                    rsi_max=sc_rsi_max,
                    change_min=sc_change_min,
                    change_max=sc_change_max,
                    vol_spike_min=sc_vol_spike_min,
                    price_min=float(sc_price_min) if sc_price_min else None,
                    price_max=sc_price_max,
                    trend=sc_trend,
                )
            progress_bar.empty()

            if screen_df.empty:
                st.warning("No tickers matched the current filters. Try relaxing the criteria.")
            else:
                change_col = f"Change {sc_change_days}d (%)"
                st.success(f"**{len(screen_df)} tickers** matched your filters.")

                _SIGNAL_STYLE = {
                    "Oversold":     "background-color: #1a3d2b; color: #26a69a; font-weight: bold",
                    "Overbought":   "background-color: #3d1a1a; color: #ef5350; font-weight: bold",
                    "Trending Up":  "background-color: #1a2d3d; color: #64b5f6; font-weight: bold",
                    "Trending Down":"background-color: #2d2010; color: #ffb300; font-weight: bold",
                    "Neutral":      "color: #aaa",
                    "N/A":          "color: #555",
                }

                def _style_signal(val: str) -> str:
                    return _SIGNAL_STYLE.get(val, "")

                def _style_rsi(val) -> str:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    if val < 30:
                        return "color: #26a69a; font-weight: bold"
                    if val > 70:
                        return "color: #ef5350; font-weight: bold"
                    return ""

                def _style_change(val) -> str:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    return f"color: {'#26a69a' if val >= 0 else '#ef5350'}"

                def _style_vol(val) -> str:
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return ""
                    if val >= 2.0:
                        return "color: #ffb300; font-weight: bold"
                    return ""

                display_cols = ["Ticker", "Sector", "Name", "Last Price",
                                "RSI", change_col, "Vol Ratio", "Signal"]
                display_cols = [c for c in display_cols if c in screen_df.columns]
                disp_df = screen_df[display_cols].copy()

                fmt: dict = {
                    "Last Price": "${:.2f}",
                    "RSI": "{:.1f}",
                    "Vol Ratio": "{:.2f}×",
                }
                if change_col in disp_df.columns:
                    fmt[change_col] = "{:+.2f}%"

                styled_screen = (
                    disp_df.style
                    .map(_style_signal,  subset=["Signal"])
                    .map(_style_rsi,     subset=["RSI"])
                    .map(_style_change,  subset=[change_col] if change_col in disp_df.columns else [])
                    .map(_style_vol,     subset=["Vol Ratio"])
                    .format(fmt, na_rep="—")
                )

                st.dataframe(styled_screen, use_container_width=True, hide_index=True)

                sig_counts = screen_df["Signal"].value_counts()
                sig_order  = ["Oversold", "Trending Up", "Neutral", "Trending Down", "Overbought"]
                sig_cols   = st.columns(len(sig_order))
                for col, sig in zip(sig_cols, sig_order):
                    count = sig_counts.get(sig, 0)
                    col.metric(sig, count)

                st.divider()
                st.markdown("**Click any ticker to load its chart** (switches to Chart tab):")
                btn_tickers = screen_df["Ticker"].tolist()
                chunk_size = 10
                for chunk_start in range(0, len(btn_tickers), chunk_size):
                    chunk = btn_tickers[chunk_start: chunk_start + chunk_size]
                    btn_cols = st.columns(len(chunk))
                    for col, sym in zip(btn_cols, chunk):
                        sig = screen_df.loc[screen_df["Ticker"] == sym, "Signal"].values[0]
                        btn_colour = {
                            "Oversold":     "🟢",
                            "Overbought":   "🔴",
                            "Trending Up":  "🔵",
                            "Trending Down":"🟡",
                        }.get(sig, "⚪")
                        if col.button(f"{btn_colour} {sym}", key=f"sc_jump_{sym}"):
                            set_ticker(sym)
                            st.rerun()
    else:
        st.info(
            "Configure filters above and click **🔍 Run Screen** to scan the universe.\n\n"
            "Results show: Ticker · Sector · Last Price · RSI · Price Change · "
            "Volume Ratio · Signal"
        )
        meta_df = pd.DataFrame(UNIVERSE)[["ticker", "sector", "name"]]
        meta_df.columns = ["Ticker", "Sector", "Name"]
        sector_counts = meta_df["Sector"].value_counts().reset_index()
        sector_counts.columns = ["Sector", "Tickers"]
        st.markdown("#### Universe breakdown")
        st.dataframe(sector_counts, use_container_width=False, hide_index=True)
