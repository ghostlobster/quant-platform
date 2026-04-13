"""
pages/alerts.py — Price & RSI alerts tab.
"""
import pandas as pd
import streamlit as st
import structlog

from data.fetcher import fetch_latest_price, fetch_ohlcv
from data.indicators import compute_rsi
from scheduler.alerts import (
    ALERT_TYPES,
    add_alert,
    check_alerts,
    delete_alert,
    get_alerts,
    toggle_alert,
)

logger = structlog.get_logger(__name__)


def render() -> None:
    ticker = st.session_state.get("active_ticker", "AAPL")

    st.subheader("🔔 Price & RSI Alerts")
    st.caption("Alerts are evaluated on-demand. Desktop notifications fire via plyer when an alert triggers.")

    # ── Alert creation form ───────────────────────────────────────────────────
    st.markdown("#### Add Alert")
    with st.form("alert_form", clear_on_submit=True):
        af1, af2, af3, af4 = st.columns([2, 2.5, 1.5, 1])
        with af1:
            al_ticker = st.text_input("Ticker", value=ticker, key="al_ticker").upper().strip()
        with af2:
            al_type_label = st.selectbox(
                "Alert Type",
                list(ALERT_TYPES.keys()),
                format_func=lambda k: ALERT_TYPES[k],
                key="al_type",
            )
        with af3:
            al_threshold = st.number_input(
                "Threshold", min_value=0.001, value=100.0,
                step=1.0, format="%.2f", key="al_threshold",
            )
        with af4:
            st.markdown("<br>", unsafe_allow_html=True)
            add_al_btn = st.form_submit_button("➕ Add Alert", type="primary", use_container_width=True)

    if add_al_btn:
        if not al_ticker:
            st.error("Enter a ticker symbol.")
        else:
            try:
                new_id = add_alert(al_ticker, al_type_label, al_threshold)
                st.success(
                    f"Alert #{new_id} added: **{al_ticker}** — "
                    f"{ALERT_TYPES[al_type_label]} {al_threshold:.2f}"
                )
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    st.divider()

    # ── Active alerts table ───────────────────────────────────────────────────
    st.markdown("#### Active Alerts")
    alerts_df = get_alerts()

    if alerts_df.empty:
        st.info("No alerts configured. Add one above.")
    else:
        _TYPE_ICON = {
            "price_above": "📈",
            "price_below": "📉",
            "rsi_above":   "🔴",
            "rsi_below":   "🟢",
        }
        _STATUS_STYLE = {
            True:  ("✅ Active", "#1a3d2b"),
            False: ("⏸ Paused", "#2a2a2a"),
        }

        for _, row in alerts_df.iterrows():
            al_id      = int(row["ID"])
            enabled    = bool(row["Enabled"])
            icon       = _TYPE_ICON.get(row["Type"], "🔔")
            status_lbl, status_bg = _STATUS_STYLE[enabled]
            type_desc  = ALERT_TYPES.get(row["Type"], row["Type"])

            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 1.2, 2.8, 1.4, 1.5, 0.85, 0.75])
            c1.markdown(f"**#{al_id}**")
            c2.markdown(f"**{row['Ticker']}**")
            c3.markdown(f"{icon} {type_desc} **{row['Threshold']:.2f}**")
            c4.markdown(
                f"<span style='background:{status_bg}; padding:2px 8px; "
                f"border-radius:4px; font-size:0.85em'>{status_lbl}</span>",
                unsafe_allow_html=True,
            )
            c5.markdown(
                f"<span style='font-size:0.82em; color:#aaa'>Fired: {row['Last Triggered']}</span>",
                unsafe_allow_html=True,
            )
            tog_lbl = "⏸" if enabled else "▶"
            if c6.button(tog_lbl, key=f"al_tog_{al_id}", help="Toggle enable/pause",
                         use_container_width=True):
                toggle_alert(al_id, not enabled)
                st.rerun()
            if c7.button("🗑", key=f"al_del_{al_id}", help="Delete alert",
                         use_container_width=True):
                delete_alert(al_id)
                st.success(f"Alert #{al_id} deleted.")
                st.rerun()

    st.divider()

    # ── Check Alerts Now ──────────────────────────────────────────────────────
    st.markdown("#### Check Alerts Now")

    al_check_col1, al_check_col2 = st.columns([1, 4])
    run_check = al_check_col1.button("🔍 Check Alerts Now", type="primary", key="al_check")

    if run_check:
        all_alerts = get_alerts()
        if all_alerts.empty:
            st.warning("No alerts configured — add some first.")
        else:
            active = all_alerts[all_alerts["Enabled"] == True]  # noqa: E712
            if active.empty:
                st.warning("All alerts are paused. Enable at least one.")
            else:
                unique_tickers = active["Ticker"].unique().tolist()
                st.caption(f"Fetching live data for {len(unique_tickers)} ticker(s): {', '.join(unique_tickers)}")

                current_data: dict = {}
                fetch_errors: list[str] = []

                progress = st.progress(0, text="Fetching data…")
                for i, t in enumerate(unique_tickers):
                    try:
                        price_info = fetch_latest_price(t)
                        if price_info["price"] is None:
                            fetch_errors.append(t)
                            continue

                        try:
                            ohlcv   = fetch_ohlcv(t, "3mo")
                            rsi_val = compute_rsi(ohlcv["Close"])
                        except Exception:
                            rsi_val = None

                        current_data[t] = {
                            "price": price_info["price"],
                            "rsi":   rsi_val,
                        }
                    except Exception as exc:
                        fetch_errors.append(t)
                        logger.warning("Alert check fetch failed for %s: %s", t, exc)

                    progress.progress(
                        int((i + 1) / len(unique_tickers) * 100),
                        text=f"Fetched {t} ({i+1}/{len(unique_tickers)})"
                    )
                progress.empty()

                if fetch_errors:
                    st.warning(f"Could not fetch data for: {', '.join(fetch_errors)}")

                if current_data:
                    snap_rows = [
                        {"Ticker": t,
                         "Price":  f"${v['price']:.2f}",
                         "RSI(14)": f"{v['rsi']:.1f}" if v["rsi"] is not None else "—"}
                        for t, v in current_data.items()
                    ]
                    with st.expander("Live data snapshot", expanded=False):
                        st.dataframe(pd.DataFrame(snap_rows), hide_index=True,
                                     use_container_width=False)

                triggered = check_alerts(current_data)

                st.divider()
                if not triggered:
                    st.success("✅ No alerts triggered — all conditions within thresholds.")
                else:
                    st.error(f"🔔 **{len(triggered)} alert(s) triggered!**")
                    for t in triggered:
                        st.warning(
                            f"**{t['ticker']}** · {ALERT_TYPES.get(t['alert_type'], t['alert_type'])} "
                            f"{t['threshold']:.2f} — "
                            f"Price: ${t['price']:.2f}"
                            + (f" · RSI: {t['rsi']:.1f}" if t["rsi"] is not None else "")
                            + f" · Fired at {t['fired_at']}"
                        )

                    trig_df = pd.DataFrame([{
                        "Fired At":   t["fired_at"],
                        "Ticker":     t["ticker"],
                        "Alert Type": ALERT_TYPES.get(t["alert_type"], t["alert_type"]),
                        "Threshold":  t["threshold"],
                        "Price":      t["price"],
                        "RSI":        round(t["rsi"], 1) if t["rsi"] is not None else None,
                    } for t in triggered])
                    st.dataframe(
                        trig_df.style.format({
                            "Threshold": "{:.2f}",
                            "Price":     "${:.2f}",
                            "RSI":       lambda v: f"{v:.1f}" if v is not None else "—",
                        }, na_rep="—"),
                        use_container_width=True, hide_index=True,
                    )
                    st.rerun()
    else:
        st.caption("Press the button above to fetch live prices/RSI and evaluate all enabled alerts.")
