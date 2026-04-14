"""
pages/journal_tab.py — Trading Journal tab.
"""
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import streamlit as st

from journal.trading_journal import (
    avg_pnl_by_regime,
    get_journal,
    win_rate_by_signal_source,
)


def render() -> None:
    st.subheader("📓 Trading Journal")
    st.caption("Log and analyse every trade entry and exit with signal and regime metadata.")

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 2, 2])
    with f1:
        start = st.date_input(
            "From",
            value=date.today() - timedelta(days=90),
            max_value=date.today(),
            key="jnl_start",
        )
    with f2:
        end = st.date_input(
            "To",
            value=date.today(),
            min_value=start,
            max_value=date.today(),
            key="jnl_end",
        )
    with f3:
        ticker_filter = st.text_input(
            "Ticker (leave blank for all)",
            value="",
            key="jnl_ticker",
        ).upper().strip() or None

    st.divider()

    # ── Journal dataframe ─────────────────────────────────────────────────────
    st.markdown("#### Journal Entries")
    df = get_journal(
        start_date=str(start),
        end_date=str(end),
        ticker=ticker_filter,
    )

    if df.empty:
        st.info("No journal entries found for the selected filters.")
    else:
        display_cols = [
            "id", "ticker", "side", "qty", "entry_price", "entry_time",
            "signal_source", "regime",
            "exit_price", "exit_time", "pnl", "exit_reason",
        ]
        present = [c for c in display_cols if c in df.columns]
        disp = df[present].copy()

        def _colour_pnl(val):
            if pd.isna(val) or not isinstance(val, (int, float)):
                return ""
            return f"color: {'#26a69a' if val >= 0 else '#ef5350'}; font-weight: bold"

        st.dataframe(
            disp.style
            .map(_colour_pnl, subset=["pnl"] if "pnl" in disp.columns else [])
            .format(
                {
                    "entry_price": "${:.4f}",
                    "exit_price":  "${:.4f}",
                    "pnl":         lambda v: f"${v:+.2f}" if pd.notna(v) else "—",
                },
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"{len(df)} row(s) shown")

    st.divider()

    # ── Analytics charts ──────────────────────────────────────────────────────
    chart_l, chart_r = st.columns(2)

    with chart_l:
        st.markdown("#### Win Rate by Signal Source")
        wr_df = win_rate_by_signal_source()
        if wr_df.empty:
            st.info("No closed trades yet.")
        else:
            fig = px.bar(
                wr_df,
                x="signal_source",
                y="win_rate",
                text=wr_df["win_rate"].map(lambda v: f"{v:.0%}"),
                color="win_rate",
                color_continuous_scale=["#ef5350", "#ffb74d", "#26a69a"],
                range_color=[0, 1],
                labels={"signal_source": "Signal Source", "win_rate": "Win Rate"},
                hover_data={"total_trades": True, "wins": True, "avg_pnl": ":.2f"},
            )
            fig.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20),
                yaxis_tickformat=".0%",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

    with chart_r:
        st.markdown("#### Avg PnL by Regime")
        rg_df = avg_pnl_by_regime()
        if rg_df.empty:
            st.info("No closed trades yet.")
        else:
            rg_df["color"] = rg_df["avg_pnl"].apply(
                lambda v: "#26a69a" if v >= 0 else "#ef5350"
            )
            fig2 = px.bar(
                rg_df,
                x="regime",
                y="avg_pnl",
                text=rg_df["avg_pnl"].map(lambda v: f"${v:+.2f}"),
                color="avg_pnl",
                color_continuous_scale=["#ef5350", "#ffb74d", "#26a69a"],
                labels={"regime": "Regime", "avg_pnl": "Avg PnL ($)"},
                hover_data={"total_trades": True, "win_rate": ":.1%"},
            )
            fig2.update_layout(
                coloraxis_showscale=False,
                margin=dict(t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            fig2.update_traces(textposition="outside")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Execution Quality ─────────────────────────────────────────────────────
    st.divider()
    with st.expander("Execution Quality", expanded=False):
        st.caption(
            "Slippage analysis across execution algorithms and symbols. "
            "Slippage = (avg fill price − decision price) / decision price × 10,000 bps."
        )
        try:
            from data.db import get_connection
            conn = get_connection()
            rows = conn.execute(
                "SELECT * FROM execution_analytics ORDER BY executed_at DESC LIMIT 500"
            ).fetchall()
            conn.close()
            if not rows:
                st.info(
                    "No execution data yet. Use execute_algo() from broker/paper_trader.py "
                    "to log fills."
                )
            else:
                exec_df = pd.DataFrame([dict(r) for r in rows])
                exec_df["executed_at"] = pd.to_datetime(
                    exec_df["executed_at"], unit="s"
                ).dt.strftime("%Y-%m-%d %H:%M")

                eq_l, eq_r = st.columns(2)

                with eq_l:
                    st.markdown("**Avg Slippage by Algorithm (bps)**")
                    algo_grp = (
                        exec_df.groupby("algo")["slippage_bps"]
                        .mean()
                        .reset_index()
                        .rename(columns={"algo": "Algorithm", "slippage_bps": "Avg Slippage (bps)"})
                    )
                    fig_algo = px.bar(
                        algo_grp,
                        x="Algorithm",
                        y="Avg Slippage (bps)",
                        color="Avg Slippage (bps)",
                        color_continuous_scale=["#26a69a", "#ffb74d", "#ef5350"],
                        text=algo_grp["Avg Slippage (bps)"].map(lambda v: f"{v:.1f}"),
                    )
                    fig_algo.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(t=20, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_algo, use_container_width=True)

                with eq_r:
                    st.markdown("**Avg Slippage by Symbol (bps)**")
                    sym_grp = (
                        exec_df.groupby("symbol")["slippage_bps"]
                        .mean()
                        .reset_index()
                        .rename(columns={"symbol": "Symbol", "slippage_bps": "Avg Slippage (bps)"})
                        .sort_values("Avg Slippage (bps)", ascending=False)
                        .head(10)
                    )
                    fig_sym = px.bar(
                        sym_grp,
                        x="Symbol",
                        y="Avg Slippage (bps)",
                        color="Avg Slippage (bps)",
                        color_continuous_scale=["#26a69a", "#ffb74d", "#ef5350"],
                        text=sym_grp["Avg Slippage (bps)"].map(lambda v: f"{v:.1f}"),
                    )
                    fig_sym.update_layout(
                        coloraxis_showscale=False,
                        margin=dict(t=20, b=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_sym, use_container_width=True)

                st.markdown("**Recent Fills**")
                st.dataframe(
                    exec_df[[
                        "executed_at", "symbol", "side", "algo",
                        "total_qty", "decision_price", "avg_fill_price", "slippage_bps",
                    ]].head(20),
                    use_container_width=True,
                )
        except Exception as exc:
            st.warning(f"Could not load execution analytics: {exc}")
