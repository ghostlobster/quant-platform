"""
pages/chart.py — Chart & price analysis tab.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.fetcher import fetch_ohlcv, fetch_latest_price
from data.watchlist import add_ticker, get_watchlist, is_in_watchlist, remove_ticker
from strategies.indicators import add_all, generate_signals
from analysis.regime import get_live_regime

_REGIME_ICON = {
    "trending_bull":  "🟢",
    "trending_bear":  "🔴",
    "mean_reverting": "🟡",
    "high_vol":       "⚠️",
}

_REGIME_COLOUR = {
    "trending_bull":  "#1a3d2b",
    "trending_bear":  "#3d1a1a",
    "mean_reverting": "#3d3010",
    "high_vol":       "#3d2010",
}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_regime() -> dict | None:
    try:
        return get_live_regime()
    except Exception:
        return None


def _render_regime_badge() -> None:
    """Display a colour-coded market regime banner at the top of the Chart tab."""
    with st.spinner("Detecting market regime…"):
        info = _fetch_regime()

    if info is None:
        st.warning("Market regime unavailable — could not fetch SPY/VIX data.")
        return

    regime = info["regime"]
    icon   = _REGIME_ICON.get(regime, "❓")
    colour = _REGIME_COLOUR.get(regime, "#222")
    label  = regime.replace("_", " ").title()

    above_below = "above" if info["spy_price"] > info["spy_sma200"] else "below"
    sma_diff_pct = (info["spy_price"] - info["spy_sma200"]) / info["spy_sma200"] * 100

    strategies_html = " &nbsp;·&nbsp; ".join(
        f"<span style='font-size:0.8em'>{s}</span>"
        for s in info["recommended_strategies"]
    )

    st.markdown(
        f"""
        <div style="background:{colour}; border-radius:10px; padding:14px 18px;
                    margin-bottom:12px; border-left:4px solid {'#26a69a' if regime == 'trending_bull' else '#ef5350' if regime in ('trending_bear', 'high_vol') else '#f9a825'}">
          <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap">
            <span style="font-size:1.6em">{icon}</span>
            <span style="font-size:1.15em; font-weight:bold; color:#fff">Market Regime: {label}</span>
            <span style="font-size:0.85em; color:#ccc; margin-left:auto">
              VIX <b>{info['vix']:.1f}</b> &nbsp;|&nbsp;
              SPY <b>${info['spy_price']:.2f}</b> &nbsp;{above_below} 200d SMA
              <b>${info['spy_sma200']:.2f}</b> ({sma_diff_pct:+.1f}%)
            </span>
          </div>
          <div style="font-size:0.82em; color:#bbb; margin-top:6px">{info['description']}</div>
          <div style="margin-top:8px; color:#aaa">Strategies: {strategies_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    ticker               = st.session_state.get("active_ticker", "AAPL")
    period               = st.session_state.get("_period", "6mo")
    selected_period_label = st.session_state.get("_period_label", "6 Months")
    chart_type           = st.session_state.get("_chart_type", "Candlestick")
    show_ema             = st.session_state.get("_show_ema", True)
    show_bb              = st.session_state.get("_show_bb", True)
    show_rsi             = st.session_state.get("_show_rsi", True)
    show_macd            = st.session_state.get("_show_macd", True)
    show_signals         = st.session_state.get("_show_signals", True)
    watchlist            = get_watchlist()

    # ── Market regime badge ───────────────────────────────────────────────────
    _render_regime_badge()

    # ── Watchlist snapshot ────────────────────────────────────────────────────
    st.subheader("Watchlist")

    @st.cache_data(ttl=300, show_spinner=False)
    def _watchlist_snapshot(tickers_key: str) -> pd.DataFrame:
        rows = []
        for t in (tickers_key.split(",") if tickers_key else []):
            d = fetch_latest_price(t)
            rows.append({
                "Ticker":     d["ticker"],
                "Last Price": d["price"],
                "Change ($)": d["change"],
                "Change (%)": d["pct_change"],
                "Status":     ("⚠️ " + d["error"]) if d["error"] else "✓",
            })
        return pd.DataFrame(rows)

    if watchlist:
        with st.spinner("Loading watchlist…"):
            snap_df = _watchlist_snapshot(",".join(watchlist))

        def _style_pct(val):
            if val is None:
                return ""
            return f"color: {'#26a69a' if val >= 0 else '#ef5350'}; font-weight: bold"

        st.dataframe(
            snap_df.style
            .map(_style_pct, subset=["Change (%)"])
            .format({
                "Last Price": lambda v: f"${v:.2f}" if v else "—",
                "Change ($)": lambda v: f"{v:+.2f}" if v else "—",
                "Change (%)": lambda v: f"{v:+.2f}%" if v else "—",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Watchlist is empty — add tickers in the sidebar.")

    st.divider()

    # ── Fetch + indicators ────────────────────────────────────────────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _load(sym: str, per: str) -> pd.DataFrame:
        return add_all(fetch_ohlcv(sym, per))

    st.subheader(f"{ticker} — {selected_period_label}")

    try:
        with st.spinner(f"Fetching {ticker}…"):
            df = _load(ticker, period)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    # ── Key metrics ───────────────────────────────────────────────────────────
    latest = float(df["Close"].iloc[-1])
    prev   = float(df["Close"].iloc[-2]) if len(df) > 1 else latest
    change = latest - prev
    pct    = (change / prev) * 100 if prev else 0.0

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Last Price",  f"${latest:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
    mc2.metric("Period High", f"${df['High'].max():.2f}")
    mc3.metric("Period Low",  f"${df['Low'].min():.2f}")
    mc4.metric("Avg Volume",  f"{df['Volume'].mean():,.0f}")
    in_wl = is_in_watchlist(ticker)
    if mc5.button(f"{'★ Unwatch' if in_wl else '☆ Watch'} {ticker}"):
        (remove_ticker if in_wl else add_ticker)(ticker)
        st.rerun()

    st.divider()

    # ── Signal summary cards ──────────────────────────────────────────────────
    if show_signals:
        sigs = generate_signals(df)
        if sigs:
            st.subheader("Signal Summary")
            sig_cols = st.columns(len(sigs))
            for col, s in zip(sig_cols, sigs):
                bg = "#1a3d2b" if s["bullish"] is True else (
                     "#3d1a1a" if s["bullish"] is False else "#2a2a2a")
                col.markdown(
                    f"""
                    <div style="background:{bg}; border-radius:8px; padding:12px 14px; height:100%">
                      <div style="font-size:1.4em">{s['icon']}</div>
                      <div style="font-size:0.75em; color:#aaa; margin-top:4px">{s['indicator']}</div>
                      <div style="font-size:1em; font-weight:bold; margin-top:2px">{s['signal']}</div>
                      <div style="font-size:0.78em; color:#ccc; margin-top:4px">{s['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("")

    # ── Price chart ───────────────────────────────────────────────────────────
    n_rows      = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
    row_heights = [0.55] + ([0.22] if show_rsi else []) + ([0.23] if show_macd else [])
    subplot_titles = ([f"{ticker} Price"]
                      + (["RSI (14)"] if show_rsi else [])
                      + (["MACD (12/26/9)"] if show_macd else []))

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name=ticker,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"],
            mode="lines", name=ticker,
            line=dict(color="#2196f3", width=2),
        ), row=1, col=1)

    if show_ema:
        for w, colour in [(20, "#ffb300"), (50, "#ab47bc")]:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[f"EMA_{w}"],
                mode="lines", name=f"EMA {w}",
                line=dict(color=colour, width=1.5),
            ), row=1, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            mode="lines", name="BB Upper",
            line=dict(color="rgba(100,181,246,0.6)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=pd.concat([df.index.to_series(), df.index.to_series()[::-1]]),
            y=pd.concat([df["BB_upper"], df["BB_lower"][::-1]]),
            fill="toself", fillcolor="rgba(100,181,246,0.06)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip", name="BB Band",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            mode="lines", name="BB Lower",
            line=dict(color="rgba(100,181,246,0.6)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_mid"],
            mode="lines", name="BB Mid",
            line=dict(color="rgba(100,181,246,0.35)", width=1),
        ), row=1, col=1)

    if show_rsi:
        rsi_row = 2
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"],
            mode="lines", name="RSI 14",
            line=dict(color="#e91e63", width=1.5),
        ), row=rsi_row, col=1)
        for lvl, clr, lbl in [(70, "rgba(239,83,80,0.5)", "OB 70"),
                               (30, "rgba(38,166,154,0.5)", "OS 30")]:
            fig.add_hline(y=lvl, line_dash="dash", line_color=clr,
                          annotation_text=lbl, annotation_position="right",
                          row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1)

    if show_macd:
        macd_row = 2 + (1 if show_rsi else 0)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_line"],
            mode="lines", name="MACD",
            line=dict(color="#2196f3", width=1.5),
        ), row=macd_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"],
            mode="lines", name="Signal",
            line=dict(color="#ff9800", width=1.5),
        ), row=macd_row, col=1)
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"],
            name="Histogram",
            marker_color=["#26a69a" if v >= 0 else "#ef5350"
                          for v in df["MACD_hist"].fillna(0)],
            opacity=0.6,
        ), row=macd_row, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=460 + (150 if show_rsi else 0) + (160 if show_macd else 0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
        margin=dict(l=50, r=20, t=40, b=40),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Volume chart ──────────────────────────────────────────────────────────
    vol_colors = ["#26a69a" if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
    fig_vol = go.Figure([
        go.Bar(x=df.index, y=df["Volume"],
               name="Volume", marker_color=vol_colors, opacity=0.8),
        go.Scatter(x=df.index, y=df["Vol_SMA_20"],
                   mode="lines", name="Vol SMA 20",
                   line=dict(color="#ffb300", width=1.5)),
    ])
    fig_vol.update_layout(
        title="Volume + SMA 20", template="plotly_dark", height=180,
        margin=dict(l=50, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="top", y=1.15, x=0),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    with st.expander("Raw OHLCV + Indicators (last 50 rows)"):
        display_cols = ["Open", "High", "Low", "Close", "Volume",
                        "EMA_20", "EMA_50", "RSI_14",
                        "MACD_line", "MACD_signal", "MACD_hist",
                        "BB_upper", "BB_mid", "BB_lower"]
        st.dataframe(
            df[[c for c in display_cols if c in df.columns]]
            .tail(50).sort_index(ascending=False).round(3),
            use_container_width=True,
        )
