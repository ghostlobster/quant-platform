"""
pages/portfolio.py — Paper trading portfolio tab.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from broker.paper_trader import (
    STARTING_CASH,
    get_account,
    get_portfolio,
    get_trade_history,
    reset_account,
)
from broker.paper_trader import (
    buy as pt_buy,
)
from broker.paper_trader import (
    sell as pt_sell,
)
from data.fetcher import fetch_latest_price, fetch_ohlcv


def render() -> None:
    ticker = st.session_state.get("active_ticker", "AAPL")

    st.subheader("💼 Paper Trading Portfolio")
    st.caption(f"Virtual account — starting cash ${STARTING_CASH:,.0f} · data via Yahoo Finance")

    # ── Fetch live prices for open positions ──────────────────────────────────
    @st.cache_data(ttl=60, show_spinner=False)
    def _live_prices(tickers_key: str) -> dict[str, float]:
        prices = {}
        for t in (tickers_key.split(",") if tickers_key else []):
            d = fetch_latest_price(t)
            if d["price"] is not None:
                prices[t] = d["price"]
        return prices

    port_df = get_portfolio()
    acct    = get_account()

    if not port_df.empty:
        held_tickers = ",".join(port_df["Ticker"].tolist())
        with st.spinner("Fetching live prices…"):
            live = _live_prices(held_tickers)
        port_df = get_portfolio(current_prices=live)
        market_value = port_df["Market Value"].sum()
    else:
        live = {}
        market_value = 0.0

    cash           = acct["cash"]
    total_value    = cash + market_value
    total_pnl      = total_value - STARTING_CASH
    realised_pnl   = acct["realised_pnl"]
    unrealised_pnl = (
        port_df["Unrealised P&L"].sum()
        if not port_df.empty and port_df["Unrealised P&L"].notna().any()
        else 0.0
    )

    # ── Account summary metrics ───────────────────────────────────────────────
    pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
    pm1.metric("Cash Balance",    f"${cash:,.2f}")
    pm2.metric("Open Positions",  f"${market_value:,.2f}")
    pm3.metric("Total Value",     f"${total_value:,.2f}",
               delta=f"{total_pnl:+,.2f}",
               delta_color="normal" if total_pnl >= 0 else "inverse")
    pm4.metric("Total P&L",       f"${total_pnl:+,.2f}",
               delta=f"{(total_pnl/STARTING_CASH*100):+.2f}%",
               delta_color="normal" if total_pnl >= 0 else "inverse")
    pm5.metric("Realised P&L",    f"${realised_pnl:+,.2f}")
    pm6.metric("Unrealised P&L",  f"${unrealised_pnl:+,.2f}")

    st.divider()

    # ── Trade form ────────────────────────────────────────────────────────────
    st.subheader("Place Order")
    with st.form("trade_form", clear_on_submit=True):
        tf1, tf2, tf3, tf4, tf5 = st.columns([2, 1.5, 1.5, 1, 1])
        with tf1:
            trade_ticker = st.text_input("Ticker", value=ticker, key="pt_ticker").upper().strip()
        with tf2:
            trade_shares = st.number_input("Shares", min_value=0.001, value=1.0,
                                           step=1.0, format="%.3f", key="pt_shares")
        with tf3:
            prefill_price = live.get(trade_ticker, 0.0) if trade_ticker else 0.0
            trade_price = st.number_input("Price ($)", min_value=0.001,
                                          value=max(prefill_price, 0.001),
                                          step=0.01, format="%.4f", key="pt_price")
        with tf4:
            st.markdown("<br>", unsafe_allow_html=True)
            buy_btn  = st.form_submit_button("🟢 Buy",  type="primary",  use_container_width=True)
        with tf5:
            st.markdown("<br>", unsafe_allow_html=True)
            sell_btn = st.form_submit_button("🔴 Sell", use_container_width=True)

    # ── Order validation + execution ──────────────────────────────────────────
    if buy_btn or sell_btn:
        action = "BUY" if buy_btn else "SELL"
        if not trade_ticker:
            st.error("Enter a ticker symbol.")
        elif trade_shares <= 0:
            st.error("Shares must be > 0.")
        elif trade_price <= 0:
            st.error("Price must be > 0.")
        else:
            try:
                if action == "BUY":
                    result = pt_buy(trade_ticker, trade_shares, trade_price)
                    st.success(
                        f"✅ Bought {result['shares']:.4f} {result['ticker']} "
                        f"@ ${result['price']:.4f} · cost ${result['cost']:,.2f} · "
                        f"cash remaining ${result['cash_remaining']:,.2f}"
                    )
                else:
                    result = pt_sell(trade_ticker, trade_shares, trade_price)
                    pnl_str = f"${result['realised_pnl']:+,.2f}"
                    st.success(
                        f"✅ Sold {result['shares']:.4f} {result['ticker']} "
                        f"@ ${result['price']:.4f} · proceeds ${result['proceeds']:,.2f} · "
                        f"realised P&L {pnl_str}"
                    )
                st.rerun()
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))

    st.divider()

    # ── Open positions table ──────────────────────────────────────────────────
    st.subheader("Open Positions")

    def _pnl_style(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return ""
        return f"color: {'#26a69a' if val >= 0 else '#ef5350'}; font-weight: bold"

    if port_df.empty:
        st.info("No open positions. Place a buy order above.")
    else:
        _display_cols = [
            "Ticker", "Shares", "Avg Cost", "Current Price",
            "Market Value", "Unrealised P&L", "Unrealised %", "Cost Basis"
        ]
        display_port = port_df[[c for c in _display_cols if c in port_df.columns]].copy()

        _pnl_subset = [c for c in ["Unrealised P&L", "Unrealised %"] if c in display_port.columns]
        _fmt = {
            "Avg Cost":       "${:.4f}",
            "Current Price":  lambda v: f"${v:.2f}" if v is not None else "—",
            "Market Value":   "${:,.2f}",
            "Cost Basis":     "${:,.2f}",
            "Unrealised P&L": lambda v: f"${v:+,.2f}" if v is not None else "—",
            "Unrealised %":   lambda v: f"{v:+.2f}%" if v is not None else "—",
        }
        styled_port = (
            display_port.style
            .map(_pnl_style, subset=_pnl_subset)
            .format({k: v for k, v in _fmt.items() if k in display_port.columns}, na_rep="—")
        )
        st.dataframe(styled_port, use_container_width=True, hide_index=True)

    st.divider()

    # ── Trade history ─────────────────────────────────────────────────────────
    st.subheader("Trade History")
    hist_df = get_trade_history()
    if hist_df.empty:
        st.info("No trades executed yet.")
    else:
        def _action_style(val: str) -> str:
            if val == "BUY":
                return "color: #26a69a; font-weight: bold"
            return "color: #ef5350; font-weight: bold"

        styled_hist = (
            hist_df.style
            .map(_action_style, subset=["Action"])
            .map(_pnl_style,    subset=["Realised P&L"])
            .format({
                "Price":            "${:.4f}",
                "Amount":           "${:,.2f}",
                "Avg Cost at Sale": lambda v: f"${v:.4f}" if v is not None else "—",
                "Realised P&L":     lambda v: f"${v:+,.2f}" if v is not None else "—",
            }, na_rep="—")
        )
        st.dataframe(styled_hist, use_container_width=True, hide_index=True)

        sells = hist_df[hist_df["Action"] == "SELL"]
        if not sells.empty:
            wins  = (sells["Realised P&L"] > 0).sum()
            total = len(sells)
            st.caption(
                f"{total} sell{'s' if total != 1 else ''} · "
                f"{wins}/{total} profitable · "
                f"avg realised P&L per sell: ${sells['Realised P&L'].mean():+,.2f}"
            )

    # ── Risk Metrics ──────────────────────────────────────────────────────────
    st.divider()
    with st.expander("📊 Portfolio Risk Metrics", expanded=False):
        from analysis.risk_metrics import compute_risk_metrics

        hist_df2 = get_trade_history()
        if hist_df2.empty:
            st.info("Need at least 5 trading days of data to compute risk metrics.")
        else:
            # Build a mock equity curve from trade history total values
            # Use cumulative realised P&L + STARTING_CASH as a proxy value series
            sells2 = hist_df2[hist_df2["Action"] == "SELL"].copy()
            if len(sells2) < 5:
                st.info("Need at least 5 trading days of data to compute risk metrics.")
            else:
                cum_pnl = sells2["Realised P&L"].cumsum()
                equity_curve = (STARTING_CASH + cum_pnl).tolist()
                rm = compute_risk_metrics(equity_curve)
                if rm is None:
                    st.info("Need at least 5 trading days of data to compute risk metrics.")
                else:
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("VaR 95% (daily)",  f"{rm.var_95 * 100:.2f}%")
                    rc2.metric("VaR 99% (daily)",  f"{rm.var_99 * 100:.2f}%")
                    rc3.metric("Annual Volatility", f"{rm.volatility_annual:.2f}%")

                    rc4, rc5, rc6 = st.columns(3)
                    rc4.metric("CVaR 95% (ES)",    f"{rm.cvar_95 * 100:.2f}%")
                    rc5.metric("CVaR 99% (ES)",    f"{rm.cvar_99 * 100:.2f}%")
                    rc6.metric("Observations",     str(rm.n_observations))

                    rc7, rc8, _ = st.columns(3)
                    rc7.metric("Worst Day",  f"{rm.worst_day_pct:.2f}%")
                    rc8.metric("Best Day",   f"{rm.best_day_pct:.2f}%")

    # ── Risk Factors (PCA / k-means) ──────────────────────────────────────────
    st.divider()
    with st.expander("🧬 Risk Factors (PCA / k-means)", expanded=False):
        st.caption(
            "Decompose a ticker universe into latent statistical factors (PCA) "
            "and group assets by correlation distance (k-means). "
            "Jansen, *ML for Algorithmic Trading* Ch 13."
        )
        _render_risk_factors()

    # ── Danger zone: reset ────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚠️ Danger Zone"):
        st.warning(
            "Resetting will **permanently delete** all positions and trade history "
            f"and restore the ${STARTING_CASH:,.0f} starting balance."
        )
        if st.button("🗑️ Reset Paper Account", type="secondary", key="pt_reset"):
            reset_account()
            st.success("Account reset. Starting fresh.")
            st.rerun()


# ── Risk Factor helpers ──────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _fetch_return_matrix(tickers_key: str, period: str) -> pd.DataFrame:
    """Fetch daily-return frame for the comma-separated ticker string."""
    cols: dict[str, pd.Series] = {}
    for ticker in tickers_key.split(","):
        t = ticker.strip().upper()
        if not t:
            continue
        try:
            df = fetch_ohlcv(t, period)
            if df is not None and not df.empty:
                cols[t] = df["Close"].astype(float)
        except Exception:
            pass
    if not cols:
        return pd.DataFrame()
    prices = pd.DataFrame(cols).dropna(how="all")
    return prices.pct_change().dropna(how="all")


def _render_risk_factors() -> None:
    """PCA scree + cluster heatmap + downloadable assignments."""
    from analysis.unsupervised import (
        cluster_assets,
        cluster_members,
        pca_risk_factors,
    )

    rf_col1, rf_col2, rf_col3 = st.columns([3, 1, 1])
    with rf_col1:
        raw_tickers = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOG, AMZN, META, NVDA, JPM, XOM",
            key="rf_tickers",
            help="Enter 3+ tickers to run PCA + k-means on their daily returns.",
        )
    with rf_col2:
        rf_period = st.selectbox(
            "History",
            options=["6mo", "1y", "2y", "5y"],
            index=2,
            key="rf_period",
        )
    with rf_col3:
        rf_k = int(st.number_input(
            "Clusters (k)",
            min_value=2, max_value=12, value=4, step=1,
            key="rf_k",
            help="k-means cluster count. Capped at the number of tickers.",
        ))

    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    if len(tickers) < 3:
        st.info("Enter at least 3 tickers to run the decomposition.")
        return

    tickers_key = ",".join(sorted(set(tickers)))
    with st.spinner("Fetching daily returns…"):
        returns = _fetch_return_matrix(tickers_key, rf_period)

    if returns.empty or returns.shape[1] < 2:
        st.error("Could not build a return matrix from the selected tickers.")
        return

    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        st.warning(f"No data for: {', '.join(missing)}. Excluded.")

    # ── PCA scree ─────────────────────────────────────────────────────────────
    n_components = min(len(returns.columns), 6)
    factors = pca_risk_factors(returns, n_components=n_components)
    if factors.explained_variance.empty:
        st.info("Not enough observations for PCA (need ≥ 20 rows).")
        return

    scree_fig = go.Figure(go.Bar(
        x=list(factors.explained_variance.index),
        y=(factors.explained_variance.values * 100).tolist(),
        marker_color="#3498db",
        text=[f"{v * 100:.1f}%" for v in factors.explained_variance.values],
        textposition="outside",
    ))
    scree_fig.update_layout(
        title="PCA Scree — Variance Explained per Component",
        xaxis_title="Component",
        yaxis_title="Variance Explained (%)",
        height=320,
        margin=dict(l=60, r=40, t=50, b=40),
    )
    st.plotly_chart(scree_fig, use_container_width=True)

    # ── Loadings heatmap (tickers × components) ───────────────────────────────
    loadings = factors.loadings
    load_fig = go.Figure(data=go.Heatmap(
        z=loadings.values,
        x=list(loadings.columns),
        y=list(loadings.index),
        colorscale="RdBu",
        zmid=0,
        text=[[f"{v:+.2f}" for v in row] for row in loadings.values],
        texttemplate="%{text}",
        showscale=True,
    ))
    load_fig.update_layout(
        title="PCA Factor Loadings (ticker × component)",
        height=max(320, len(loadings.index) * 28),
        margin=dict(l=80, r=40, t=50, b=40),
    )
    st.plotly_chart(load_fig, use_container_width=True)

    # ── k-means cluster assignments ───────────────────────────────────────────
    labels = cluster_assets(returns, n_clusters=rf_k)
    if labels.empty:
        st.info("Clustering skipped — not enough data.")
        return

    members = cluster_members(labels)
    cluster_rows = [
        {"Cluster": cid, "N": len(tickers_in_cluster),
         "Tickers": ", ".join(tickers_in_cluster)}
        for cid, tickers_in_cluster in members.items()
    ]
    cluster_df = pd.DataFrame(cluster_rows)
    st.markdown("**k-means clusters (correlation distance)**")
    st.dataframe(cluster_df, use_container_width=True, hide_index=True)

    csv_df = pd.DataFrame({"ticker": labels.index, "cluster": labels.values})
    st.download_button(
        label="Download cluster assignments (CSV)",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="risk_factor_clusters.csv",
        mime="text/csv",
        key="rf_download",
    )
