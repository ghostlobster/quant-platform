"""
pages/portfolio.py — Paper trading portfolio tab.
"""
import numpy as np
import streamlit as st

from data.fetcher import fetch_latest_price
from broker.paper_trader import (
    buy as pt_buy,
    sell as pt_sell,
    get_portfolio,
    get_trade_history,
    get_account,
    reset_account,
    STARTING_CASH,
)


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

    # ---- Risk Metrics ----
    with st.expander("📊 Portfolio Risk Metrics", expanded=False):
        try:
            from analysis.risk_metrics import compute_risk_metrics

            _sample_vals = [100.0 * (1.01 ** i) for i in range(30)]  # placeholder
            _metrics = compute_risk_metrics(_sample_vals)
            if _metrics is None:
                st.info("Need at least 5 trading days of data to compute risk metrics.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("VaR 95%", f"{_metrics.var_95 * 100:.2f}%")
                c2.metric("VaR 99%", f"{_metrics.var_99 * 100:.2f}%")
                c3.metric("CVaR 95%", f"{_metrics.cvar_95 * 100:.2f}%")
                c4, c5, c6 = st.columns(3)
                c4.metric("CVaR 99%", f"{_metrics.cvar_99 * 100:.2f}%")
                c5.metric("Ann. Volatility", f"{_metrics.volatility_annual:.2f}%")
                c6.metric("Worst Day", f"{_metrics.worst_day_pct:.2f}%")
        except Exception as exc:
            st.warning(f"Risk metrics unavailable: {exc}")

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
