"""
pages/efficient_frontier.py — Markowitz Efficient Frontier + Portfolio Rebalancer.
"""
import pandas as pd
import streamlit as st

from data.fetcher import fetch_ohlcv, fetch_latest_price
from risk.markowitz import (
    build_efficient_frontier_chart,
    get_max_sharpe_portfolio,
    get_min_volatility_portfolio,
)
from strategies.rebalancer import compute_rebalance_trades, rebalance_summary


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_price_data(tickers_key: str, period: str) -> dict:
    """Download OHLCV for each ticker; return {ticker: close_series}."""
    result = {}
    for ticker in tickers_key.split(","):
        try:
            df = fetch_ohlcv(ticker.strip(), period)
            if df is not None and not df.empty:
                result[ticker.strip()] = df["Close"]
        except Exception:
            pass
    return result


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_current_prices(tickers_key: str) -> dict:
    """Return {ticker: latest_price} for a comma-separated ticker string."""
    prices = {}
    for ticker in tickers_key.split(","):
        t = ticker.strip()
        if not t:
            continue
        d = fetch_latest_price(t)
        if d.get("price") is not None:
            prices[t] = d["price"]
    return prices


def _parse_holdings(text: str) -> dict:
    """
    Parse a text block of 'TICKER:VALUE' lines into {ticker: float}.
    Silently skips blank lines and malformed entries.
    """
    holdings = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        parts = line.split(":", 1)
        ticker = parts[0].strip().upper()
        try:
            value = float(parts[1].strip().replace(",", ""))
        except ValueError:
            continue
        if ticker:
            holdings[ticker] = value
    return holdings


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    st.subheader("📐 Efficient Frontier & Portfolio Optimiser")
    st.caption("Monte Carlo simulation of random portfolios · Sharpe-optimal weights · Rebalance calculator")

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])
    with col_left:
        raw_tickers = st.text_input(
            "Tickers (comma-separated)",
            value="AAPL, MSFT, GOOG, AMZN",
            help="Enter 2+ tickers to compute the efficient frontier.",
        )
    with col_right:
        period = st.selectbox(
            "History window",
            ["1y", "2y", "3y", "5y"],
            index=1,
        )
        risk_free = st.number_input(
            "Risk-free rate (%)", min_value=0.0, max_value=20.0,
            value=5.0, step=0.25, format="%.2f",
        ) / 100

    tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]

    if len(tickers) < 2:
        st.warning("Enter at least 2 tickers to run the optimisation.")
        return

    # ── Fetch data ────────────────────────────────────────────────────────────
    tickers_key = ",".join(sorted(tickers))
    with st.spinner("Fetching price history…"):
        price_data = _fetch_price_data(tickers_key, period)

    missing = [t for t in tickers if t not in price_data]
    if missing:
        st.warning(f"Could not fetch data for: {', '.join(missing)}. They will be excluded.")

    if len(price_data) < 2:
        st.error("Need at least 2 tickers with valid price data.")
        return

    # ── Efficient frontier chart ───────────────────────────────────────────────
    st.divider()
    with st.spinner("Running Monte Carlo simulation…"):
        fig = build_efficient_frontier_chart(price_data, risk_free_rate=risk_free)
        max_sharpe = get_max_sharpe_portfolio(price_data, risk_free_rate=risk_free)
        min_vol    = get_min_volatility_portfolio(price_data)

    st.plotly_chart(fig, use_container_width=True)

    # ── Optimal portfolio stats ───────────────────────────────────────────────
    if max_sharpe is None:
        st.error("Optimisation failed — not enough data.")
        return

    st.subheader("Optimal Portfolios")
    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown("**Max Sharpe Ratio**")
        ms1, ms2, ms3 = st.columns(3)
        ms1.metric("Expected Return", f"{max_sharpe.expected_return * 100:.1f}%")
        ms2.metric("Volatility",      f"{max_sharpe.expected_volatility * 100:.1f}%")
        ms3.metric("Sharpe Ratio",    f"{max_sharpe.sharpe_ratio:.2f}")
        weights_df = pd.DataFrame(
            [{"Ticker": k, "Weight": f"{v * 100:.1f}%", "Allocation %": v * 100}
             for k, v in sorted(max_sharpe.weights.items(), key=lambda x: -x[1])]
        )
        st.dataframe(
            weights_df[["Ticker", "Weight"]].style.bar(
                subset=["Weight"], color="#26a69a"
            ),
            use_container_width=True, hide_index=True,
        )

    with oc2:
        if min_vol:
            st.markdown("**Min Volatility**")
            mv1, mv2, mv3 = st.columns(3)
            mv1.metric("Expected Return", f"{min_vol.expected_return * 100:.1f}%")
            mv2.metric("Volatility",      f"{min_vol.expected_volatility * 100:.1f}%")
            mv3.metric("Sharpe Ratio",    f"{min_vol.sharpe_ratio:.2f}")
            minvol_df = pd.DataFrame(
                [{"Ticker": k, "Weight": f"{v * 100:.1f}%"}
                 for k, v in sorted(min_vol.weights.items(), key=lambda x: -x[1])]
            )
            st.dataframe(minvol_df, use_container_width=True, hide_index=True)

    # ── Rebalancer ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("⚖️ Portfolio Rebalancer")
    st.caption(
        "Enter your current holdings to see what trades would move you to the "
        "max-Sharpe-ratio allocation."
    )

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        holdings_text = st.text_area(
            "Current holdings  (one per line: TICKER:VALUE)",
            height=160,
            placeholder="AAPL:15000\nMSFT:8000\nGOOG:5000",
            help="Enter market value in dollars for each position you currently hold.",
        )
    with rc2:
        total_equity = st.number_input(
            "Total equity ($)",
            min_value=1_000.0,
            value=100_000.0,
            step=1_000.0,
            format="%.2f",
            help="Total portfolio value including cash. Used to size target positions.",
        )
        min_trade = st.number_input(
            "Min trade value ($)",
            min_value=0.0,
            value=500.0,
            step=100.0,
            format="%.0f",
            help="Trades smaller than this amount are skipped.",
        )

    run_rebalance = st.button("Calculate Rebalance Trades", type="primary")

    if run_rebalance:
        current_positions = _parse_holdings(holdings_text)

        # Fetch current prices for all tickers in the optimal portfolio
        price_tickers = set(max_sharpe.weights) | set(current_positions)
        prices_key = ",".join(sorted(price_tickers))
        with st.spinner("Fetching current prices…"):
            current_prices = _fetch_current_prices(prices_key)

        missing_prices = [t for t in price_tickers if t not in current_prices]
        if missing_prices:
            st.warning(
                f"Could not fetch live prices for: {', '.join(missing_prices)}. "
                "Those tickers will be excluded from the rebalance."
            )

        trades = compute_rebalance_trades(
            current_positions=current_positions,
            target_weights=max_sharpe.weights,
            total_equity=total_equity,
            current_prices=current_prices,
            min_trade_value=min_trade,
        )

        if not trades:
            st.success("Portfolio is already within tolerance — no trades needed.")
        else:
            summary = rebalance_summary(trades)

            # ── Summary metrics ───────────────────────────────────────────
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Trades",        summary["num_trades"])
            sm2.metric("Total Buys",    f"${summary['total_buys']:,.0f}")
            sm3.metric("Total Sells",   f"${summary['total_sells']:,.0f}")
            sm4.metric("Net Cash",      f"${summary['net_cash_impact']:+,.0f}")
            sm5.metric("Est. Commission", f"${summary['estimated_commission']:.2f}")

            # ── Trade table ───────────────────────────────────────────────
            trade_rows = []
            for t in trades:
                trade_rows.append({
                    "Ticker":          t.ticker,
                    "Action":          t.action.upper(),
                    "Current Value":   t.current_value,
                    "Target Value":    t.target_value,
                    "Delta ($)":       t.delta_value,
                    "Delta (%)":       t.delta_pct,
                    "Approx Shares":   t.shares_approx,
                })
            trades_df = pd.DataFrame(trade_rows)

            def _action_color(val: str) -> str:
                return (
                    "color: #26a69a; font-weight: bold" if val == "BUY"
                    else "color: #ef5350; font-weight: bold"
                )

            def _delta_color(val: float) -> str:
                return f"color: {'#26a69a' if val >= 0 else '#ef5350'}"

            styled = (
                trades_df.style
                .map(_action_color, subset=["Action"])
                .map(_delta_color,  subset=["Delta ($)", "Delta (%)"])
                .format({
                    "Current Value":  "${:,.0f}",
                    "Target Value":   "${:,.0f}",
                    "Delta ($)":      "${:+,.0f}",
                    "Delta (%)":      "{:+.1f}%",
                })
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.info(
                "Review these trades before execution — "
                "this tool does not place orders automatically."
            )
