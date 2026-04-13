"""
pages/backtest.py — Strategy backtester tab.
"""
from datetime import date, timedelta

import streamlit as st

from backtester.engine import build_equity_chart, build_trade_log_df, run_backtest
from data.fetcher import fetch_ohlcv

_STRATEGIES = {
    "SMA Crossover (20/50)":       "sma_crossover",
    "RSI Mean Reversion (30/70)":  "rsi_mean_revert",
}
_STRATEGY_DESC = {
    "sma_crossover":   "Buy when SMA20 crosses above SMA50; sell when it crosses below.",
    "rsi_mean_revert": "Buy when RSI(14) drops below 30 (oversold); sell when it rises above 70.",
}


def render() -> None:
    ticker = st.session_state.get("active_ticker", "AAPL")

    st.subheader(f"Strategy Backtester — {ticker}")

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns([2, 2, 2, 1])
    with bt_c1:
        bt_strategy_label = st.selectbox("Strategy", list(_STRATEGIES.keys()), key="bt_strategy")
        bt_strategy = _STRATEGIES[bt_strategy_label]
    with bt_c2:
        bt_start = st.date_input(
            "Start date",
            value=date.today() - timedelta(days=365 * 2),
            max_value=date.today() - timedelta(days=60),
            key="bt_start",
        )
    with bt_c3:
        bt_end = st.date_input(
            "End date",
            value=date.today(),
            min_value=bt_start + timedelta(days=60),
            max_value=date.today(),
            key="bt_end",
        )
    with bt_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    st.caption(_STRATEGY_DESC[bt_strategy])

    if run_bt:
        with st.spinner(f"Running {bt_strategy_label} on {ticker}…"):
            try:
                bt_raw = fetch_ohlcv(ticker, "5y")
            except ValueError:
                try:
                    bt_raw = fetch_ohlcv(ticker, "2y")
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
            try:
                result = run_backtest(
                    bt_raw,
                    strategy=bt_strategy,
                    ticker=ticker,
                    start_date=str(bt_start),
                    end_date=str(bt_end),
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.markdown("#### Results")
        rm1, rm2, rm3, rm4, rm5, rm6, rm7, rm8 = st.columns(8)
        alpha = result.total_return_pct - result.buy_hold_return_pct
        rm1.metric("Total Return", f"{result.total_return_pct:+.2f}%",
                   delta=f"{alpha:+.2f}% vs B&H",
                   delta_color="normal" if alpha >= 0 else "inverse")
        rm2.metric("Buy & Hold",   f"{result.buy_hold_return_pct:+.2f}%")
        rm3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
        rm4.metric("Sortino",      f"{result.sortino_ratio:.3f}")
        rm5.metric("Calmar",       f"{result.calmar_ratio:.3f}")
        rm6.metric("Max Drawdown", f"{result.max_drawdown_pct:.2f}%")
        rm7.metric("Win Rate",     f"{result.win_rate_pct:.1f}%")
        rm8.metric("# Trades",     str(result.num_trades))

        st.plotly_chart(build_equity_chart(result), use_container_width=True)

        with st.expander(f"Trade Log ({result.num_trades} trades)"):
            trade_df = build_trade_log_df(result)
            if trade_df.empty:
                st.info("No trades executed in this period.")
            else:
                def _colour_ret(val):
                    if not isinstance(val, (int, float)):
                        return ""
                    return f"color: {'#26a69a' if val >= 0 else '#ef5350'}; font-weight: bold"

                st.dataframe(
                    trade_df.style
                    .map(_colour_ret, subset=["Return (%)"])
                    .format({"Entry Price": "${:.2f}", "Exit Price": "${:.2f}",
                             "Return (%)": "{:+.2f}%"}),
                    use_container_width=True, hide_index=True,
                )
                st.caption(f"Average trade return: {trade_df['Return (%)'].mean():+.2f}%")
