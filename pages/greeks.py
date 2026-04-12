"""Options Greeks — Single-contract pricer and portfolio aggregator tab."""
from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from analysis.greeks import (
        black_scholes_price,
        compute_greeks,
        portfolio_greeks,
    )
    _GREEKS_AVAILABLE = True
except Exception as _import_err:  # noqa: BLE001
    _GREEKS_AVAILABLE = False
    _IMPORT_ERROR = str(_import_err)


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt(val: float, decimals: int = 4) -> str:
    return f"{val:.{decimals}f}"


def _payoff_figure(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str, price: float
) -> go.Figure:
    """Plotly figure: expiry P&L vs BS current value across a spot range."""
    spots = [S * (0.7 + 0.006 * i) for i in range(101)]  # S*0.7 → S*1.3

    expiry_pnl: list[float] = []
    bs_pnl: list[float] = []

    for sp in spots:
        if option_type == "call":
            intrinsic = max(0.0, sp - K)
        else:
            intrinsic = max(0.0, K - sp)
        expiry_pnl.append(intrinsic - price)

        bs_val = black_scholes_price(sp, K, T, r, sigma, option_type) if T > 0 else intrinsic
        bs_pnl.append(bs_val - price)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spots, y=expiry_pnl,
        mode="lines", name="Expiry P&L",
        line=dict(color="#636EFA", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=spots, y=bs_pnl,
        mode="lines", name="Current BS value",
        line=dict(color="#EF553B"),
    ))
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="grey")
    fig.add_vline(x=S, line_width=1, line_dash="dot", line_color="grey",
                  annotation_text="Spot", annotation_position="top")
    fig.add_vline(x=K, line_width=1, line_dash="dot", line_color="orange",
                  annotation_text="Strike", annotation_position="top")
    fig.update_layout(
        title="Payoff diagram (P&L per share)",
        xaxis_title="Underlying price",
        yaxis_title="P&L ($)",
        legend=dict(orientation="h", y=-0.15),
        height=380,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def _delta_bar_figure(tickers: list[str], deltas: list[float]) -> go.Figure:
    colours = ["#636EFA" if d >= 0 else "#EF553B" for d in deltas]
    fig = go.Figure(go.Bar(x=tickers, y=deltas, marker_color=colours))
    fig.update_layout(
        title="Delta contribution by position",
        xaxis_title="Position",
        yaxis_title="Delta (shares equivalent)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


# ── default portfolio rows ────────────────────────────────────────────────────

_DEFAULT_PORTFOLIO = pd.DataFrame({
    "Ticker":  ["", ""],
    "S":       [450.0, 450.0],
    "K":       [450.0, 460.0],
    "Days":    [30, 30],
    "r%":      [5.0, 5.0],
    "sigma%":  [25.0, 25.0],
    "Type":    ["call", "put"],
    "Qty":     [1, -1],
})

_COLUMN_CONFIG = {
    "Ticker": st.column_config.TextColumn("Ticker"),
    "S":      st.column_config.NumberColumn("S", min_value=0.01, format="%.2f"),
    "K":      st.column_config.NumberColumn("K", min_value=0.01, format="%.2f"),
    "Days":   st.column_config.NumberColumn("Days", min_value=0, step=1, format="%d"),
    "r%":     st.column_config.NumberColumn("r%", min_value=0.0, max_value=100.0, format="%.2f"),
    "sigma%": st.column_config.NumberColumn("sigma%", min_value=0.01, max_value=500.0, format="%.2f"),
    "Type":   st.column_config.SelectboxColumn("Type", options=["call", "put"]),
    "Qty":    st.column_config.NumberColumn("Qty", step=1, format="%d"),
}


# ── main render ───────────────────────────────────────────────────────────────

def render() -> None:  # noqa: C901
    st.header("🧮 Options Greeks")

    if not _GREEKS_AVAILABLE:
        st.error(f"Greeks engine unavailable: {_IMPORT_ERROR}")
        return

    # ── Section 1: single-contract pricer ────────────────────────────────────
    st.subheader("Single Contract Pricer")

    col_l, col_r = st.columns([1, 2])

    with col_l:
        S = st.number_input("Underlying Price (S)", min_value=0.01, value=450.0,
                            step=1.0, format="%.2f", key="g_S")
        K = st.number_input("Strike Price (K)", min_value=0.01, value=450.0,
                            step=1.0, format="%.2f", key="g_K")
        days = st.number_input("Days to Expiry", min_value=0, value=30,
                               step=1, key="g_days")
        r_pct = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.1, key="g_r")
        sigma_pct = st.slider("Implied Volatility (%)", 1, 200, 25, 1, key="g_sigma")
        option_type = st.selectbox("Option Type", ["call", "put"], key="g_type")
        contract_price_input = st.number_input(
            "Contract Price (for IV, 0 = skip)", min_value=0.0, value=0.0,
            step=0.01, format="%.2f", key="g_cprice",
        )

    T = days / 365.0
    r = r_pct / 100.0
    sigma = sigma_pct / 100.0
    contract_price = contract_price_input if contract_price_input > 0.0 else None

    g = compute_greeks(S, K, T, r, sigma, option_type, contract_price)
    theo_price = black_scholes_price(S, K, T, r, sigma, option_type)

    with col_r:
        m1, m2, m3 = st.columns(3)
        m4, m5, m6, m7 = st.columns(4)

        m1.metric("Theoretical Price", f"${theo_price:.4f}")
        m2.metric("Delta", _fmt(g.delta))
        m3.metric("Gamma", _fmt(g.gamma))
        m4.metric("Theta / day", f"${g.theta:.4f}")
        m5.metric("Vega / 1% vol", f"${g.vega:.4f}")
        m6.metric("Rho / 1% rate", f"${g.rho:.4f}")
        if contract_price is not None:
            m7.metric("IV (estimated)", f"{g.iv * 100:.2f}%")
        else:
            m7.metric("IV (input)", f"{sigma_pct:.1f}%")

        display_price = contract_price if contract_price is not None else theo_price
        st.plotly_chart(
            _payoff_figure(S, K, T, r, sigma, option_type, display_price),
            use_container_width=True,
        )

    st.divider()

    # ── Section 2: portfolio Greeks aggregator ────────────────────────────────
    st.subheader("Portfolio Greeks Aggregator")
    st.caption(
        "Enter positions below. Qty is signed: positive = long, negative = short. "
        "Each row = 1 contract (100 shares)."
    )

    edited_df: pd.DataFrame = st.data_editor(
        _DEFAULT_PORTFOLIO,
        column_config=_COLUMN_CONFIG,
        num_rows="dynamic",
        use_container_width=True,
        key="g_portfolio_editor",
    )

    if st.button("Calculate Portfolio Greeks", type="primary", key="g_calc"):
        positions = []
        tickers: list[str] = []
        per_position_delta: list[float] = []

        for _, row in edited_df.iterrows():
            qty = int(row["Qty"]) if not math.isnan(float(row["Qty"])) else 0
            if qty == 0:
                continue
            row_T = max(float(row["Days"]), 0) / 365.0
            row_r = float(row["r%"]) / 100.0
            row_sigma = float(row["sigma%"]) / 100.0
            pos = {
                "S": float(row["S"]),
                "K": float(row["K"]),
                "T": row_T,
                "r": row_r,
                "sigma": row_sigma,
                "option_type": str(row["Type"]),
                "qty": qty,
            }
            positions.append(pos)

            ticker_label = str(row["Ticker"]).strip() or f"Pos {len(tickers) + 1}"
            tickers.append(ticker_label)

            # Per-position delta for bar chart
            g_pos = compute_greeks(pos["S"], pos["K"], pos["T"], pos["r"],
                                   pos["sigma"], pos["option_type"])
            per_position_delta.append(g_pos.delta * qty * 100.0)

        if not positions:
            st.warning("Add at least one position with non-zero Qty.")
            return

        agg = portfolio_greeks(positions)

        pa, pb, pc, pd_ = st.columns(4)
        pa.metric("Total Delta (shares)", f"{agg['delta']:.2f}")
        pb.metric("Total Gamma", f"{agg['gamma']:.4f}")
        pc.metric("Total Theta / day", f"${agg['theta']:.2f}")
        pd_.metric("Total Vega / 1%", f"${agg['vega']:.2f}")

        if per_position_delta:
            st.plotly_chart(
                _delta_bar_figure(tickers, per_position_delta),
                use_container_width=True,
            )
