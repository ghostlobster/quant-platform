"""
pages/ml_signals.py — ML Alpha Signals tab.

Exposes the LightGBM alpha model pipeline:
  - Train / retrain the model on a selected ticker universe
  - Display training IC / ICIR metrics
  - Show current alpha scores per ticker (colour-coded bar chart)
  - Feature importance chart (top 20)
  - Information Coefficient table for each feature

All heavy computation is wrapped in st.spinner() to avoid blocking the UI.
Optional-dep imports (lightgbm, etc.) are performed inside button blocks so
that a missing package does not crash the entire Streamlit app at startup.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Reuse the existing 32-ticker universe defined in the screener
from screener.screener import TICKERS


def render() -> None:
    st.subheader("ML Alpha Signals")
    st.caption(
        "LightGBM gradient-boosting model trained cross-sectionally to predict "
        "5-day forward returns.  Signals are ranked and normalised to [−1, 1] across "
        "the universe.  Falls back to momentum score when no model has been trained."
    )

    # ── Universe & period selectors ───────────────────────────────────────────
    selected_tickers = st.multiselect(
        "Ticker universe",
        options=TICKERS,
        default=TICKERS,
        key="ml_tickers",
    )
    period = st.selectbox(
        "Training data period",
        options=["1y", "2y", "5y"],
        index=1,
        key="ml_period",
    )

    if not selected_tickers:
        st.warning("Select at least one ticker.")
        return

    # ── Train / Retrain ───────────────────────────────────────────────────────
    col_btn, col_status = st.columns([2, 3])
    with col_btn:
        do_train = st.button("Train / Retrain Model", type="primary", key="ml_train_btn")

    if do_train:
        with st.spinner("Building feature matrix and training LightGBM alpha model…"):
            try:
                from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
                if not _LGBM_AVAILABLE:
                    st.error(
                        "lightgbm is not installed. "
                        "Run `pip install lightgbm>=4.0.0` and restart the app."
                    )
                else:
                    model = MLSignal()
                    metrics = model.train(selected_tickers, period=period)
                    st.session_state["ml_model_instance"] = model
                    st.session_state["ml_train_metrics"] = metrics
                    st.success("Model trained successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    # ── Training metrics ──────────────────────────────────────────────────────
    if "ml_train_metrics" in st.session_state:
        m = st.session_state["ml_train_metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Train IC",   f"{m.get('train_ic', 0):.4f}")
        mc2.metric("Test IC",    f"{m.get('test_ic', 0):.4f}")
        mc3.metric("Train ICIR", f"{m.get('train_icir', 0):.3f}")
        mc4.metric("Test ICIR",  f"{m.get('test_icir', 0):.3f}")

    st.divider()

    # ── Alpha scores ──────────────────────────────────────────────────────────
    st.markdown("#### Current Alpha Scores")
    if st.button("Compute Alpha Scores", key="ml_predict_btn"):
        with st.spinner("Scoring tickers…"):
            try:
                from strategies.ml_signal import MLSignal
                model = st.session_state.get("ml_model_instance") or MLSignal()
                scores = model.predict(selected_tickers, period="6mo")
                st.session_state["ml_scores"] = scores
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    if "ml_scores" in st.session_state:
        _render_alpha_chart(st.session_state["ml_scores"])

    st.divider()

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown("#### Feature Importance (Top 20)")
    with st.spinner("Loading feature importances…"):
        try:
            from strategies.ml_signal import MLSignal
            model = st.session_state.get("ml_model_instance") or MLSignal()
            fi_df = model.feature_importance()
        except Exception:
            fi_df = pd.DataFrame(columns=["feature", "importance"])

    if fi_df.empty:
        st.info("No trained model found. Click **Train / Retrain Model** to train one.")
    else:
        _render_feature_importance(fi_df.head(20))

    st.divider()

    # ── IC / ICIR Table ───────────────────────────────────────────────────────
    st.markdown("#### Information Coefficient Analysis")
    st.caption(
        "IC = Spearman correlation of each feature vs 5-day forward return, averaged "
        "across dates.  ICIR > 0.5 indicates a consistent signal."
    )
    if st.button("Compute IC Table", key="ml_ic_btn"):
        with st.spinner("Computing IC statistics — this may take a moment…"):
            try:
                from analysis.factor_ic import compute_ic
                from data.features import build_feature_matrix
                fm = build_feature_matrix(selected_tickers, period=period)
                if fm.empty:
                    st.warning("Feature matrix is empty — check tickers and period.")
                else:
                    ic_results = compute_ic(fm)
                    st.session_state["ml_ic_results"] = ic_results
            except Exception as exc:
                st.error(f"IC computation failed: {exc}")

    if "ml_ic_results" in st.session_state:
        _render_ic_table(st.session_state["ml_ic_results"])


# ── Private rendering helpers ─────────────────────────────────────────────────

def _render_alpha_chart(scores: dict[str, float]) -> None:
    """Colour-coded horizontal bar chart of alpha scores."""
    if not scores:
        st.info("No scores available.")
        return

    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    tickers = [t for t, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=tickers,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Alpha Scores (ranked, 5-day forward return prediction)",
        xaxis_title="Score",
        yaxis_title="Ticker",
        xaxis=dict(range=[-1.1, 1.1]),
        height=max(300, len(tickers) * 22),
        margin=dict(l=80, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_feature_importance(fi_df: pd.DataFrame) -> None:
    """Horizontal bar chart of feature importances."""
    fig = go.Figure(go.Bar(
        x=fi_df["importance"].values,
        y=fi_df["feature"].values,
        orientation="h",
        marker_color="#3498db",
    ))
    fig.update_layout(
        title="LightGBM Feature Importances",
        xaxis_title="Importance (split count)",
        yaxis_title="Feature",
        height=max(300, len(fi_df) * 28),
        margin=dict(l=140, r=40, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_ic_table(ic_results: dict[str, dict]) -> None:
    """Styled DataFrame: Feature | IC Mean | IC Std | ICIR | p-value | N Dates."""
    if not ic_results:
        st.info("No IC results to display.")
        return

    rows = []
    for feat, stats in ic_results.items():
        rows.append({
            "Feature": feat,
            "IC Mean": round(stats.get("ic_mean", float("nan")), 4),
            "IC Std": round(stats.get("ic_std", float("nan")), 4),
            "ICIR": round(stats.get("icir", float("nan")), 3),
            "p-value": round(stats.get("p_value", float("nan")), 4),
            "N Dates": stats.get("n_dates", 0),
        })

    df = pd.DataFrame(rows).sort_values("ICIR", ascending=False)

    def _colour_icir(val: float) -> str:
        if pd.isna(val):
            return ""
        if val > 0.5:
            return "background-color: #d5f5e3"
        if val < -0.5:
            return "background-color: #fadbd8"
        return ""

    styled = df.style.applymap(_colour_icir, subset=["ICIR"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
