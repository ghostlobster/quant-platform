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
    col_btn, col_regime_btn = st.columns(2)
    with col_btn:
        do_train = st.button("Train / Retrain Baseline Model", type="primary", key="ml_train_btn")
    with col_regime_btn:
        do_train_regime = st.button("Train Regime Models", key="ml_train_regime_btn")

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
                    st.success("Baseline model trained successfully.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")

    if do_train_regime:
        with st.spinner("Fetching SPY/VIX regime history and training per-regime models…"):
            try:
                from strategies.ml_signal import _LGBM_AVAILABLE, MLSignal
                if not _LGBM_AVAILABLE:
                    st.error("lightgbm is not installed.")
                else:
                    model = st.session_state.get("ml_model_instance") or MLSignal()
                    regime_results = model.train_regime_models(selected_tickers, period=period)
                    st.session_state["ml_model_instance"] = model
                    st.session_state["ml_regime_results"] = regime_results
                    if regime_results:
                        st.success(
                            f"Regime models trained for: {', '.join(regime_results.keys())}"
                        )
                    else:
                        st.warning(
                            "No regimes had enough samples to train. "
                            "Try a longer training period."
                        )
            except Exception as exc:
                st.error(f"Regime training failed: {exc}")

    # ── Training metrics ──────────────────────────────────────────────────────
    if "ml_train_metrics" in st.session_state:
        m = st.session_state["ml_train_metrics"]
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Train IC",   f"{m.get('train_ic', 0):.4f}")
        mc2.metric("Test IC",    f"{m.get('test_ic', 0):.4f}")
        mc3.metric("Train ICIR", f"{m.get('train_icir', 0):.3f}")
        mc4.metric("Test ICIR",  f"{m.get('test_icir', 0):.3f}")

    # ── Regime model metrics ──────────────────────────────────────────────────
    if "ml_regime_results" in st.session_state:
        rr = st.session_state["ml_regime_results"]
        if rr:
            st.markdown("**Regime Model Performance**")
            rows = [
                {
                    "Regime": regime,
                    "Train IC": f"{v['train_ic']:.4f}",
                    "Test IC": f"{v['test_ic']:.4f}",
                    "Train ICIR": f"{v['train_icir']:.3f}",
                    "Test ICIR": f"{v['test_icir']:.3f}",
                    "N Train": v["n_train"],
                    "N Test": v["n_test"],
                }
                for regime, v in rr.items()
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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

    # ── Backtest ML Signal ────────────────────────────────────────────────────
    st.markdown("#### Backtest ML Signal")
    st.caption(
        "Runs the trained ML model across all historical dates for a single ticker "
        "(using the full universe for cross-sectional z-scoring) and backtests the "
        "resulting long/flat signal through the event-driven engine."
    )

    if len(selected_tickers) >= 1:
        focus_ticker = st.selectbox(
            "Focus ticker for backtest",
            options=selected_tickers,
            key="ml_backtest_ticker",
        )
        if st.button("Run ML Backtest", key="ml_backtest_btn"):
            with st.spinner(f"Building signals for {focus_ticker} and running backtest…"):
                try:
                    from backtester.engine import build_equity_chart, run_signal_backtest
                    from data.features import _FEATURE_COLS, build_feature_matrix
                    from data.fetcher import fetch_ohlcv
                    from strategies.ml_signal import MLSignal

                    model = st.session_state.get("ml_model_instance") or MLSignal()
                    if model._model is None:
                        st.warning("No trained model found. Click **Train / Retrain Model** first.")
                    else:
                        fm = build_feature_matrix(selected_tickers, period=period)
                        if fm.empty:
                            st.error("Feature matrix is empty — check tickers and period.")
                        else:
                            feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
                            try:
                                ticker_fm = fm.xs(focus_ticker, level="ticker")[feature_cols].fillna(0.0)
                            except KeyError:
                                st.error(f"{focus_ticker} not found in feature matrix.")
                                ticker_fm = None

                            if ticker_fm is not None and not ticker_fm.empty:
                                raw_preds = model._model.predict(ticker_fm.values)
                                signals = pd.Series(raw_preds, index=ticker_fm.index)

                                ohlcv = fetch_ohlcv(focus_ticker, period)
                                if ohlcv is None or ohlcv.empty:
                                    st.error(f"Could not fetch OHLCV data for {focus_ticker}.")
                                else:
                                    bt_result = run_signal_backtest(
                                        ohlcv, signals,
                                        strategy_name="ML Signal",
                                        ticker=focus_ticker,
                                    )
                                    st.session_state["ml_backtest_result"] = bt_result

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

    if "ml_backtest_result" in st.session_state:
        r = st.session_state["ml_backtest_result"]
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Total Return", f"{r.total_return_pct:.2f}%")
        bc2.metric("Sharpe", f"{r.sharpe_ratio:.3f}")
        bc3.metric("Max DD", f"{r.max_drawdown_pct:.2f}%")
        bc4.metric("Trades", str(r.num_trades))
        from backtester.engine import build_equity_chart
        st.plotly_chart(build_equity_chart(r), use_container_width=True)

    st.divider()

    # ── Execute ML Signals ────────────────────────────────────────────────────
    st.markdown("#### Execute ML Signals (Paper Trading)")
    st.caption(
        "Translates current alpha scores into paper trading orders: "
        "long top-scoring tickers, exit bearish positions."
    )
    exec_threshold = st.slider(
        "Score threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
        key="ml_exec_threshold",
        help="Minimum |score| required to act. Longs: score > threshold. Exits: score < −threshold.",
    )
    exec_max_pos = st.number_input(
        "Max long positions", min_value=1, max_value=20, value=5, step=1,
        key="ml_exec_max_pos",
    )

    if st.button("Execute Signals (Paper)", key="ml_exec_btn", type="secondary"):
        if "ml_scores" not in st.session_state:
            st.warning("Compute Alpha Scores first.")
        else:
            with st.spinner("Submitting paper orders…"):
                try:
                    from strategies.ml_execution import execute_ml_signals
                    actions = execute_ml_signals(
                        st.session_state["ml_scores"],
                        threshold=exec_threshold,
                        max_positions=int(exec_max_pos),
                    )
                    if actions:
                        st.success(f"Executed {len(actions)} order(s): {', '.join(actions)}")
                    else:
                        st.info("No orders to execute — scores within neutral band or no changes needed.")
                except Exception as exc:
                    st.error(f"Execution failed: {exc}")

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
