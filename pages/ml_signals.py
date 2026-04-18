"""
pages/ml_signals.py — ML Alpha Signals tab.

Exposes three parallel alpha model pipelines:
  - LightGBM gradient-boosting model (non-linear, high-capacity)
  - Ridge regression linear model (interpretable factor loadings)
  - Ensemble blend of the two (weighted average, reduces overfitting)

All heavy computation is wrapped in st.spinner() to avoid blocking the UI.
Optional-dep imports (lightgbm, sklearn) are performed inside button blocks so
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
        "Three parallel alpha models trained cross-sectionally to predict 5-day forward returns. "
        "LightGBM: non-linear, high-capacity.  Ridge: linear, interpretable factor loadings. "
        "Ensemble: weighted blend of both.  All signals ranked and normalised to [−1, 1]."
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

    # ── Label type selector (López de Prado Ch 3) ─────────────────────────────
    label_type = st.selectbox(
        "Label type",
        options=["fwd_ret", "triple_barrier"],
        index=0,
        key="ml_label_type",
        help=(
            "fwd_ret: regressor on 5-day forward return (default). "
            "triple_barrier: classifier on the {-1, 0, +1} label from the "
            "first-touched profit-take / stop-loss / vertical barrier."
        ),
    )
    pt_sl: tuple[float, float] = (1.0, 1.0)
    num_days = 5
    if label_type == "triple_barrier":
        lab_pt, lab_sl, lab_nd = st.columns(3)
        with lab_pt:
            pt_val = st.number_input(
                "Profit-take × σ", min_value=0.5, max_value=5.0,
                value=1.0, step=0.25, key="ml_pt",
            )
        with lab_sl:
            sl_val = st.number_input(
                "Stop-loss × σ", min_value=0.5, max_value=5.0,
                value=1.0, step=0.25, key="ml_sl",
            )
        with lab_nd:
            num_days = int(st.number_input(
                "Vertical barrier (days)", min_value=2, max_value=30,
                value=5, step=1, key="ml_nd",
            ))
        pt_sl = (float(pt_val), float(sl_val))

    # ── Train / Retrain ───────────────────────────────────────────────────────
    col_lgbm, col_regime, col_ridge, col_bayes, col_tune = st.columns(5)
    with col_lgbm:
        do_train = st.button("Train LGBM Baseline", type="primary", key="ml_train_btn")
    with col_regime:
        do_train_regime = st.button("Train Regime Models", key="ml_train_regime_btn")
    with col_ridge:
        do_train_ridge = st.button("Train Ridge Model", key="ml_train_ridge_btn")
    with col_bayes:
        do_train_bayes = st.button(
            "Train Bayesian Model", key="ml_train_bayes_btn",
            help="BayesianRidge — posterior-std exposed for Kelly sizing (ML4T Ch 10)."
        )
    with col_tune:
        tune_trials = int(st.number_input(
            "Tune trials", min_value=5, max_value=100, value=20, step=5,
            key="ml_tune_trials",
        ))
        do_tune = st.button("Optimize Hyperparameters", key="ml_tune_btn")

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
                    metrics = model.train(
                        selected_tickers, period=period,
                        label_type=label_type, pt_sl=pt_sl, num_days=num_days,
                        lgbm_params=st.session_state.get("ml_best_params"),
                    )
                    st.session_state["ml_model_instance"] = model
                    st.session_state["ml_train_metrics"] = metrics
                    st.success("LGBM baseline model trained successfully.")
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
                    regime_results = model.train_regime_models(
                        selected_tickers, period=period,
                        label_type=label_type, pt_sl=pt_sl, num_days=num_days,
                        lgbm_params=st.session_state.get("ml_best_params"),
                    )
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

    if do_train_ridge:
        with st.spinner("Building feature matrix and training Ridge linear model…"):
            try:
                from strategies.linear_signal import _SKLEARN_AVAILABLE, LinearSignal
                if not _SKLEARN_AVAILABLE:
                    st.error(
                        "scikit-learn is not installed. "
                        "Run `pip install scikit-learn>=1.3.0` and restart the app."
                    )
                else:
                    ridge_model = LinearSignal()
                    ridge_metrics = ridge_model.train(selected_tickers, period=period)
                    st.session_state["ridge_model_instance"] = ridge_model
                    st.session_state["ridge_train_metrics"] = ridge_metrics
                    st.success("Ridge model trained successfully.")
            except Exception as exc:
                st.error(f"Ridge training failed: {exc}")

    if do_train_bayes:
        with st.spinner("Training BayesianRidge alpha model…"):
            try:
                from strategies.bayesian_signal import (
                    _SKLEARN_AVAILABLE,
                    BayesianSignal,
                )
                if not _SKLEARN_AVAILABLE:
                    st.error("scikit-learn is not installed.")
                else:
                    bayes_model = BayesianSignal()
                    bayes_metrics = bayes_model.train(selected_tickers, period=period)
                    st.session_state["bayes_model_instance"] = bayes_model
                    st.session_state["bayes_train_metrics"] = bayes_metrics
                    st.success("Bayesian model trained successfully.")
            except Exception as exc:
                st.error(f"Bayesian training failed: {exc}")

    if do_tune:
        with st.spinner(f"Running Optuna TPE search ({tune_trials} trials)…"):
            try:
                from strategies.ml_tuning import (
                    _OPTUNA_AVAILABLE,
                    save_best_params,
                    tune_lgbm_hyperparams,
                )
                if not _OPTUNA_AVAILABLE:
                    st.error(
                        "optuna is not installed. "
                        "Run `pip install optuna>=3.5.0` and restart the app."
                    )
                else:
                    tune_result = tune_lgbm_hyperparams(
                        selected_tickers, period=period, n_trials=tune_trials,
                    )
                    st.session_state["ml_best_params"] = tune_result["best_params"]
                    st.session_state["ml_tune_result"] = tune_result
                    save_best_params(
                        "lgbm_alpha",
                        tune_result["best_params"],
                        tune_result["best_ic"],
                    )
                    st.success(
                        f"Tuning complete — best IC {tune_result['best_ic']:.4f} "
                        f"over {tune_result['n_trials']} trials."
                    )
            except Exception as exc:
                st.error(f"Hyperparameter tuning failed: {exc}")

    if "ml_tune_result" in st.session_state:
        tr = st.session_state["ml_tune_result"]
        st.caption("**Best Hyperparameters (Optuna TPE)**")
        tm1, tm2 = st.columns(2)
        tm1.metric("Best IC", f"{tr.get('best_ic', 0):.4f}")
        tm2.metric("Trials", str(tr.get("n_trials", 0)))
        params_df = pd.DataFrame(
            [(k, v) for k, v in tr.get("best_params", {}).items()],
            columns=["param", "value"],
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    # ── LGBM training metrics ─────────────────────────────────────────────────
    if "ml_train_metrics" in st.session_state:
        m = st.session_state["ml_train_metrics"]
        st.caption("**LGBM Baseline Metrics**")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Train IC",   f"{m.get('train_ic', 0):.4f}")
        mc2.metric("Test IC",    f"{m.get('test_ic', 0):.4f}")
        mc3.metric("Train ICIR", f"{m.get('train_icir', 0):.3f}")
        mc4.metric("Test ICIR",  f"{m.get('test_icir', 0):.3f}")

    # ── Ridge training metrics ────────────────────────────────────────────────
    if "ridge_train_metrics" in st.session_state:
        rm = st.session_state["ridge_train_metrics"]
        st.caption("**Ridge Linear Metrics**")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Train IC",   f"{rm.get('train_ic', 0):.4f}")
        rc2.metric("Test IC",    f"{rm.get('test_ic', 0):.4f}")
        rc3.metric("Train ICIR", f"{rm.get('train_icir', 0):.3f}")
        rc4.metric("Test ICIR",  f"{rm.get('test_icir', 0):.3f}")

    # ── Bayesian training metrics ─────────────────────────────────────────────
    if "bayes_train_metrics" in st.session_state:
        bm = st.session_state["bayes_train_metrics"]
        st.caption("**Bayesian Ridge Metrics**")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Train IC",   f"{bm.get('train_ic', 0):.4f}")
        bc2.metric("Test IC",    f"{bm.get('test_ic', 0):.4f}")
        bc3.metric("Train ICIR", f"{bm.get('train_icir', 0):.3f}")
        bc4.metric("Test ICIR",  f"{bm.get('test_icir', 0):.3f}")

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
    st.caption(
        "Compute all four signals at once, or use individual tabs to score each model separately."
    )

    enable_sentiment = st.checkbox(
        "Enable sentiment blend",
        value=False,
        key="ml_enable_sentiment",
        help=(
            "Include cross-sectional sentiment (Jansen Ch 14) as a fourth "
            "weighted source in the ensemble.  Uses the SENTIMENT_PROVIDER "
            "env var — VADER by default."
        ),
    )

    if st.button("Compute All Scores", key="ml_compute_all_btn", type="primary"):
        with st.spinner("Scoring all models…"):
            try:
                from strategies.bayesian_signal import BayesianSignal
                from strategies.ensemble_signal import blend_signals
                from strategies.linear_signal import LinearSignal
                from strategies.ml_signal import MLSignal

                lgbm_model = st.session_state.get("ml_model_instance") or MLSignal()
                lgbm_scores = lgbm_model.predict(selected_tickers, period="6mo")
                st.session_state["ml_model_instance"] = lgbm_model
                st.session_state["ml_scores"] = lgbm_scores

                ridge_model = st.session_state.get("ridge_model_instance") or LinearSignal()
                ridge_scores = ridge_model.predict(selected_tickers, period="6mo")
                st.session_state["ridge_model_instance"] = ridge_model
                st.session_state["ridge_scores"] = ridge_scores

                bayes_model = st.session_state.get("bayes_model_instance") or BayesianSignal()
                bayes_scores, bayes_sigma = bayes_model.predict_with_uncertainty(
                    selected_tickers, period="6mo",
                )
                st.session_state["bayes_model_instance"] = bayes_model
                st.session_state["bayes_scores"] = bayes_scores
                st.session_state["bayes_sigma"] = bayes_sigma

                blend_sources: list[dict] = [lgbm_scores, ridge_scores, bayes_scores]
                blend_model_names: list[str] = ["lgbm_alpha", "linear_ridge", "bayesian"]

                if enable_sentiment:
                    from strategies.sentiment_signal import sentiment_alpha_scores

                    sentiment_scores = sentiment_alpha_scores(selected_tickers)
                    st.session_state["sentiment_scores"] = sentiment_scores
                    if sentiment_scores:
                        blend_sources.append(sentiment_scores)
                        blend_model_names.append("sentiment")

                ensemble_scores = blend_signals(
                    *blend_sources, model_names=blend_model_names,
                )
                st.session_state["ensemble_scores"] = ensemble_scores
            except Exception as exc:
                st.error(f"Scoring failed: {exc}")

    tab_lgbm, tab_ridge, tab_bayes, tab_dl, tab_ensemble = st.tabs(
        ["LGBM", "Ridge", "Bayesian", "DL", "Ensemble"]
    )

    with tab_lgbm:
        if st.button("Compute LGBM Scores", key="ml_predict_btn"):
            with st.spinner("Scoring tickers with LGBM…"):
                try:
                    from strategies.ml_signal import MLSignal
                    model = st.session_state.get("ml_model_instance") or MLSignal()
                    scores = model.predict(selected_tickers, period="6mo")
                    st.session_state["ml_scores"] = scores
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        if "ml_scores" in st.session_state:
            _render_alpha_chart(st.session_state["ml_scores"])

    with tab_ridge:
        if st.button("Compute Ridge Scores", key="ridge_predict_btn"):
            with st.spinner("Scoring tickers with Ridge…"):
                try:
                    from strategies.linear_signal import LinearSignal
                    ridge_model = st.session_state.get("ridge_model_instance") or LinearSignal()
                    ridge_scores = ridge_model.predict(selected_tickers, period="6mo")
                    st.session_state["ridge_scores"] = ridge_scores
                except Exception as exc:
                    st.error(f"Ridge prediction failed: {exc}")
        if "ridge_scores" in st.session_state:
            _render_alpha_chart(st.session_state["ridge_scores"])

    with tab_bayes:
        if st.button("Compute Bayesian Scores", key="bayes_predict_btn"):
            with st.spinner("Scoring tickers with Bayesian model…"):
                try:
                    from strategies.bayesian_signal import BayesianSignal
                    bayes_model = st.session_state.get("bayes_model_instance") or BayesianSignal()
                    mean, sigma = bayes_model.predict_with_uncertainty(
                        selected_tickers, period="6mo",
                    )
                    st.session_state["bayes_scores"] = mean
                    st.session_state["bayes_sigma"] = sigma
                except Exception as exc:
                    st.error(f"Bayesian prediction failed: {exc}")
        if "bayes_scores" in st.session_state:
            _render_alpha_chart(st.session_state["bayes_scores"])
            if "bayes_sigma" in st.session_state:
                sigma_df = pd.DataFrame(
                    [
                        {"Ticker": t, "σ": s}
                        for t, s in st.session_state["bayes_sigma"].items()
                    ]
                ).sort_values("σ")
                st.caption("**Posterior predictive std (smaller = more confident)**")
                st.dataframe(sigma_df, use_container_width=True, hide_index=True)

    with tab_dl:
        st.caption(
            "LSTM sequence model over per-ticker feature windows "
            "(Jansen Ch 17-19).  Optional dep on `torch`."
        )
        dl_c1, dl_c2, dl_c3 = st.columns([2, 1, 1])
        with dl_c1:
            dl_window = int(st.number_input(
                "Window (bars)", min_value=3, max_value=60,
                value=10, step=1, key="dl_window",
            ))
        with dl_c2:
            dl_epochs = int(st.number_input(
                "Epochs", min_value=1, max_value=100,
                value=10, step=1, key="dl_epochs",
            ))
        with dl_c3:
            dl_hidden = int(st.number_input(
                "Hidden size", min_value=8, max_value=128,
                value=32, step=8, key="dl_hidden",
            ))

        do_train_dl = st.button("Train DL Model", key="dl_train_btn")
        if do_train_dl:
            with st.spinner("Training LSTM alpha model (this can take a while)…"):
                try:
                    from strategies.dl_signal import _TORCH_AVAILABLE, DLSignal
                    if not _TORCH_AVAILABLE:
                        st.error(
                            "torch is not installed.  "
                            "Run `pip install torch>=2.0` and restart."
                        )
                    else:
                        dl_model = DLSignal(window=dl_window)
                        metrics = dl_model.train(
                            selected_tickers, period=period,
                            epochs=dl_epochs, hidden=dl_hidden,
                        )
                        st.session_state["dl_model_instance"] = dl_model
                        st.session_state["dl_train_metrics"] = metrics
                        st.success("DL model trained successfully.")
                except Exception as exc:
                    st.error(f"DL training failed: {exc}")

        if "dl_train_metrics" in st.session_state:
            dm = st.session_state["dl_train_metrics"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Train IC", f"{dm.get('train_ic', 0):.4f}")
            m2.metric("Test IC",  f"{dm.get('test_ic', 0):.4f}")
            m3.metric("N Train", str(dm.get("n_train_samples", 0)))
            m4.metric("N Test",  str(dm.get("n_test_samples", 0)))

        if st.button("Compute DL Scores", key="dl_predict_btn"):
            with st.spinner("Scoring tickers with LSTM…"):
                try:
                    from strategies.dl_signal import DLSignal
                    dl_model = st.session_state.get("dl_model_instance") or DLSignal()
                    scores = dl_model.predict(selected_tickers, period="6mo")
                    st.session_state["dl_scores"] = scores
                except Exception as exc:
                    st.error(f"DL prediction failed: {exc}")
        if "dl_scores" in st.session_state:
            _render_alpha_chart(st.session_state["dl_scores"])

    with tab_ensemble:
        if st.button("Compute Ensemble Scores", key="ensemble_predict_btn"):
            with st.spinner("Blending all available signals…"):
                try:
                    from strategies.ensemble_signal import blend_signals
                    source_keys = [
                        "ml_scores", "ridge_scores", "bayes_scores", "dl_scores",
                    ]
                    source_model_names = [
                        "lgbm_alpha", "linear_ridge", "bayesian", "dl_lstm",
                    ]
                    if enable_sentiment:
                        from strategies.sentiment_signal import (
                            sentiment_alpha_scores,
                        )
                        st.session_state["sentiment_scores"] = (
                            sentiment_alpha_scores(selected_tickers)
                        )
                        source_keys.append("sentiment_scores")
                        source_model_names.append("sentiment")
                    pairs = [
                        (st.session_state.get(k, {}), name)
                        for k, name in zip(source_keys, source_model_names)
                    ]
                    non_empty = [(s, name) for s, name in pairs if s]
                    if not non_empty:
                        st.warning(
                            "Compute at least one of LGBM / Ridge / Bayesian / DL "
                            "scores first."
                        )
                    else:
                        non_empty_sources = [s for s, _ in non_empty]
                        non_empty_names = [name for _, name in non_empty]
                        st.session_state["ensemble_scores"] = blend_signals(
                            *non_empty_sources, model_names=non_empty_names,
                        )
                except Exception as exc:
                    st.error(f"Ensemble blending failed: {exc}")
        if "ensemble_scores" in st.session_state:
            _render_alpha_chart(st.session_state["ensemble_scores"])
        if enable_sentiment and "sentiment_scores" in st.session_state:
            st.caption("**Cross-sectional sentiment (z-scored, clipped)**")
            sent_df = pd.DataFrame(
                [
                    {"Ticker": t, "Sentiment z": s}
                    for t, s in st.session_state["sentiment_scores"].items()
                ]
            ).sort_values("Sentiment z", ascending=False)
            st.dataframe(sent_df, use_container_width=True, hide_index=True)

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
        apply_meta = st.checkbox(
            "Apply meta-labeling",
            value=False,
            key="ml_bt_meta",
            help=(
                "Wrap the primary LGBM signal with a RandomForest confidence "
                "filter trained on triple-barrier labels (López de Prado Ch 3)."
            ),
        )
        min_conf = st.slider(
            "Min confidence",
            min_value=0.0, max_value=0.9, value=0.5, step=0.05,
            key="ml_bt_min_conf",
            disabled=not apply_meta,
        )

        if st.button("Run ML Backtest", key="ml_backtest_btn"):
            with st.spinner(f"Building signals for {focus_ticker} and running backtest…"):
                try:
                    from backtester.engine import run_signal_backtest
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
                                raw_preds = model.score_features(ticker_fm.values)
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
                                    st.session_state.pop("ml_backtest_result_meta", None)

                                    if apply_meta:
                                        meta_result = _run_meta_labeled_backtest(
                                            ohlcv=ohlcv,
                                            ticker=focus_ticker,
                                            ticker_fm=ticker_fm,
                                            raw_preds=raw_preds,
                                            min_confidence=float(min_conf),
                                        )
                                        if meta_result is not None:
                                            st.session_state["ml_backtest_result_meta"] = meta_result

                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")

    _render_backtest_results(
        st.session_state.get("ml_backtest_result"),
        st.session_state.get("ml_backtest_result_meta"),
    )

    st.divider()

    # ── Execute ML Signals ────────────────────────────────────────────────────
    st.markdown("#### Execute ML Signals")
    st.caption(
        "Translates current alpha scores into broker orders via the configured "
        "`BROKER_PROVIDER` (paper by default). Position sizes use Kelly × regime × |score|."
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

    # ── Feature Importance & Coefficients ────────────────────────────────────
    fi_col, coef_col = st.columns(2)

    with fi_col:
        st.markdown("#### LGBM Feature Importance (MDI, Top 20)")
        st.caption(
            "Mean Decrease in Impurity — LightGBM's built-in importance "
            "(López de Prado AFML Ch 8)."
        )
        with st.spinner("Loading feature importances…"):
            try:
                from strategies.ml_signal import MLSignal
                lgbm_inst = st.session_state.get("ml_model_instance") or MLSignal()
                fi_df = lgbm_inst.feature_importance()
            except Exception:
                fi_df = pd.DataFrame(columns=["feature", "importance"])

        if fi_df.empty:
            st.info("No LGBM model. Click **Train LGBM Baseline** to train one.")
        else:
            _render_feature_importance(fi_df.head(20))

        if st.button(
            "Compute MDA Importance (permutation)", key="ml_mda_btn",
            help=(
                "Mean Decrease in Accuracy — shuffle each feature on a "
                "held-out fold and measure the drop in Spearman IC.  "
                "Slower than MDI but more robust to correlated features."
            ),
        ):
            with st.spinner("Running permutation importance…"):
                try:
                    _compute_and_store_mda(selected_tickers, period)
                except Exception as exc:
                    st.error(f"MDA computation failed: {exc}")

        if "ml_mda_importance" in st.session_state:
            _render_mda_importance(st.session_state["ml_mda_importance"])

    with coef_col:
        st.markdown("#### Ridge Feature Coefficients (Top 20)")
        with st.spinner("Loading Ridge coefficients…"):
            try:
                from strategies.linear_signal import LinearSignal
                ridge_inst = st.session_state.get("ridge_model_instance") or LinearSignal()
                coef_df = ridge_inst.feature_coefficients()
            except Exception:
                coef_df = pd.DataFrame(columns=["feature", "coefficient"])

        if coef_df.empty:
            st.info("No Ridge model. Click **Train Ridge Model** to train one.")
        else:
            _render_feature_coefficients(coef_df.head(20))

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

def _run_meta_labeled_backtest(
    *,
    ohlcv: pd.DataFrame,
    ticker: str,
    ticker_fm: pd.DataFrame,
    raw_preds,
    min_confidence: float,
):
    """Fit a meta-labeler on triple-barrier bins and backtest the filtered signal.

    Returns a ``BacktestResult`` on success or ``None`` on failure (with a
    Streamlit warning surfaced inline).  Failures are expected when sklearn
    isn't installed or when all bins are neutral — in both cases there is
    nothing meaningful to overlay.
    """
    import numpy as np

    from analysis.triple_barrier import triple_barrier_labels
    from backtester.engine import run_signal_backtest
    from strategies.meta_label import (
        MetaLabeler,
        filter_primary_by_confidence,
    )

    close = ohlcv["Close"].astype(float)
    bins = triple_barrier_labels(
        close, events=close.index, pt_sl=(1.0, 1.0), num_days=5,
    )["bin"]

    primary = pd.Series(
        np.sign(np.asarray(raw_preds, dtype=float)),
        index=ticker_fm.index,
    )

    try:
        labeler = MetaLabeler()
        labeler.fit(primary, bins, ticker_fm)
        final = labeler.predict(primary, ticker_fm)
    except (RuntimeError, ValueError) as exc:
        st.warning(f"Meta-labeling skipped: {exc}")
        return None

    filtered = filter_primary_by_confidence(
        primary, final, min_confidence=min_confidence,
    )

    return run_signal_backtest(
        ohlcv, filtered,
        strategy_name="ML Signal (meta)",
        ticker=ticker,
    )


def _render_backtest_results(raw_result, meta_result) -> None:
    """Render the raw backtest metrics + optional meta-labeled overlay."""
    if raw_result is None:
        return

    from backtester.engine import build_equity_chart

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Total Return", f"{raw_result.total_return_pct:.2f}%")
    bc2.metric("Sharpe", f"{raw_result.sharpe_ratio:.3f}")
    bc3.metric("Max DD", f"{raw_result.max_drawdown_pct:.2f}%")
    bc4.metric("Trades", str(raw_result.num_trades))

    if meta_result is None:
        st.plotly_chart(build_equity_chart(raw_result), use_container_width=True)
        return

    st.caption("**Meta-labeled backtest (RandomForest × primary LGBM)**")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Return", f"{meta_result.total_return_pct:.2f}%")
    mc2.metric("Sharpe", f"{meta_result.sharpe_ratio:.3f}")
    mc3.metric("Max DD", f"{meta_result.max_drawdown_pct:.2f}%")
    mc4.metric("Trades", str(meta_result.num_trades))

    fig = go.Figure()
    raw_eq = getattr(raw_result, "equity_curve", None)
    if raw_eq is not None and not raw_eq.empty and "Equity" in raw_eq.columns:
        fig.add_trace(go.Scatter(
            x=raw_eq.index,
            y=raw_eq["Equity"],
            mode="lines",
            name="Raw LGBM",
            line=dict(color="#3498db", width=2),
        ))
    meta_eq = getattr(meta_result, "equity_curve", None)
    if meta_eq is not None and not meta_eq.empty and "Equity" in meta_eq.columns:
        fig.add_trace(go.Scatter(
            x=meta_eq.index,
            y=meta_eq["Equity"],
            mode="lines",
            name="Meta-labeled",
            line=dict(color="#e67e22", width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Equity Curve — Raw vs Meta-labeled",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=400,
        margin=dict(l=60, r=40, t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


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


def _compute_and_store_mda(selected_tickers: list[str], period: str) -> None:
    """Compute MDA feature importance for the currently trained LGBM model
    and stash it in ``st.session_state['ml_mda_importance']``."""
    from data.features import _FEATURE_COLS, build_feature_matrix
    from strategies.ml_signal import MLSignal

    lgbm_inst = st.session_state.get("ml_model_instance") or MLSignal()
    if lgbm_inst._model is None:
        st.warning("Train the LGBM model first (MDA requires a fitted model).")
        return

    fm = build_feature_matrix(selected_tickers, period=period)
    if fm.empty:
        st.error("Feature matrix is empty — nothing to permute.")
        return

    feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
    target_col = "fwd_ret_5d"
    fm = fm.dropna(subset=[target_col])
    if fm.empty:
        st.error("No rows with a non-NaN target for MDA.")
        return

    X = fm[feature_cols]
    y = fm[target_col].values

    # Wrap the trained model so mda_importance can call .fit() / .predict()
    # against the same object without retraining a new estimator from scratch.
    class _FrozenModel:
        def __init__(self, model): self._m = model
        def fit(self, X_arr, y_arr): return self
        def predict(self, X_arr): return self._m.predict(X_arr)

    from analysis.feature_importance import mda_importance
    result = mda_importance(_FrozenModel(lgbm_inst._model), X, y)
    st.session_state["ml_mda_importance"] = result.importance


def _render_mda_importance(mda_series: pd.Series) -> None:
    if mda_series is None or mda_series.empty:
        st.info("No MDA data available.")
        return
    top = mda_series.sort_values(ascending=False).head(20)
    colors = ["#e67e22" if v >= 0 else "#95a5a6" for v in top.values]
    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="MDA Feature Importance (Δ test IC when shuffled)",
        xaxis_title="Importance (IC drop)",
        yaxis_title="Feature",
        height=max(300, len(top) * 26),
        margin=dict(l=140, r=40, t=50, b=40),
        yaxis=dict(autorange="reversed"),
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


def _render_feature_coefficients(coef_df: pd.DataFrame) -> None:
    """Horizontal bar chart of Ridge coefficients (positive=green, negative=red)."""
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in coef_df["coefficient"].values]
    fig = go.Figure(go.Bar(
        x=coef_df["coefficient"].values,
        y=coef_df["feature"].values,
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Ridge Feature Coefficients",
        xaxis_title="Coefficient",
        yaxis_title="Feature",
        height=max(300, len(coef_df) * 28),
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
