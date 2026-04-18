"""
pages/model_health.py — Knowledge-adaption dashboard.

Surfaces the state ``KnowledgeAdaptionAgent`` sees every run — the verdict
plus the observational signals that produce it — so operators can see at
a glance why conviction is being softened without grepping structlog.

Four panels:
  1. Model inventory + per-model verdict + Kelly multiplier.
  2. Live vs trained IC (requires #115 ``live_predictions`` data).
  3. Regime-coverage matrix (which models cover which regimes).
  4. Retrain history with ``test_ic`` (and ``test_ic_delta`` from #122)
     sparklines.

Read-only dashboard — no writes, no broker calls.

# TODO(public-api): promote ``_read_regime_coverage`` and
# ``_confine_pickle_path`` out of ``agents/knowledge_agent.py``'s
# module-private namespace when that module is next refactored. The
# regime-coverage panel is the only consumer of those helpers today.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import structlog
from plotly.subplots import make_subplots

from agents.knowledge_agent import (
    _confine_pickle_path,
    _read_regime_coverage,
    recommendation_multiplier,
)
from analysis.regime import REGIME_STATES

log = structlog.get_logger(__name__)


# ── DB helpers (cached) ──────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner=False)
def _latest_metadata() -> pd.DataFrame:
    """Return the most recent ``model_metadata`` row per model.

    Columns: ``model_name, trained_at, test_ic, test_ic_delta,
    n_tickers, period``. Empty DataFrame when the table has no rows
    (fresh install / pre-retrain).
    """
    from data.db import get_connection

    sql = (
        "SELECT m.model_name, m.trained_at, m.test_ic, m.test_ic_delta, "
        "       m.n_tickers, m.period "
        "  FROM model_metadata m "
        "  JOIN (SELECT model_name, MAX(trained_at) AS max_ts "
        "          FROM model_metadata GROUP BY model_name) latest "
        "    ON m.model_name = latest.model_name "
        "   AND m.trained_at = latest.max_ts "
        " ORDER BY m.model_name"
    )
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return df


@st.cache_data(ttl=60, show_spinner=False)
def _retrain_history(model_name: str, n: int = 10) -> pd.DataFrame:
    """Return the last ``n`` ``model_metadata`` rows for ``model_name``."""
    from data.db import get_connection

    sql = (
        "SELECT trained_at, test_ic, test_ic_delta, n_tickers, period "
        "  FROM model_metadata "
        " WHERE model_name = ? "
        " ORDER BY trained_at DESC LIMIT ?"
    )
    conn = get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=(model_name, int(n)))
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return df.iloc[::-1].reset_index(drop=True)  # chronological for plotting


@st.cache_data(ttl=300, show_spinner="Querying knowledge agent…")
def _knowledge_verdict() -> dict:
    """Invoke KnowledgeAdaptionAgent once and return its metadata dict.

    Cached for 5 minutes — the pickle reads + alert dispatch are the most
    expensive piece of this page. Returns an empty dict on any failure so
    the inventory panel can still render the model list.
    """
    try:
        from agents.knowledge_agent import KnowledgeAdaptionAgent

        sig = KnowledgeAdaptionAgent().run({})
        return dict(sig.metadata or {})
    except Exception as exc:
        log.warning("model_health: knowledge agent run failed", error=str(exc))
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def _regime_coverage_map() -> dict[str, list[str]]:
    """Return ``{model_name: [regime, ...]}`` for every model with regime coverage.

    Today only the LightGBM regime bundle exposes coverage via its pickle
    payload; other families are baseline-only.
    """
    env_var, rel = "LGBM_REGIME_MODELS_PATH", "models/lgbm_regime_models.pkl"
    try:
        path = _confine_pickle_path(
            __import__("os").environ.get(env_var) or str(Path(rel)),
        )
        coverage = _read_regime_coverage(path)
    except Exception as exc:
        log.warning("model_health: regime coverage read failed", error=str(exc))
        coverage = []
    return {"lgbm_regime": list(coverage)}


def _rolling_live_ic(model_name: str) -> float | None:
    """Thin wrapper so tests can monkeypatch a single attribute."""
    try:
        from analysis.live_ic import rolling_live_ic

        return rolling_live_ic(model_name)
    except Exception as exc:
        log.debug("model_health: rolling_live_ic failed", error=str(exc))
        return None


# ── Panels ───────────────────────────────────────────────────────────────────

def _render_inventory_panel() -> None:
    st.markdown("### Model inventory")
    try:
        df = _latest_metadata()
    except Exception as exc:
        log.warning("model_health: inventory load failed", error=str(exc))
        st.warning(f"Could not load model inventory: {exc}")
        return

    if df.empty:
        st.info(
            "No models trained yet — run `python -m cron.monthly_ml_retrain` "
            "to populate `model_metadata`."
        )
        return

    meta = _knowledge_verdict()
    verdict = str(meta.get("recommendation") or "fresh")
    multiplier = recommendation_multiplier(verdict)
    now = pd.Timestamp.now("UTC").tz_localize(None)
    trained_at = pd.to_datetime(df["trained_at"], unit="s")
    df = df.copy()
    df["trained_at"] = trained_at
    df["days_old"] = (now - trained_at).dt.total_seconds() / 86400.0
    # Verdict + multiplier are agent-wide (one knowledge agent per process);
    # attach per row so the operator sees them alongside every model.
    df["verdict"] = verdict
    df["kelly_mult"] = multiplier

    display_cols = [
        "model_name", "trained_at", "test_ic", "test_ic_delta",
        "n_tickers", "period", "days_old", "verdict", "kelly_mult",
    ]
    st.dataframe(
        df[display_cols], use_container_width=True, hide_index=True,
    )
    st.caption(
        f"Agent verdict: **{verdict}** → Kelly multiplier **{multiplier:.2f}**. "
        f"fresh=1.0 · monitor=0.7 · retrain=0.4."
    )


def _render_live_ic_panel() -> None:
    st.markdown("### Live vs trained IC")
    try:
        df = _latest_metadata()
    except Exception as exc:
        log.warning("model_health: live IC load failed", error=str(exc))
        st.warning(f"Could not load metadata for live-IC panel: {exc}")
        return

    if df.empty:
        st.info("No models in `model_metadata` yet.")
        return

    for _, row in df.iterrows():
        model_name = str(row["model_name"])
        trained_ic = row.get("test_ic")
        live_ic = _rolling_live_ic(model_name)

        if live_ic is None:
            st.info(
                f"**{model_name}** — live IC warming up (need ≥ 30 realized "
                f"rows in `live_predictions`). Trained IC: "
                f"{trained_ic if trained_ic is not None else 'n/a'}."
            )
            continue

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=["trained", "live"],
                y=[trained_ic, live_ic],
                mode="lines+markers+text",
                text=[
                    f"{trained_ic:.3f}" if trained_ic is not None else "n/a",
                    f"{live_ic:.3f}",
                ],
                textposition="top center",
                name=model_name,
            ),
        )
        fig.update_layout(
            title=f"{model_name} — trained IC vs rolling live IC",
            yaxis_title="IC",
            showlegend=False,
            height=220,
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_regime_coverage_panel() -> None:
    st.markdown("### Regime coverage")
    try:
        inv = _latest_metadata()
        coverage = _regime_coverage_map()
    except Exception as exc:
        log.warning("model_health: coverage load failed", error=str(exc))
        st.warning(f"Could not load regime coverage: {exc}")
        return

    if inv.empty:
        st.info("No models yet — nothing to show coverage for.")
        return

    models = inv["model_name"].tolist()
    rows = []
    for regime in REGIME_STATES:
        row = {"regime": regime}
        for model in models:
            # Only regime-aware models have per-regime coverage.
            if model == "lgbm_regime":
                row[model] = "✅" if regime in coverage.get("lgbm_regime", []) else "❌"
            else:
                row[model] = "ℹ︎"
        rows.append(row)
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        "✅ covered · ❌ missing · ℹ︎ baseline model (pooled across regimes)."
    )


def _render_retrain_history_panel() -> None:
    st.markdown("### Retrain history")
    try:
        inv = _latest_metadata()
    except Exception as exc:
        log.warning("model_health: retrain history load failed", error=str(exc))
        st.warning(f"Could not load retrain history: {exc}")
        return

    if inv.empty:
        st.info("No retrain history yet.")
        return

    for model_name in inv["model_name"]:
        try:
            history = _retrain_history(str(model_name), n=10)
        except Exception as exc:
            log.warning(
                "model_health: per-model history load failed",
                model=str(model_name), error=str(exc),
            )
            st.warning(f"Could not load history for {model_name}: {exc}")
            continue

        if history.empty:
            st.info(f"**{model_name}** — no prior retrains.")
            continue

        history = history.copy()
        history["trained_at"] = pd.to_datetime(
            history["trained_at"], unit="s",
        ).dt.strftime("%Y-%m-%d")

        col_tbl, col_chart = st.columns([3, 2])
        with col_tbl:
            st.markdown(f"**{model_name}**")
            st.dataframe(
                history[["trained_at", "test_ic", "test_ic_delta",
                         "n_tickers", "period"]],
                use_container_width=True, hide_index=True,
            )
        with col_chart:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(
                    x=history["trained_at"],
                    y=history["test_ic"],
                    mode="lines+markers", name="test_ic",
                ),
                secondary_y=False,
            )
            if history["test_ic_delta"].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=history["trained_at"],
                        y=history["test_ic_delta"],
                        mode="lines", name="Δ test_ic",
                        line=dict(dash="dash"),
                    ),
                    secondary_y=True,
                )
            fig.update_layout(
                height=180,
                margin=dict(l=30, r=20, t=20, b=30),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Entry point ──────────────────────────────────────────────────────────────

def render() -> None:
    st.subheader("Model Health")
    st.caption(
        "Adoption state the knowledge agent reports before every trade. "
        "Verdict drives the Kelly multiplier: fresh=1.0, monitor=0.7, "
        "retrain=0.4. `KNOWLEDGE_AUTO_RETRAIN=1` fires "
        "`cron.monthly_ml_retrain` automatically on a retrain verdict "
        "(opt-in; see `MAINTENANCE_AND_BROKERS.md §11.3`)."
    )

    _render_inventory_panel()
    st.divider()
    _render_live_ic_panel()
    st.divider()
    _render_regime_coverage_panel()
    st.divider()
    _render_retrain_history_panel()
