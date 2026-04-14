"""
monitoring/sidecar.py — FastAPI metrics sidecar exposing /metrics for Prometheus.

Run as a standalone process alongside the Streamlit app:
    uvicorn monitoring.sidecar:app --host 0.0.0.0 --port 9090

Or via docker-compose (see docker-compose.yml metrics service).

ENV vars
--------
    METRICS_PORT   port to listen on (default: 9090)

Requires (optional): pip install fastapi>=0.110.0 uvicorn>=0.29.0 prometheus-client>=0.19.0
"""
from __future__ import annotations

import os

try:
    from fastapi import FastAPI  # type: ignore[import]
    from fastapi.responses import PlainTextResponse  # type: ignore[import]
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False

from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="quant-platform metrics", docs_url=None, redoc_url=None) \
    if _FASTAPI_AVAILABLE else None  # type: ignore[assignment]


def _get_metrics_text() -> str:
    """Collect and return current Prometheus metrics in text exposition format."""
    # Refresh portfolio metrics on each scrape
    try:
        from broker.paper_trader import get_account, get_portfolio
        from monitoring.metrics import update_portfolio_metrics, update_regime_metric

        acct = get_account()
        nav = acct.get("total_value", 0.0)
        portfolio_df = get_portfolio()
        open_pnl = 0.0
        if not portfolio_df.empty and "Unrealised P&L" in portfolio_df.columns:
            open_pnl = float(portfolio_df["Unrealised P&L"].fillna(0).sum())
        update_portfolio_metrics(nav=nav, open_pnl=open_pnl)
    except Exception as exc:
        logger.debug("Metrics refresh: portfolio update failed: %s", exc)

    try:
        from analysis.regime import get_live_regime
        from monitoring.metrics import update_regime_metric
        regime_data = get_live_regime()
        update_regime_metric(regime_data["regime"])
    except Exception as exc:
        logger.debug("Metrics refresh: regime update failed: %s", exc)

    try:
        from prometheus_client import generate_latest  # type: ignore
        return generate_latest().decode("utf-8")
    except ImportError:
        return "# prometheus-client not installed\n"


if _FASTAPI_AVAILABLE and app is not None:
    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint():
        """Prometheus scrape endpoint."""
        text = _get_metrics_text()
        return PlainTextResponse(
            content=text,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}


if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        raise SystemExit("fastapi and uvicorn are required. pip install fastapi uvicorn")
    import uvicorn  # type: ignore[import]
    port = int(os.environ.get("METRICS_PORT", "9090"))
    uvicorn.run("monitoring.sidecar:app", host="0.0.0.0", port=port, log_level="info")
