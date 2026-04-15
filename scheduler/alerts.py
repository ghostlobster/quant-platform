"""
scheduler/alerts.py — SQLite-backed price & RSI alert engine.

Schema (alerts table)
---------------------
  id           : INTEGER PK autoincrement
  ticker       : TEXT    e.g. "AAPL"
  alert_type   : TEXT    one of: price_above | price_below | rsi_above | rsi_below
  threshold    : REAL    the trigger level
  enabled      : INTEGER 1 = active, 0 = disabled
  created_at   : REAL    Unix timestamp
  triggered_at : REAL    Unix timestamp of last trigger, or NULL if never fired

Public API
----------
  add_alert(ticker, alert_type, threshold) → int  (new alert id)
  get_alerts()                             → pd.DataFrame of all alerts
  delete_alert(alert_id)                  → None
  toggle_alert(alert_id, enabled)         → None
  check_alerts(current_data)              → list[dict] of triggered alerts
  init_alerts_table()                     → None  (called from app bootstrap)

Desktop notifications
---------------------
  Uses plyer.notification — fires a native desktop toast when an alert
  triggers.  Silently skipped if plyer fails (headless / CI environments).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

import pandas as pd
import structlog

from alerts.channels import broadcast
from data.db import get_connection

logger = structlog.get_logger(__name__)

# ── Alert type constants ──────────────────────────────────────────────────────

ALERT_TYPES = {
    "price_above": "Price ≥ threshold",
    "price_below": "Price ≤ threshold",
    "rsi_above":   "RSI(14) ≥ threshold",
    "rsi_below":   "RSI(14) ≤ threshold",
}


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_alerts_table() -> None:
    """Create the alerts table if it doesn't exist. Safe to call on every startup."""
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker       TEXT    NOT NULL,
                alert_type   TEXT    NOT NULL,
                threshold    REAL    NOT NULL,
                enabled      INTEGER NOT NULL DEFAULT 1,
                created_at   REAL    NOT NULL,
                triggered_at REAL
            )
        """)
    conn.close()


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def add_alert(ticker: str, alert_type: str, threshold: float) -> int:
    """
    Insert a new enabled alert and return its id.

    Parameters
    ----------
    ticker     : e.g. "AAPL"
    alert_type : one of price_above | price_below | rsi_above | rsi_below
    threshold  : numeric trigger level

    Raises
    ------
    ValueError if alert_type is not recognised.
    """
    ticker     = ticker.upper().strip()
    alert_type = alert_type.lower().strip()
    threshold  = float(threshold)

    if alert_type not in ALERT_TYPES:
        raise ValueError(
            f"Unknown alert_type '{alert_type}'. "
            f"Valid types: {list(ALERT_TYPES.keys())}"
        )

    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (ticker, alert_type, threshold, enabled, created_at)
                VALUES (?, ?, ?, 1, ?)
                """,
                (ticker, alert_type, threshold, time.time()),
            )
            alert_id = cur.lastrowid
        logger.info("Alert %d added: %s %s %.4f", alert_id, ticker, alert_type, threshold)
        return alert_id
    finally:
        conn.close()


def get_alerts() -> pd.DataFrame:
    """
    Return all alerts as a DataFrame, newest first.

    Columns: ID | Ticker | Type | Threshold | Enabled | Created | Last Triggered
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "ID", "Ticker", "Type", "Threshold", "Enabled", "Created", "Last Triggered",
        ])

    records = []
    for row in rows:
        records.append({
            "ID":             row["id"],
            "Ticker":         row["ticker"],
            "Type":           row["alert_type"],
            "Threshold":      row["threshold"],
            "Enabled":        bool(row["enabled"]),
            "Created":        pd.Timestamp(row["created_at"], unit="s").strftime("%Y-%m-%d %H:%M"),
            "Last Triggered": (
                pd.Timestamp(row["triggered_at"], unit="s").strftime("%Y-%m-%d %H:%M")
                if row["triggered_at"] else "—"
            ),
        })
    df = pd.DataFrame(records)
    df["Enabled"] = df["Enabled"].astype(object)
    return df


def delete_alert(alert_id: int) -> None:
    """Permanently remove an alert by id."""
    conn = get_connection()
    try:
        with conn:
            conn.execute("DELETE FROM alerts WHERE id=?", (alert_id,))
        logger.info("Alert %d deleted.", alert_id)
    finally:
        conn.close()


def toggle_alert(alert_id: int, enabled: bool) -> None:
    """Enable or disable an alert without deleting it."""
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                "UPDATE alerts SET enabled=? WHERE id=?",
                (1 if enabled else 0, alert_id),
            )
    finally:
        conn.close()




# ── Desktop notification (best-effort) ────────────────────────────────────────

def _notify(title: str, message: str) -> None:
    """Fire a native desktop notification. Silently swallows any plyer error."""
    try:
        from plyer import notification
        notification.notify(
            title=title,
            message=message,
            app_name="Quant Platform",
            timeout=8,
        )
    except Exception as exc:
        logger.debug("Desktop notification skipped: %s", exc)


# ── Core evaluation engine ────────────────────────────────────────────────────

def check_alerts(current_data: dict[str, dict[str, Any]]) -> list[dict]:
    """
    Evaluate all enabled alerts against *current_data* and return a list of
    triggered alert dicts.  Also stamps triggered_at on firing alerts.

    Parameters
    ----------
    current_data : dict keyed by ticker (uppercase).
        Each value is a dict with at least:
            price : float   latest close / last trade price
            rsi   : float | None   RSI(14) value (may be None)

    Returns
    -------
    list of dicts, one per triggered alert:
        {id, ticker, alert_type, threshold, price, rsi, message}
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id=str(uuid.uuid4())[:8], component="scheduler")

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE enabled=1"
        ).fetchall()
    finally:
        conn.close()

    triggered: list[dict] = []
    now = time.time()

    for row in rows:
        ticker     = row["ticker"]
        alert_type = row["alert_type"]
        threshold  = float(row["threshold"])
        alert_id   = row["id"]

        data = current_data.get(ticker)
        if data is None:
            continue

        price = data.get("price")
        rsi   = data.get("rsi")

        fired = False
        if alert_type == "price_above" and price is not None and price >= threshold:
            fired = True
        elif alert_type == "price_below" and price is not None and price <= threshold:
            fired = True
        elif alert_type == "rsi_above" and rsi is not None and rsi >= threshold:
            fired = True
        elif alert_type == "rsi_below" and rsi is not None and rsi <= threshold:
            fired = True

        if fired:
            _label = ALERT_TYPES.get(alert_type, alert_type)
            msg = (
                f"{ticker}: {_label} {threshold:.2f} triggered — "
                f"price=${price:.2f}" +
                (f", RSI={rsi:.1f}" if rsi is not None else "")
            )
            logger.info("ALERT FIRED: %s", msg)
            _notify(f"Alert: {ticker}", msg)
            broadcast(subject=f"Alert: {ticker}", body=msg)

            # Stamp triggered_at
            conn2 = get_connection()
            try:
                with conn2:
                    conn2.execute(
                        "UPDATE alerts SET triggered_at=? WHERE id=?",
                        (now, alert_id),
                    )
            finally:
                conn2.close()

            triggered.append({
                "id":         alert_id,
                "ticker":     ticker,
                "alert_type": alert_type,
                "threshold":  threshold,
                "price":      price,
                "rsi":        rsi,
                "message":    msg,
                "fired_at":   pd.Timestamp(now, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
            })

    return triggered


# ── Daily VaR check ───────────────────────────────────────────────────────────

def run_var_check(var_threshold: float = 0.03) -> dict | None:
    """
    Read recent portfolio values from the DB (last 252 trading days) and
    compute risk metrics.  If VaR 95% exceeds *var_threshold*, fire an alert
    via the broadcast channel and return the triggered result dict.

    Parameters
    ----------
    var_threshold : float
        Fractional daily VaR threshold (default 0.03 = 3 %).

    Returns
    -------
    dict with risk metrics if the threshold was breached, else None.
    """
    from analysis.risk_metrics import compute_risk_metrics

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT total_value
            FROM   paper_portfolio_snapshots
            ORDER  BY recorded_at DESC
            LIMIT  252
            """
        ).fetchall()
    except Exception as exc:
        logger.debug("VaR check: could not read snapshots (%s), falling back.", exc)
        rows = []
    finally:
        conn.close()

    if not rows:
        logger.info("VaR check: no portfolio snapshot data available.")
        return None

    # Snapshots are newest-first; reverse so oldest-first for the calc.
    values = [float(r[0]) for r in reversed(rows)]
    rm = compute_risk_metrics(values)
    if rm is None:
        logger.info("VaR check: insufficient data (%d values).", len(values))
        return None

    logger.info(
        "VaR check: VaR95=%.4f VaR99=%.4f CVaR95=%.4f ann_vol=%.4f n=%d",
        rm.var_95, rm.var_99, rm.cvar_95, rm.volatility_annual, rm.n_observations,
    )

    if rm.var_95 > var_threshold:
        msg = (
            f"Portfolio VaR alert: 95% daily VaR = {rm.var_95 * 100:.2f}% "
            f"exceeds threshold of {var_threshold * 100:.2f}%. "
            f"Ann. volatility = {rm.volatility_annual:.2f}%, "
            f"CVaR 95% = {rm.cvar_95 * 100:.2f}%."
        )
        logger.warning("VAR ALERT: %s", msg)
        _notify("VaR Alert", msg)
        broadcast(subject="VaR Alert: threshold breached", body=msg)
        return {
            "var_95":            rm.var_95,
            "var_99":            rm.var_99,
            "cvar_95":           rm.cvar_95,
            "cvar_99":           rm.cvar_99,
            "volatility_annual": rm.volatility_annual,
            "n_observations":    rm.n_observations,
            "message":           msg,
        }

    return None


# ── Daily correlation & concentration monitor ─────────────────────────────────

def run_correlation_check(
    price_data: "dict[str, pd.Series] | None" = None,
    positions: "dict[str, float] | None" = None,
    sector_map: "dict[str, str] | None" = None,
) -> list[dict]:
    """
    Run the correlation and concentration monitor and broadcast any alerts.

    Parameters
    ----------
    price_data  : {ticker: price Series} — fetched from data/fetcher if None
    positions   : {ticker: market_value} — read from paper_trader if None
    sector_map  : optional {ticker: sector} for sector concentration checks

    Returns
    -------
    List of alert dicts that were fired.
    """
    import os
    if os.environ.get("CORRELATION_MONITOR_ENABLED", "0") != "1":
        return []

    from risk.correlation import check_correlation_alerts

    # Fetch live portfolio if not provided
    if positions is None:
        try:
            from broker.paper_trader import get_portfolio
            port_df = get_portfolio()
            if not port_df.empty and "Market Value" in port_df.columns:
                positions = {
                    str(row["Ticker"]): float(row["Market Value"])
                    for _, row in port_df.iterrows()
                    if row.get("Market Value") is not None
                }
            else:
                positions = {}
        except Exception as exc:
            logger.warning("correlation_check: portfolio fetch failed: %s", exc)
            positions = {}

    # Fetch price history if not provided
    if price_data is None and positions:
        try:
            from data.fetcher import fetch_ohlcv
            price_data = {}
            for ticker in positions:
                df = fetch_ohlcv(ticker, period="3mo")
                if not df.empty and "Close" in df.columns:
                    price_data[ticker] = df["Close"]
        except Exception as exc:
            logger.warning("correlation_check: price fetch failed: %s", exc)
            price_data = {}

    if not positions:
        logger.info("correlation_check: no positions to check")
        return []

    alerts = check_correlation_alerts(
        price_data=price_data or {},
        positions=positions,
        sector_map=sector_map,
    )

    fired: list[dict] = []
    for alert in alerts:
        msg = alert.message
        logger.warning("CORRELATION ALERT [%s]: %s", alert.alert_type, msg)
        broadcast(subject=f"Portfolio Alert: {alert.alert_type}", body=msg)
        fired.append({
            "alert_type": alert.alert_type,
            "value": alert.value,
            "threshold": alert.threshold,
            "message": msg,
        })

    if not fired:
        logger.info("correlation_check: all thresholds clear")

    return fired


# ── Anomaly detection scheduler hook ─────────────────────────────────────────

def run_anomaly_checks(
    watchlist: "list[str] | None" = None,
    current_prices: "dict[str, float] | None" = None,
) -> list[dict]:
    """
    Run all anomaly detection checks and broadcast results.

    Parameters
    ----------
    watchlist       : tickers to check — reads from DB watchlist if None
    current_prices  : current prices dict — fetched from market data if None

    Returns
    -------
    List of anomaly dicts that were detected.
    """
    from analysis.anomaly_detector import AnomalyDetector

    # Load watchlist
    if watchlist is None:
        try:
            from data.watchlist import get_watchlist
            watchlist = get_watchlist()
        except Exception:
            watchlist = []

    # Load current prices
    if current_prices is None and watchlist:
        try:
            from data.fetcher import fetch_ohlcv
            current_prices = {}
            for ticker in watchlist:
                df = fetch_ohlcv(ticker, period="5d")
                if not df.empty and "Close" in df.columns:
                    current_prices[ticker] = float(df["Close"].iloc[-1])
        except Exception as exc:
            logger.warning("anomaly_checks: price fetch failed: %s", exc)
            current_prices = {}

    detector = AnomalyDetector()
    results = detector.run_all_checks(
        watchlist=watchlist or [],
        current_prices=current_prices or {},
    )

    fired: list[dict] = []
    for anomaly in results:
        msg = anomaly.get("message", str(anomaly))
        logger.warning("ANOMALY DETECTED [%s]: %s", anomaly.get("type", "?"), msg)
        broadcast(subject=f"Anomaly: {anomaly.get('type', 'unknown')}", body=msg)
        fired.append(anomaly)

    return fired
