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

import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from data.db import get_connection
from data.indicators import compute_rsi
from utils.logger import get_logger
from alerts.channels import broadcast

logger = get_logger(__name__)

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


# ── Portfolio VaR limit check ─────────────────────────────────────────────────

def check_portfolio_var_limit() -> Optional[dict]:
    """
    Check whether 1-day VaR (95%) exceeds the configured limit.

    Reads PORTFOLIO_VAR_LIMIT env var (default 0.05 = 5%).
    Builds NAV history from paper-trade records and computes risk metrics.
    If var_95 > limit, broadcasts an alert and returns a dict with details.
    Returns None if there is insufficient history or VaR is within limits.
    """
    from broker.paper_trader import get_trade_history, STARTING_CASH
    from analysis.risk_metrics import compute_risk_metrics

    var_limit = float(os.getenv("PORTFOLIO_VAR_LIMIT", "0.05"))

    trade_hist = get_trade_history()
    nav_values: list[float] = []
    if not trade_hist.empty:
        chrono = trade_hist.iloc[::-1].reset_index(drop=True)
        running = STARTING_CASH
        nav_values = [running]
        for _, row in chrono.iterrows():
            pnl = row.get("Realised P&L")
            if pnl is not None and pnl == pnl:  # not NaN
                running += float(pnl)
                nav_values.append(running)

    metrics = compute_risk_metrics(nav_values) if len(nav_values) >= 10 else None
    if metrics is None:
        logger.debug("check_portfolio_var_limit: insufficient history (%d obs)", len(nav_values))
        return None

    if metrics.var_95 > var_limit:
        msg = (
            f"⚠️ Portfolio VaR Alert: 1-day VaR at 95% confidence is "
            f"{metrics.var_95 * 100:.1f}% — exceeds limit of {var_limit * 100:.1f}%"
        )
        logger.warning(msg)
        broadcast(subject="Portfolio VaR Alert", body=msg)
        return {
            "var_95":     metrics.var_95,
            "var_limit":  var_limit,
            "message":    msg,
        }

    logger.info(
        "check_portfolio_var_limit: VaR %.2f%% within limit %.2f%%",
        metrics.var_95 * 100,
        var_limit * 100,
    )
    return None
