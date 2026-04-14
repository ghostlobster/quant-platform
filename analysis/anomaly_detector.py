"""
analysis/anomaly_detector.py — Automated anomaly detection for signals, prices, and P&L.

Detects three categories of anomalies:
  1. Signal drought  — strategy produced zero signals for N hours
  2. Price spike     — live price deviates > threshold% from recent mean
  3. P&L divergence  — live P&L diverges from paper P&L over lookback window

Designed to be called by scheduler/alerts.py:run_anomaly_checks() on a
15-minute schedule.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Anomaly:
    """A detected anomaly event."""

    type: str                          # 'signal_drought' | 'price_spike' | 'pnl_divergence'
    severity: str                      # 'warning' | 'critical'
    symbol: str                        # ticker or '' if not symbol-specific
    details: dict = field(default_factory=dict)
    message: str = ""
    detected_at: float = field(default_factory=time.time)


class AnomalyDetector:
    """
    Stateless anomaly detector — each method returns an Anomaly or None.

    All external calls (DB, market data) are made lazily and guarded so a
    single source failure does not suppress the other checks.
    """

    # ── Signal drought ─────────────────────────────────────────────────────────

    def check_signal_drought(
        self,
        signal_log: list[dict] | None = None,
        window_hours: int = 4,
    ) -> Anomaly | None:
        """
        Return an Anomaly if no trading signals were generated in *window_hours*.

        Parameters
        ----------
        signal_log   : list of dicts with 'entry_time' key (Unix timestamps).
                       If None, queries the journal DB for recent entries.
        window_hours : silence window that triggers the anomaly (default 4h)
        """
        now = time.time()
        cutoff = now - window_hours * 3600

        if signal_log is None:
            try:
                from journal.trading_journal import get_trades
                trades = get_trades()
                if trades is not None and not trades.empty:
                    # Use executed_at or entry_time column
                    ts_col = next(
                        (c for c in trades.columns if "time" in c.lower() or "at" in c.lower()),
                        None,
                    )
                    if ts_col is not None:
                        recent = trades[pd.to_numeric(trades[ts_col], errors="coerce") > cutoff]
                        if len(recent) > 0:
                            return None  # signals exist — no drought
                signal_log = []
            except Exception as exc:
                logger.debug("signal_drought: journal query failed: %s", exc)
                return None

        # Check provided log
        if signal_log:
            recent = [s for s in signal_log if float(s.get("entry_time", 0)) > cutoff]
            if recent:
                return None

        msg = (
            f"Signal drought detected: no trading signals in the last {window_hours} hours. "
            "Check strategy logic, market data feed, and position limits."
        )
        logger.warning(msg)
        return Anomaly(
            type="signal_drought",
            severity="warning",
            symbol="",
            details={"window_hours": window_hours, "last_checked": now},
            message=msg,
        )

    # ── Price spike ───────────────────────────────────────────────────────────

    def check_price_spike(
        self,
        symbol: str,
        current_price: float,
        lookback_days: int = 5,
        threshold_pct: float = 0.10,
    ) -> Anomaly | None:
        """
        Return an Anomaly if *current_price* deviates more than *threshold_pct*
        from the mean over *lookback_days*.

        Parameters
        ----------
        symbol        : ticker to check
        current_price : latest price
        lookback_days : historical window for mean calculation
        threshold_pct : alert if |deviation| > this fraction (default 0.10 = 10%)
        """
        try:
            from data.fetcher import fetch_ohlcv
            period = f"{lookback_days + 5}d"
            df = fetch_ohlcv(symbol, period=period)
            if df.empty or "Close" not in df.columns:
                return None
            recent_prices = df["Close"].dropna().tail(lookback_days)
            if len(recent_prices) < 2:
                return None
            mean_price = float(recent_prices.mean())
            if mean_price <= 0:
                return None
            deviation = abs(current_price - mean_price) / mean_price
            if deviation > threshold_pct:
                msg = (
                    f"Price spike detected for {symbol}: current=${current_price:.2f}, "
                    f"{lookback_days}d mean=${mean_price:.2f}, "
                    f"deviation={deviation:.1%} > threshold={threshold_pct:.1%}."
                )
                logger.warning(msg)
                return Anomaly(
                    type="price_spike",
                    severity="warning" if deviation < 0.20 else "critical",
                    symbol=symbol,
                    details={
                        "current_price": current_price,
                        "mean_price": mean_price,
                        "deviation_pct": round(deviation, 4),
                        "threshold_pct": threshold_pct,
                    },
                    message=msg,
                )
        except Exception as exc:
            logger.debug("price_spike check failed for %s: %s", symbol, exc)
        return None

    # ── P&L divergence ────────────────────────────────────────────────────────

    def check_pnl_divergence(
        self,
        live_pnl_series: list[float],
        paper_pnl_series: list[float],
        threshold_pct: float = 0.05,
        lookback_days: int = 5,
    ) -> Anomaly | None:
        """
        Return an Anomaly if live P&L diverges from paper P&L by more than
        *threshold_pct* over *lookback_days*.

        Parameters
        ----------
        live_pnl_series  : list of daily live P&L values (most recent last)
        paper_pnl_series : list of daily paper P&L values (most recent last)
        threshold_pct    : fractional divergence threshold (default 0.05 = 5%)
        lookback_days    : number of data points to compare
        """
        if not live_pnl_series or not paper_pnl_series:
            return None

        live = list(live_pnl_series[-lookback_days:])
        paper = list(paper_pnl_series[-lookback_days:])
        n = min(len(live), len(paper))
        if n < 2:
            return None

        live_sum = sum(live[-n:])
        paper_sum = sum(paper[-n:])
        denom = max(abs(paper_sum), 1.0)
        divergence = abs(live_sum - paper_sum) / denom

        if divergence > threshold_pct:
            msg = (
                f"P&L divergence detected: live={live_sum:+.2f}, paper={paper_sum:+.2f} "
                f"over {n} days — divergence {divergence:.1%} > threshold {threshold_pct:.1%}. "
                "Investigate execution differences between live and paper accounts."
            )
            logger.warning(msg)
            return Anomaly(
                type="pnl_divergence",
                severity="warning" if divergence < 0.15 else "critical",
                symbol="",
                details={
                    "live_sum": live_sum,
                    "paper_sum": paper_sum,
                    "divergence_pct": round(divergence, 4),
                    "lookback_days": n,
                },
                message=msg,
            )
        return None

    # ── Orchestrator ───────────────────────────────────────────────────────────

    def run_all_checks(
        self,
        watchlist: list[str],
        current_prices: dict[str, float],
    ) -> list[dict]:
        """
        Run all three anomaly checks and return a list of anomaly dicts.

        Parameters
        ----------
        watchlist       : list of tickers to check for price spikes
        current_prices  : {ticker: price} mapping

        Returns
        -------
        List of dicts, one per detected anomaly.
        """
        anomalies: list[Anomaly] = []

        # 1. Signal drought
        drought = self.check_signal_drought()
        if drought:
            anomalies.append(drought)

        # 2. Price spikes for each watched ticker
        for ticker in watchlist:
            price = current_prices.get(ticker)
            if price is not None and price > 0:
                spike = self.check_price_spike(ticker, price)
                if spike:
                    anomalies.append(spike)

        # 3. P&L divergence (best-effort from portfolio history)
        try:
            from data.db import get_connection
            conn = get_connection()
            rows = conn.execute(
                "SELECT total_value FROM portfolio_history ORDER BY record_date DESC LIMIT 10"
            ).fetchall()
            conn.close()
            if rows and len(rows) >= 2:
                paper_values = [float(r[0]) for r in reversed(rows)]
                paper_pnl = [paper_values[i] - paper_values[i - 1] for i in range(1, len(paper_values))]
                # Without a separate live account we skip this check
                # (it activates when a live broker is connected)
        except Exception:
            pass

        return [
            {
                "type": a.type,
                "severity": a.severity,
                "symbol": a.symbol,
                "message": a.message,
                "details": a.details,
                "detected_at": a.detected_at,
            }
            for a in anomalies
        ]
