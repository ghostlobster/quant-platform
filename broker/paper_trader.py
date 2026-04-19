"""
broker/paper_trader.py — SQLite-backed paper trading simulator.

Schema
------
  paper_account  : single row holding cash_balance
  paper_positions: open positions (one row per ticker, avg-cost method)
  paper_trades   : immutable ledger of every fill (buy or sell)

Public API
----------
  buy(ticker, shares, price)  → dict with fill details or raises ValueError
  sell(ticker, shares, price) → dict with fill details or raises ValueError
  get_portfolio()             → DataFrame of open positions + unrealised P&L
  get_trade_history()         → DataFrame of all closed/partial sells + realised P&L
  get_account()               → dict {cash, market_value, total_value, realised_pnl}
  reset_account()             → wipe all positions/trades, restore starting cash
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from adapters.execution_algo.result import ExecutionResult

import pandas as pd
from dotenv import load_dotenv

from data.db import get_connection
from utils.logger import get_logger

load_dotenv()

# ── Optional journal hook ─────────────────────────────────────────────────────
# Guard: if the journal module is unavailable the hook is silently disabled so
# paper trading continues to work regardless of journal DB state.
try:
    from journal.trading_journal import log_entry as _journal_log_entry
except Exception:  # ImportError or any import-time failure
    _journal_log_entry = None  # type: ignore[assignment]
logger = get_logger(__name__)

# Starting cash — override via PAPER_STARTING_CASH in .env
STARTING_CASH: float = float(os.getenv("PAPER_STARTING_CASH", "100000"))

# Max drawdown circuit breaker — override via MAX_DRAWDOWN_PCT in .env (e.g. "0.20" = 20 %)
MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "0.20"))


# ── Schema bootstrap ──────────────────────────────────────────────────────────

def init_paper_tables() -> None:
    """Create paper-trading tables if they don't exist. Safe to call repeatedly."""
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_account (
                id            INTEGER PRIMARY KEY CHECK (id = 1),
                cash_balance  REAL    NOT NULL,
                realised_pnl  REAL    NOT NULL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_positions (
                ticker        TEXT    PRIMARY KEY,
                shares        REAL    NOT NULL,
                avg_cost      REAL    NOT NULL,
                total_cost    REAL    NOT NULL,
                opened_at     REAL    NOT NULL   -- Unix timestamp of first buy
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                executed_at   REAL    NOT NULL,  -- Unix timestamp
                ticker        TEXT    NOT NULL,
                action        TEXT    NOT NULL,  -- 'BUY' | 'SELL'
                shares        REAL    NOT NULL,
                price         REAL    NOT NULL,
                cost_basis    REAL,              -- avg cost at time of sell (NULL for buys)
                realised_pnl  REAL               -- NULL for buys
            )
        """)
        # P1.3 — bracket / OCO / trailing-stop child orders. Parent fills
        # immediately via buy()/sell(); this table tracks the pending
        # take-profit / stop-loss / trailing-stop legs until check_brackets()
        # fires them.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_bracket_orders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_order_id TEXT    NOT NULL,
                ticker          TEXT    NOT NULL,
                parent_side     TEXT    NOT NULL,  -- BUY | SELL (the opened side)
                shares          REAL    NOT NULL,
                parent_price    REAL    NOT NULL,
                take_profit     REAL,
                stop_loss       REAL,
                trail_percent   REAL,
                peak_price      REAL,              -- running watermark for trailing stop
                status          TEXT    NOT NULL,  -- pending | filled | cancelled
                created_at      REAL    NOT NULL,
                closed_at       REAL,
                close_reason    TEXT               -- take_profit | stop_loss | trail | manual
            )
        """)

        # Seed account row if absent
        existing = conn.execute("SELECT id FROM paper_account WHERE id=1").fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO paper_account (id, cash_balance, realised_pnl) VALUES (1, ?, 0)",
                (STARTING_CASH,),
            )
    conn.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_cash(conn) -> float:
    row = conn.execute("SELECT cash_balance FROM paper_account WHERE id=1").fetchone()
    return float(row["cash_balance"])


def _get_position(conn, ticker: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM paper_positions WHERE ticker=?", (ticker.upper(),)
    ).fetchone()


# ── Public API ────────────────────────────────────────────────────────────────

def buy(ticker: str, shares: float, price: float) -> dict:
    """
    Execute a paper buy order.

    Parameters
    ----------
    ticker : str   Symbol (case-insensitive)
    shares : float Number of shares (must be > 0)
    price  : float Fill price per share (must be > 0)

    Returns
    -------
    dict with keys: ticker, action, shares, price, cost, cash_remaining

    Raises
    ------
    ValueError if shares/price invalid or insufficient cash.
    """
    ticker = ticker.upper().strip()
    shares = float(shares)
    price  = float(price)

    if math.isnan(shares) or math.isinf(shares) or shares <= 0:
        raise ValueError("Shares must be a finite positive number.")
    if math.isnan(price) or math.isinf(price) or price <= 0:
        raise ValueError("Price must be a finite positive number.")

    total_cost = round(shares * price, 4)

    conn = get_connection()
    try:
        with conn:
            cash = _get_cash(conn)

            # ── Circuit breaker: halt if drawdown from starting capital ≥ threshold ──
            positions_cost = conn.execute(
                "SELECT COALESCE(SUM(total_cost), 0) FROM paper_positions"
            ).fetchone()[0]
            current_equity = cash + positions_cost
            drawdown = (STARTING_CASH - current_equity) / STARTING_CASH
            if drawdown >= MAX_DRAWDOWN_PCT:
                raise RuntimeError(
                    f"Circuit breaker: max drawdown exceeded "
                    f"({drawdown:.1%} ≥ {MAX_DRAWDOWN_PCT:.1%}) — trading halted."
                )

            if total_cost > cash:
                raise ValueError(
                    f"Insufficient cash: need ${total_cost:,.2f} but have ${cash:,.2f}."
                )

            now = time.time()
            existing = _get_position(conn, ticker)

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO paper_positions (ticker, shares, avg_cost, total_cost, opened_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ticker, shares, price, total_cost, now),
                )
            else:
                new_shares     = existing["shares"] + shares
                new_total_cost = existing["total_cost"] + total_cost
                new_avg_cost   = new_total_cost / new_shares
                conn.execute(
                    """
                    UPDATE paper_positions
                    SET shares=?, avg_cost=?, total_cost=?
                    WHERE ticker=?
                    """,
                    (new_shares, new_avg_cost, new_total_cost, ticker),
                )

            # Debit cash
            conn.execute(
                "UPDATE paper_account SET cash_balance=cash_balance-? WHERE id=1",
                (total_cost,),
            )

            # Log trade
            conn.execute(
                """
                INSERT INTO paper_trades (executed_at, ticker, action, shares, price)
                VALUES (?, ?, 'BUY', ?, ?)
                """,
                (now, ticker, shares, price),
            )

            new_cash = cash - total_cost

        logger.info("BUY %s x%.4f @ $%.4f  cost=$%.2f  cash=$%.2f",
                    ticker, shares, price, total_cost, new_cash)
        if _journal_log_entry is not None:
            try:
                _journal_log_entry(ticker, "BUY", int(shares), price,
                                   signal_source="paper_trader")
            except Exception as _je:
                logger.warning("Journal hook failed (buy %s): %s", ticker, _je)
        return {
            "ticker":         ticker,
            "action":         "BUY",
            "shares":         shares,
            "price":          price,
            "cost":           total_cost,
            "cash_remaining": new_cash,
        }
    finally:
        conn.close()


def sell(ticker: str, shares: float, price: float) -> dict:
    """
    Execute a paper sell order.

    Parameters
    ----------
    ticker : str   Symbol (case-insensitive)
    shares : float Number of shares to sell (must be > 0 and ≤ held)
    price  : float Fill price per share (must be > 0)

    Returns
    -------
    dict with keys: ticker, action, shares, price, proceeds,
                    realised_pnl, cash_remaining

    Raises
    ------
    ValueError if shares/price invalid, no position, or selling more than held.
    """
    ticker  = ticker.upper().strip()
    shares  = float(shares)
    price   = float(price)

    if math.isnan(shares) or math.isinf(shares) or shares <= 0:
        raise ValueError("Shares must be a finite positive number.")
    if math.isnan(price) or math.isinf(price) or price <= 0:
        raise ValueError("Price must be a finite positive number.")

    conn = get_connection()
    try:
        with conn:
            existing = _get_position(conn, ticker)
            if existing is None:
                raise ValueError(f"No open position in {ticker}.")
            if shares > existing["shares"]:
                raise ValueError(
                    f"Cannot sell {shares} shares — only {existing['shares']:.4f} held."
                )

            proceeds      = round(shares * price, 4)
            avg_cost      = float(existing["avg_cost"])
            cost_basis    = round(shares * avg_cost, 4)
            realised_pnl  = round(proceeds - cost_basis, 4)
            remaining     = round(existing["shares"] - shares, 8)

            if remaining < 1e-6:
                conn.execute("DELETE FROM paper_positions WHERE ticker=?", (ticker,))
            else:
                new_total_cost = round(remaining * avg_cost, 4)
                conn.execute(
                    """
                    UPDATE paper_positions
                    SET shares=?, total_cost=?
                    WHERE ticker=?
                    """,
                    (remaining, new_total_cost, ticker),
                )

            # Credit cash and accumulate realised P&L
            conn.execute(
                "UPDATE paper_account SET cash_balance=cash_balance+?, realised_pnl=realised_pnl+? WHERE id=1",
                (proceeds, realised_pnl),
            )

            now = time.time()
            conn.execute(
                """
                INSERT INTO paper_trades
                    (executed_at, ticker, action, shares, price, cost_basis, realised_pnl)
                VALUES (?, ?, 'SELL', ?, ?, ?, ?)
                """,
                (now, ticker, shares, price, avg_cost, realised_pnl),
            )

            new_cash = _get_cash(conn)

        logger.info("SELL %s x%.4f @ $%.4f  pnl=$%.2f  cash=$%.2f",
                    ticker, shares, price, realised_pnl, new_cash)
        if _journal_log_entry is not None:
            try:
                _journal_log_entry(ticker, "SELL", int(shares), price,
                                   signal_source="paper_trader")
            except Exception as _je:
                logger.warning("Journal hook failed (sell %s): %s", ticker, _je)
        return {
            "ticker":         ticker,
            "action":         "SELL",
            "shares":         shares,
            "price":          price,
            "proceeds":       proceeds,
            "realised_pnl":   realised_pnl,
            "cash_remaining": new_cash,
        }
    finally:
        conn.close()


def get_portfolio(current_prices: Optional[dict[str, float]] = None) -> pd.DataFrame:
    """
    Return a DataFrame of all open positions enriched with current market prices.

    Parameters
    ----------
    current_prices : optional dict {ticker: price}.  If provided, used for
                     unrealised P&L calculation instead of fetching live prices.
                     Pass an empty dict to skip price enrichment.

    Columns
    -------
    Ticker | Shares | Avg Cost | Current Price | Market Value |
    Unrealised P&L | Unrealised % | Cost Basis
    """
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM paper_positions ORDER BY ticker").fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "Ticker", "Shares", "Avg Cost", "Current Price",
            "Market Value", "Unrealised P&L", "Unrealised %", "Cost Basis",
        ])

    records = []
    for row in rows:
        ticker     = row["ticker"]
        shares     = float(row["shares"])
        avg_cost   = float(row["avg_cost"])
        cost_basis = float(row["total_cost"])

        cur_price = None
        if current_prices is not None:
            raw_price = current_prices.get(ticker)
            if raw_price is not None:
                try:
                    p = float(raw_price)
                    if math.isnan(p) or math.isinf(p) or p <= 0:
                        raise ValueError(f"non-positive or non-finite price: {p}")
                    cur_price = p
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "get_portfolio: invalid price for %s (%s); skipping P&L enrichment",
                        ticker, exc,
                    )

        if cur_price is not None:
            market_val    = round(shares * cur_price, 2)
            unreal_pnl    = round(market_val - cost_basis, 2)
            unreal_pct    = round((unreal_pnl / cost_basis) * 100, 2) if cost_basis else 0.0
        else:
            market_val = cost_basis   # fallback: show cost as value
            unreal_pnl = None
            unreal_pct = None

        records.append({
            "Ticker":        ticker,
            "Shares":        shares,
            "Avg Cost":      round(avg_cost, 4),
            "Current Price": round(cur_price, 2) if cur_price is not None else None,
            "Market Value":  market_val,
            "Unrealised P&L": unreal_pnl,
            "Unrealised %":  unreal_pct,
            "Cost Basis":    round(cost_basis, 2),
        })

    return pd.DataFrame(records)


def get_trade_history() -> pd.DataFrame:
    """
    Return a DataFrame of all executed trades (buys and sells), newest first.

    Columns
    -------
    Date | Ticker | Action | Shares | Price | Proceeds/Cost | Realised P&L
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM paper_trades ORDER BY executed_at DESC"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return pd.DataFrame(columns=[
            "Date", "Ticker", "Action", "Shares", "Price",
            "Amount", "Avg Cost at Sale", "Realised P&L",
        ])

    records = []
    for row in rows:
        amount = round(row["shares"] * row["price"], 2)
        records.append({
            "Date":             pd.Timestamp(row["executed_at"], unit="s").strftime("%Y-%m-%d %H:%M"),
            "Ticker":           row["ticker"],
            "Action":           row["action"],
            "Shares":           row["shares"],
            "Price":            round(row["price"], 4),
            "Amount":           amount,
            "Avg Cost at Sale": round(row["cost_basis"], 4) if row["cost_basis"] else None,
            "Realised P&L":     round(row["realised_pnl"], 2) if row["realised_pnl"] is not None else None,
        })

    return pd.DataFrame(records)


def get_account() -> dict:
    """
    Return a summary dict of the paper account.

    Keys: cash, realised_pnl, market_value (sum of position cost bases as
    fallback — caller should pass current prices for accuracy), total_value
    """
    conn = get_connection()
    try:
        acc  = conn.execute("SELECT * FROM paper_account WHERE id=1").fetchone()
        mval = conn.execute("SELECT COALESCE(SUM(total_cost),0) AS mv FROM paper_positions").fetchone()
    finally:
        conn.close()

    cash          = float(acc["cash_balance"])
    realised_pnl  = float(acc["realised_pnl"])
    market_value  = float(mval["mv"])    # cost-basis fallback
    total_value   = cash + market_value

    return {
        "cash":          cash,
        "realised_pnl":  realised_pnl,
        "market_value":  market_value,
        "total_value":   total_value,
        "starting_cash": STARTING_CASH,
    }


def reset_account() -> None:
    """
    Wipe all positions and trade history and restore the starting cash balance.
    Intended for development/testing only.
    """
    conn = get_connection()
    with conn:
        conn.execute("DELETE FROM paper_positions")
        conn.execute("DELETE FROM paper_trades")
        conn.execute(
            "UPDATE paper_account SET cash_balance=?, realised_pnl=0 WHERE id=1",
            (STARTING_CASH,),
        )
    conn.close()
    logger.warning("Paper account reset to $%.2f starting cash.", STARTING_CASH)


def execute_algo(
    ticker: str,
    qty: float,
    side: str,
    algo: Optional[str] = None,
    decision_price: float = 0.0,
) -> "ExecutionResult":
    """
    Execute an order through the configured execution algorithm.

    This is the single integration point between execution algos and the paper
    trader.  It calls `get_execution_algo()` to select the algorithm, executes
    via PaperBrokerAdapter, and logs the result to `execution_analytics`.

    Parameters
    ----------
    ticker         : ticker symbol
    qty            : number of shares (must be > 0)
    side           : 'buy' or 'sell'
    algo           : override EXECUTION_ALGO env var; None uses env default
    decision_price : price at time of trading decision (for slippage calculation)

    Returns
    -------
    ExecutionResult with fill details and slippage metrics.
    """
    from adapters.broker.paper_adapter import PaperBrokerAdapter
    from providers.execution_algo import get_execution_algo

    execution_algo_inst = get_execution_algo(algo)
    broker = PaperBrokerAdapter()

    # If a decision price is provided, patch place_order to use it so fills
    # happen at the same price (avoids an extra live-price lookup in paper mode).
    if decision_price > 0:
        _orig_place_order = broker.place_order

        def _place_at_decision(symbol, qty, side, order_type="market", limit_price=None):
            return _orig_place_order(symbol, qty, side, order_type, limit_price=decision_price)

        broker.place_order = _place_at_decision  # type: ignore[method-assign]

    result: ExecutionResult = execution_algo_inst.execute(
        symbol=ticker,
        total_qty=qty,
        side=side,
        broker=broker,
        decision_price=decision_price,
        duration_minutes=1,  # paper trades execute immediately
    )

    # Log to execution_analytics table
    try:
        conn = get_connection()
        with conn:
            conn.execute(
                """
                INSERT INTO execution_analytics
                    (executed_at, symbol, side, algo, total_qty,
                     decision_price, avg_fill_price, slippage_bps, broker)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'paper')
                """,
                (
                    result.executed_at,
                    result.symbol,
                    result.side,
                    result.algo,
                    result.total_qty,
                    result.decision_price,
                    result.avg_fill_price,
                    result.slippage_bps,
                ),
            )
        conn.close()
    except Exception as _log_exc:
        logger.warning("Failed to log execution analytics: %s", _log_exc)

    return result


# ── P1.3 — bracket / OCO / trailing-stop simulation ──────────────────────────

def place_bracket(
    ticker: str,
    shares: float,
    side: str,
    entry_price: float,
    take_profit: Optional[float] = None,
    stop_loss: Optional[float] = None,
    trail_percent: Optional[float] = None,
) -> dict:
    """Fill a bracket-order parent and record the pending children.

    The parent market leg is executed immediately via ``buy``/``sell`` so
    the simulator behaves identically to a live bracket that fills
    instantly. Children (TP / SL / trailing stop) stay pending in
    ``paper_bracket_orders`` until :func:`check_brackets` evaluates them
    against a current-price feed.
    """
    side_norm = side.lower().strip()
    if side_norm not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    if take_profit is None and stop_loss is None and trail_percent is None:
        raise ValueError(
            "place_bracket requires at least one of take_profit, "
            "stop_loss, trail_percent"
        )

    # Fill the parent leg.
    if side_norm == "buy":
        parent = buy(ticker, shares, entry_price)
    else:
        parent = sell(ticker, shares, entry_price)

    parent_order_id = f"paper-bracket-{int(time.time() * 1000)}"
    now = time.time()

    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO paper_bracket_orders (
                    parent_order_id, ticker, parent_side, shares,
                    parent_price, take_profit, stop_loss, trail_percent,
                    peak_price, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    parent_order_id,
                    ticker.upper().strip(),
                    side_norm.upper(),
                    float(shares),
                    float(entry_price),
                    take_profit,
                    stop_loss,
                    trail_percent,
                    float(entry_price),
                    now,
                ),
            )
            bracket_id = cur.lastrowid
    finally:
        conn.close()

    logger.info(
        "Paper bracket opened id=%d parent=%s side=%s tp=%s sl=%s trail=%s",
        bracket_id, parent_order_id, side_norm, take_profit, stop_loss, trail_percent,
    )

    return {
        "order_id": parent_order_id,
        "bracket_id": bracket_id,
        "ticker": ticker.upper().strip(),
        "side": side_norm,
        "qty": float(shares),
        "status": "parent_filled",
        "parent_fill": parent,
        "children": {
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "trail_percent": trail_percent,
        },
    }


def _close_bracket(
    conn,
    row: sqlite3.Row,
    reason: str,
    fill_price: float,
) -> dict:
    """Internal: close an open bracket by flipping the parent side."""
    closing_side = "sell" if row["parent_side"].lower() == "buy" else "buy"
    if closing_side == "sell":
        fill = sell(row["ticker"], row["shares"], fill_price)
    else:
        fill = buy(row["ticker"], row["shares"], fill_price)

    conn.execute(
        """
        UPDATE paper_bracket_orders
        SET status='filled', closed_at=?, close_reason=?
        WHERE id=?
        """,
        (time.time(), reason, row["id"]),
    )
    return {
        "bracket_id": row["id"],
        "ticker": row["ticker"],
        "shares": row["shares"],
        "reason": reason,
        "fill_price": fill_price,
        "child_fill": fill,
    }


def check_brackets(current_prices: dict[str, float]) -> list[dict]:
    """Evaluate every pending bracket against ``current_prices``.

    Fires take-profit / stop-loss / trailing-stop children when triggered;
    updates ``peak_price`` on trailing brackets on every tick. Returns a
    list of filled child orders (empty when nothing fires).
    """
    conn = get_connection()
    fires: list[dict] = []
    try:
        rows = conn.execute(
            "SELECT * FROM paper_bracket_orders WHERE status='pending'"
        ).fetchall()
        for row in rows:
            ticker = row["ticker"]
            price = current_prices.get(ticker)
            if price is None:
                continue
            price = float(price)

            parent_side = row["parent_side"].lower()
            tp = row["take_profit"]
            sl = row["stop_loss"]
            trail = row["trail_percent"]
            peak = row["peak_price"] if row["peak_price"] is not None else row["parent_price"]

            if parent_side == "buy":
                # Long bracket — TP fires when price >= tp, SL / trail fire below.
                if tp is not None and price >= float(tp):
                    with conn:
                        fires.append(_close_bracket(conn, row, "take_profit", price))
                    continue
                if sl is not None and price <= float(sl):
                    with conn:
                        fires.append(_close_bracket(conn, row, "stop_loss", price))
                    continue
                if trail is not None:
                    new_peak = max(peak, price)
                    if new_peak != peak:
                        with conn:
                            conn.execute(
                                "UPDATE paper_bracket_orders SET peak_price=? WHERE id=?",
                                (new_peak, row["id"]),
                            )
                        peak = new_peak
                    trigger = peak * (1.0 - float(trail))
                    if price <= trigger:
                        with conn:
                            fires.append(_close_bracket(conn, row, "trail", price))
                        continue
            else:
                # Short bracket — TP fires when price <= tp, SL / trail above.
                if tp is not None and price <= float(tp):
                    with conn:
                        fires.append(_close_bracket(conn, row, "take_profit", price))
                    continue
                if sl is not None and price >= float(sl):
                    with conn:
                        fires.append(_close_bracket(conn, row, "stop_loss", price))
                    continue
                if trail is not None:
                    new_peak = min(peak, price)
                    if new_peak != peak:
                        with conn:
                            conn.execute(
                                "UPDATE paper_bracket_orders SET peak_price=? WHERE id=?",
                                (new_peak, row["id"]),
                            )
                        peak = new_peak
                    trigger = peak * (1.0 + float(trail))
                    if price >= trigger:
                        with conn:
                            fires.append(_close_bracket(conn, row, "trail", price))
                        continue
    finally:
        conn.close()
    return fires


def cancel_bracket(bracket_id: int) -> bool:
    """Mark a pending bracket cancelled. Returns True when a row was updated."""
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                UPDATE paper_bracket_orders
                SET status='cancelled', closed_at=?, close_reason='manual'
                WHERE id=? AND status='pending'
                """,
                (time.time(), bracket_id),
            )
            return cur.rowcount > 0
    finally:
        conn.close()


def get_pending_brackets() -> list[dict]:
    """Return every pending bracket row as a list of dicts."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM paper_bracket_orders WHERE status='pending' "
            "ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
