"""
data/db.py — SQLite connection and schema helpers.

Single source of truth for all DB access. All tables are created here on
first use (CREATE TABLE IF NOT EXISTS), so no migration scripts are needed
for a personal-use local app.
"""
import sqlite3
from pathlib import Path

# Database lives at project root — excluded from git via .gitignore
_DB_PATH = Path(__file__).parent.parent / "quant.db"


def get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection with WAL mode for better concurrency."""
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row  # rows accessible by column name
    return conn


def init_db() -> None:
    """Create all tables if they don't exist yet. Safe to call on every startup."""
    conn = get_connection()
    with conn:
        # ----- Price cache -----
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_cache (
                ticker      TEXT    NOT NULL,
                period      TEXT    NOT NULL,
                fetched_at  REAL    NOT NULL,   -- Unix timestamp (seconds)
                data_json   TEXT    NOT NULL,   -- JSON-serialised DataFrame
                PRIMARY KEY (ticker, period)
            )
        """)

        # ----- Watchlist -----
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                ticker      TEXT    PRIMARY KEY,
                added_at    REAL    NOT NULL    -- Unix timestamp
            )
        """)

        # ----- Paper trading -----
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
                opened_at     REAL    NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                executed_at   REAL    NOT NULL,
                ticker        TEXT    NOT NULL,
                action        TEXT    NOT NULL,
                shares        REAL    NOT NULL,
                price         REAL    NOT NULL,
                cost_basis    REAL,
                realised_pnl  REAL
            )
        """)

        # ----- Portfolio history -----
        conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                record_date TEXT    NOT NULL UNIQUE,
                total_value REAL    NOT NULL
            )
        """)

        # Seed paper_account if absent
        import os
        starting_cash = float(os.getenv("PAPER_STARTING_CASH", "100000"))
        existing = conn.execute("SELECT id FROM paper_account WHERE id=1").fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO paper_account (id, cash_balance, realised_pnl) VALUES (1, ?, 0)",
                (starting_cash,),
            )

    conn.close()
