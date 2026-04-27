"""
Pytest configuration for the quant-platform test suite.

Force single-threaded BLAS/OpenBLAS so background compute threads are not
still live when Python finalises after the test session.  Without this,
numpy's OpenBLAS worker threads can call std::terminate() during cleanup,
causing a SIGABRT (exit 134) on Linux CI runners even though all tests pass.

Also exposes shared e2e fixtures (``e2e_paper_env``, ``e2e_journal_db``,
``e2e_isolated_caches``, ``E2EFakeBroker``) so the five
``tests/test_e2e_*.py`` files do not redefine the same scaffolding.
The fixtures are name-prefixed ``e2e_`` so unit tests that reuse the
same names (``paper_env``, ``journal_db``) still resolve to their own
file-level fixtures and pay no fee for the shared ones.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import pytest

# ── Shared e2e fixtures ──────────────────────────────────────────────────────
# Each fixture isolates the chain it touches to a per-test tmp file so a
# parallel run (pytest-xdist) is safe.

class E2EFakeBroker:
    """Minimal BrokerProvider stand-in used by every e2e file that needs to
    drive ``compute_risk_snapshot`` / risk gauges without a real broker.

    ``equity`` mirrors Alpaca's native key (the live adapter passes it
    through unchanged); ``total_value`` is also populated so the same fake
    works for the paper-broker code path that #183 fixed.
    """

    def __init__(self, equity: float = 100_000.0, positions=None):
        self.equity = float(equity)
        self._positions = list(positions or [])

    def get_account_info(self) -> dict:
        return {
            "equity":      self.equity,
            "total_value": self.equity,
            "cash":        self.equity,
        }

    def get_positions(self) -> list[dict]:
        return list(self._positions)


@pytest.fixture
def e2e_paper_env(tmp_path, monkeypatch):
    """Isolate paper_trader + journal under per-test tmp SQLite files.

    Disables the legacy paper-trader circuit breaker so the e2e suite
    measures whatever guard / bracket logic is under test, not the
    historical drawdown floor.
    """
    import broker.paper_trader as pt
    import data.db as db_module
    import journal.trading_journal as jt

    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    monkeypatch.setattr(pt, "STARTING_CASH", 100_000.0)
    monkeypatch.setattr(pt, "MAX_DRAWDOWN_PCT", 0.99)
    pt.init_paper_tables()
    jt.init_journal_table()
    return tmp_path


@pytest.fixture
def e2e_journal_db(tmp_path, monkeypatch):
    """Journal-only isolation for tests that don't touch paper_trader."""
    import journal.trading_journal as jt

    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.db"))
    jt.init_journal_table()
    return tmp_path


@pytest.fixture
def e2e_isolated_caches(tmp_path, monkeypatch):
    """Per-test SQLite + DuckDB cache files.

    Resets the DuckDB connection singleton so each test owns its own
    file (the adapter caches the connection module-globally otherwise).
    """
    import data.db as db_module

    monkeypatch.setattr(db_module, "_DB_PATH", str(tmp_path / "quant.db"))
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "quant_tsdb.duckdb"))
    try:
        import adapters.tsdb.duckdb_adapter as dad

        dad._connection = None
    except ImportError:
        pass
    yield tmp_path
    try:
        import adapters.tsdb.duckdb_adapter as dad

        dad._connection = None
    except ImportError:
        pass
