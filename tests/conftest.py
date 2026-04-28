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


# ── Failure-injection fixtures (closes part of #221) ────────────────────────
# Used by Phase-2 e2e files to exercise the unhappy paths — broker down,
# journal write fails, kill-switch tripped mid-flow. Each fixture is a
# factory: the test body calls it with arguments to install the failure.


@pytest.fixture
def inject_broker_failure(monkeypatch):
    """Return a factory that makes ``adapter.place_order`` raise after
    ``after_n`` successful calls.

    Usage::

        adapter = PaperBrokerAdapter()
        inject_broker_failure(adapter, after_n=2, reason="broker offline")
        adapter.place_order(...)   # ok
        adapter.place_order(...)   # ok
        adapter.place_order(...)   # raises RuntimeError("broker offline")
    """
    def _factory(adapter, after_n: int = 0, reason: str = "broker failure"):
        original = adapter.place_order
        state = {"calls": 0}

        def _wrapped(*args, **kwargs):
            if state["calls"] >= after_n:
                raise RuntimeError(reason)
            state["calls"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(adapter, "place_order", _wrapped)

    return _factory


@pytest.fixture
def inject_journal_failure(monkeypatch):
    """Return a factory that makes ``journal.trading_journal.log_entry``
    raise on the next call.

    Usage::

        inject_journal_failure(reason="disk full")
        # next call to jt.log_entry raises RuntimeError("disk full")
    """
    def _factory(reason: str = "journal write failed"):
        import journal.trading_journal as jt

        def _raise(*args, **kwargs):
            raise RuntimeError(reason)

        monkeypatch.setattr(jt, "log_entry", _raise)

    return _factory


@pytest.fixture
def trip_killswitch(tmp_path, monkeypatch):
    """Return a factory that touches the kill-switch flag file mid-test.

    The flag location is pointed at a tmp file; the calling test can
    check ``os.path.exists(flag)`` after invoking ``trip_killswitch()``
    and verify the broker rejection path.
    """
    flag = tmp_path / ".killswitch"
    monkeypatch.setenv("KILLSWITCH_PATH", str(flag))

    def _factory():
        flag.touch()
        return flag

    yield _factory
    if flag.exists():
        flag.unlink()


# ── Cleanup-invariant autouse for the e2e suite (closes part of #221) ───────


def _journal_count() -> int:
    """Return the number of rows in journal_trades for the current
    JOURNAL_DB_PATH, or -1 if the journal hasn't been initialised."""
    import sqlite3

    path = os.environ.get("JOURNAL_DB_PATH")
    if not path or not os.path.exists(path):
        return -1
    try:
        conn = sqlite3.connect(path)
        cur = conn.execute("SELECT COUNT(*) FROM journal_trades")
        n = int(cur.fetchone()[0])
        conn.close()
        return n
    except sqlite3.Error:
        return -1


def _paper_trade_count() -> int:
    """Return the number of rows in paper_trades for the active DB."""
    import sqlite3

    try:
        import data.db as db_module
    except ImportError:
        return -1
    path = getattr(db_module, "_DB_PATH", None)
    if not path or not os.path.exists(path):
        return -1
    try:
        conn = sqlite3.connect(path)
        cur = conn.execute("SELECT COUNT(*) FROM paper_trades")
        n = int(cur.fetchone()[0])
        conn.close()
        return n
    except sqlite3.Error:
        return -1


@pytest.fixture(autouse=True)
def e2e_cleanup_invariant(request):
    """Cleanup-invariant gate for every test marked ``@pytest.mark.e2e``.

    Snapshot live-thread count + paper_trade / journal counts at fixture
    setup; after the test, assert:

      * Number of paper_trades rows can grow but every new fill must be
        accompanied by a journal_trades row (within a small tolerance —
        guard rejections legitimately don't journal).
      * No orphan threads vs the snapshot.

    The invariant is **opt-out**: a test that intentionally violates
    the journal-per-fill rule (e.g. the journal-write-failure test
    itself) can mark itself ``@pytest.mark.e2e_skip_invariant`` and
    the autouse fixture becomes a no-op for that case.

    Non-e2e tests are not affected — the fixture short-circuits unless
    the test carries the ``e2e`` marker.
    """
    if request.node.get_closest_marker("e2e") is None:
        yield
        return
    if request.node.get_closest_marker("e2e_skip_invariant") is not None:
        yield
        return

    import threading

    threads_before = {t.ident for t in threading.enumerate()}
    paper_before = _paper_trade_count()
    journal_before = _journal_count()

    yield

    paper_after = _paper_trade_count()
    journal_after = _journal_count()
    threads_after = {t.ident for t in threading.enumerate()}

    # Thread-leak invariant — only assert when both snapshots succeeded.
    new_threads = threads_after - threads_before
    if new_threads:
        # Daemon threads from APScheduler / sklearn are noisy; only fail
        # when the test added a non-daemon thread that could block exit.
        leaked = [
            t for t in threading.enumerate()
            if t.ident in new_threads and not t.daemon
        ]
        assert not leaked, (
            f"e2e cleanup-invariant: leaked non-daemon threads: "
            f"{[t.name for t in leaked]}"
        )

    # Trade-vs-journal invariant — only when both DBs are reachable.
    if paper_before >= 0 and paper_after >= 0:
        paper_added = paper_after - paper_before
        if paper_added > 0 and journal_before >= 0 and journal_after >= 0:
            journal_added = journal_after - journal_before
            # Buys + sells produce 2 paper_trades per round trip; the
            # journal records 1 entry + 1 exit per round trip. So the
            # journal can be ≤ paper_added but not zero when paper grew.
            assert journal_added > 0 or paper_added == 0, (
                "e2e cleanup-invariant: paper_trades grew "
                f"by {paper_added} but journal_trades did not grow — "
                "every fill should be journaled. If the test "
                "deliberately exercises the no-journal path, mark it "
                "@pytest.mark.e2e_skip_invariant."
            )


def pytest_configure(config):
    """Register the ``e2e_skip_invariant`` marker so the cleanup-
    invariant fixture can opt-out cleanly under
    ``--strict-markers``."""
    config.addinivalue_line(
        "markers",
        "e2e_skip_invariant: opt out of the e2e cleanup-invariant "
        "fixture. Use only on tests that intentionally exercise the "
        "no-journal-on-fill path (e.g. journal-write-failure tests).",
    )
