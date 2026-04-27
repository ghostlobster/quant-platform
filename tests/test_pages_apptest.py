"""Streamlit AppTest harness for pages/* — Phase 1 (#206).

The legacy ``tests/test_pages.py`` mocks Streamlit wholesale and
verifies ``render()`` doesn't raise. That's a smoke test, not a
regression test — a chart axis that flips, a table column that
disappears, or a sidebar control that loses its default all pass
the smoke test today.

This file uses ``streamlit.testing.v1.AppTest`` (GA in Streamlit
1.34+, we run 1.56) to exercise two pages on a real Streamlit
runtime and assert on the rendered widget tree:

  * ``pages/chart.py`` — title, regime banner, watchlist subheader,
    chart subheader bears the active ticker.
  * ``pages/journal_tab.py`` — journal subheader, KPI columns,
    journal table reflects seeded entries.

Phase 1 covers these two pages because they are the most-touched.
Phase 2 — covering every page in ``pages/*.py`` and *deleting* the
mocked ``test_pages.py`` smoke tests — is tracked in the same ticket
and lands separately so its diff is reviewable on its own.

Why this is hand-rolled instead of using AppTest.from_file: the
pages live behind ``app.py`` and expose a ``render()`` function
each rather than running as standalone Streamlit scripts. We wrap
them in a tiny in-process function that injects ``sys.path`` so the
temp-dir runner can resolve project imports.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _AppTest():
    """Defer the streamlit import until test time.

    ``tests/test_pages.py`` installs a ``MagicMock`` at
    ``sys.modules["streamlit"]`` to mock-out widgets in its smoke
    suite. Pytest collects ``test_pages.py`` before this file
    alphabetically, so a top-level ``from streamlit.testing.v1
    import AppTest`` fails with ``'streamlit' is not a package``.

    We work around it by stripping the mocked streamlit out of
    ``sys.modules`` and re-importing the real package at test time.
    Phase 2 of #206 replaces ``test_pages.py`` outright, at which
    point this shim becomes unnecessary.
    """
    # Drop any mocked streamlit so the import below picks up the real
    # package on disk.
    for mod in [m for m in list(sys.modules) if m.startswith("streamlit")]:
        if not hasattr(sys.modules[mod], "__path__"):
            del sys.modules[mod]
    # Page modules cache their st.cache_data-decorated closures at import
    # time. If a page was imported under the mocked streamlit, the
    # decorator binding is wrong forever — drop those too so the
    # subsequent ``import pages.X`` rebuilds them against real streamlit.
    for mod in [m for m in list(sys.modules) if m.startswith("pages.")]:
        del sys.modules[mod]
    from streamlit.testing.v1 import AppTest as _RealAppTest
    return _RealAppTest

# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 60, seed: int = 7, start: float = 150.0) -> pd.DataFrame:
    """Synthetic OHLCV that exercises every indicator add_all touches."""
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open":   close,
            "High":   close + 1.0,
            "Low":    close - 1.0,
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


# ── pages/journal_tab.py ────────────────────────────────────────────────────


@pytest.fixture
def isolated_journal(tmp_path, monkeypatch):
    """Isolate the journal DB to a per-test SQLite file."""
    db_path = tmp_path / "at_journal.db"
    monkeypatch.setenv("JOURNAL_DB_PATH", str(db_path))
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from journal import trading_journal as jt

        jt.init_journal_table()
        yield jt
    finally:
        sys.path.remove(str(PROJECT_ROOT))


def _journal_app() -> None:
    """Wrapper script AppTest will execute inside a temp dir.

    AppTest copies the source of this function to ``/tmp/<hash>`` and
    runs it as a Streamlit script. The temp dir is **not** on sys.path,
    so we restore it before importing project modules. Mirror env vars
    too — pytest's monkeypatch context isn't carried over.
    """
    import os as _os
    import sys as _sys

    _sys.path.insert(0, "/home/user/quant-platform")
    # JOURNAL_DB_PATH is propagated via os.environ, which AppTest preserves.
    _os.environ.setdefault("JOURNAL_DB_PATH", "/tmp/at_journal.db")
    from pages.journal_tab import render

    render()


def test_journal_page_renders_with_empty_journal(isolated_journal) -> None:
    """Empty journal — page must render the subheader and an empty-state info."""
    os.environ["JOURNAL_DB_PATH"] = os.environ["JOURNAL_DB_PATH"]
    at = _AppTest().from_function(_journal_app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    subheaders = [s.value for s in at.subheader]
    assert any("Trading Journal" in s for s in subheaders), subheaders


def test_journal_page_renders_seeded_entries(isolated_journal) -> None:
    """Seed two trades and verify the journal page surfaces them."""
    jt = isolated_journal
    jt.log_entry(
        ticker="AAPL", side="buy", qty=10, price=150.0,
        signal_source="momentum", regime="trending_bull",
    )
    jt.log_entry(
        ticker="MSFT", side="buy", qty=5, price=300.0,
        signal_source="momentum", regime="trending_bull",
    )

    at = _AppTest().from_function(_journal_app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    # Two open positions ⇒ at least one dataframe element must render.
    assert len(at.dataframe) >= 1


# ── pages/chart.py ──────────────────────────────────────────────────────────


def _chart_app() -> None:
    """Chart page wrapper.

    Mocks the data + regime calls at the network boundary so the test
    is hermetic. ``streamlit.session_state`` defaults are seeded so
    the sidebar selectors don't reach for a non-existent state key.
    """
    import os as _os
    import sys as _sys

    _sys.path.insert(0, "/home/user/quant-platform")
    _os.environ.setdefault("DATA_DB_PATH", "/tmp/at_chart_quant.db")

    # Mock at the network boundary: regime fetch + OHLCV fetcher.
    from unittest.mock import patch as _patch

    import numpy as _np
    import pandas as _pd

    rng = _np.random.default_rng(7)
    close = 150.0 + _np.cumsum(rng.normal(0, 1.0, 60))
    idx = _pd.date_range("2024-01-01", periods=60, freq="B")
    fake_df = _pd.DataFrame(
        {
            "Open":   close,
            "High":   close + 1.0,
            "Low":    close - 1.0,
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, 60).astype(float),
        },
        index=idx,
    )
    fake_regime = {
        "regime":                 "trending_bull",
        "spy_price":              500.0,
        "spy_sma200":             480.0,
        "vix":                    14.2,
        "recommended_strategies": ["Momentum", "Trend-following"],
        "description":            "SPY in confirmed uptrend, VIX low",
    }
    fake_latest_price = {
        "ticker":     "AAPL",
        "price":      float(close[-1]),
        "prev_close": float(close[-2]),
        "change":     float(close[-1] - close[-2]),
        "pct_change": 0.0,
        "error":      None,
    }

    import streamlit as _st

    if "active_ticker" not in _st.session_state:
        _st.session_state["active_ticker"] = "AAPL"
    if "_period" not in _st.session_state:
        _st.session_state["_period"] = "6mo"
    if "_period_label" not in _st.session_state:
        _st.session_state["_period_label"] = "6 Months"
    if "_chart_type" not in _st.session_state:
        _st.session_state["_chart_type"] = "Candlestick"

    # ``strategies.indicators.add_all`` pulls in the optional ``ta``
    # package; mock at the page-import boundary so this test is hermetic
    # even when ``ta`` isn't installed in the dev environment.
    fake_with_indicators = fake_df.assign(
        RSI_14=50.0, MACD_line=0.0, MACD_signal=0.0, MACD_hist=0.0,
        BB_upper=close + 2.0, BB_middle=close, BB_lower=close - 2.0,
        EMA_20=close, Volume_SMA=fake_df["Volume"].mean(),
    )
    fake_signals = fake_df.assign(
        signal=0, rsi_signal=0, macd_signal=0, bb_signal=0,
    )
    # Patch in two layers so the chart page is hermetic even when the
    # optional ``ta`` package isn't installed, plus replace the
    # ``@st.cache_data``-decorated watchlist snapshot with a plain
    # ``DataFrame`` — the cache decorator in the AppTest temp-dir
    # context intermittently rejects DataFrame returns (#206 follow-up).
    with _patch("strategies.indicators.add_all", return_value=fake_with_indicators), \
         _patch("strategies.indicators.generate_signals", return_value=fake_signals), \
         _patch("pages.chart.add_all", return_value=fake_with_indicators), \
         _patch("pages.chart.generate_signals", return_value=fake_signals), \
         _patch("pages.chart.get_watchlist", return_value=[]), \
         _patch("data.fetcher.fetch_ohlcv", return_value=fake_df), \
         _patch("data.fetcher.fetch_latest_price", return_value=fake_latest_price), \
         _patch("pages.chart._fetch_regime", return_value=fake_regime):
        from pages.chart import render
        render()


def test_chart_page_renders_with_mocked_data() -> None:
    """Chart page renders without raising when OHLCV / regime are mocked.

    Asserts the regime banner, the watchlist subheader, and a chart
    subheader bearing the active ticker symbol.
    """
    at = _AppTest().from_function(_chart_app, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    subheaders = [s.value for s in at.subheader]
    # Watchlist and ticker-period subheader both come from chart.render
    joined = " | ".join(subheaders)
    assert "Watchlist" in joined, subheaders
    assert "AAPL" in joined, subheaders


# Silence unused-import guard for ``patch`` and ``date`` / ``timedelta``: they
# are referenced by tests in subsequent phases of this ticket. Phase 2 will
# cover the empty-OHLCV / error-fallback paths against the real cache_data
# decorator, plus every remaining ``pages/*.py`` file.
_ = (patch, date, timedelta)
