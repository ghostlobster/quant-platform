"""
Smoke tests for all pages/* render() functions.

Streamlit is mocked entirely — these tests verify that render() can be
called without crashing, and that the basic code paths execute.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# ── Build a comprehensive streamlit mock BEFORE importing any page ────────────

class _SessionState(dict):
    """Mimics st.session_state: both attribute and item access."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            return MagicMock()

    def __setattr__(self, k, v):
        self[k] = v


def _widget_mock() -> MagicMock:
    """Return a MagicMock configured with sensible Streamlit widget defaults."""
    m = MagicMock()
    m.text_input.return_value = ""
    m.text_area.return_value = ""
    m.number_input.return_value = 0.0
    m.button.return_value = False
    m.form_submit_button.return_value = False
    m.toggle.return_value = False
    m.checkbox.return_value = True
    m.date_input.return_value = date.today()
    m.markdown.return_value = None
    m.multiselect.side_effect = (
        lambda label, options=(), *a, **kw: list(kw.get("default", list(options)))
    )
    m.selectbox.side_effect = (
        lambda label, options=(), *a, **kw:
        list(options)[kw.get("index", 0)] if list(options) else ""
    )
    m.radio.side_effect = (
        lambda label, options=(), *a, **kw: list(options)[0] if list(options) else "Any"
    )
    m.slider.side_effect = (
        lambda label, *a, **kw: kw.get("value", a[2] if len(a) > 2 else 0)
    )
    m.columns.side_effect = (
        lambda spec, **kw: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    )
    return m


def _make_st_mock() -> MagicMock:
    """Return a fully-configured streamlit MagicMock."""
    st = _widget_mock()

    # session_state — shared and mutable
    st.session_state = _SessionState(
        {
            "active_ticker":          "AAPL",
            "_period":                "6mo",
            "_period_label":          "6 Months",
            "_chart_type":            "Candlestick",
            "_show_ema":              True,
            "_show_bb":               True,
            "_show_rsi":              True,
            "_show_macd":             True,
            "_show_signals":          True,
            "_sidebar_ticker":        "AAPL",
            "autorefresh_toggle":     False,
            "refresh_interval_select": "5 min",
            "bt_strategy":            "SMA Crossover (20/50)",
            "jnl_start":              date.today() - timedelta(days=90),
            "jnl_end":                date.today(),
            "jnl_ticker":             "",
        }
    )

    # sidebar — also a configured widget mock
    st.sidebar = _widget_mock()

    # cache_data / cache_resource — pass-through decorators
    # Handles both @st.cache_data (no parens) and @st.cache_data(...) (with parens)
    def _cache_data(*args, **kw):
        if args and callable(args[0]):
            return args[0]
        def decorator(fn):
            return fn
        return decorator

    st.cache_data = _cache_data
    st.cache_resource = _cache_data

    # stop() → StopIteration so render() terminates cleanly after st.stop()
    st.stop.side_effect = StopIteration

    return st


_ST = _make_st_mock()

# Inject the mock before any page is imported
sys.modules["streamlit"]          = _ST
sys.modules["streamlit_autorefresh"] = MagicMock()

# Now it's safe to import page modules
import pages.alerts as pg_alerts  # noqa: E402
import pages.backtest as pg_backtest  # noqa: E402
import pages.efficient_frontier as pg_ef  # noqa: E402
import pages.journal_tab as pg_journal  # noqa: E402
import pages.portfolio as pg_portfolio  # noqa: E402
import pages.screener as pg_screener  # noqa: E402
import pages.shared as pg_shared  # noqa: E402

# ── Helpers ───────────────────────────────────────────────────────────────────

def _reset_session(**overrides):
    """Reset session state to safe defaults between tests."""
    _ST.session_state.clear()
    _ST.session_state.update(
        {
            "active_ticker":          "AAPL",
            "_period":                "6mo",
            "_period_label":          "6 Months",
            "_chart_type":            "Candlestick",
            "_show_ema":              True,
            "_show_bb":               True,
            "_show_rsi":              True,
            "_show_macd":             True,
            "_show_signals":          True,
            "_sidebar_ticker":        "AAPL",
            "autorefresh_toggle":     False,
            "refresh_interval_select": "5 min",
            "bt_strategy":            "SMA Crossover (20/50)",
            "jnl_start":              date.today() - timedelta(days=90),
            "jnl_end":                date.today(),
            "jnl_ticker":             "",
        }
    )
    _ST.session_state.update(overrides)
    # Re-reset common widget mocks
    _ST.button.return_value           = False
    _ST.form_submit_button.return_value = False
    _ST.sidebar.button.return_value   = False
    _ST.date_input.return_value       = date.today() - timedelta(days=30)
    _ST.number_input.return_value     = 5.0
    _ST.text_input.return_value       = "AAPL"
    _ST.text_area.return_value        = ""


def _ohlcv(n: int = 60) -> pd.DataFrame:
    np.random.seed(0)
    c = 100 + np.cumsum(np.random.randn(n) * 0.5)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": np.ones(n) * 1e6},
        index=idx,
    )


# ── pages/shared.py ───────────────────────────────────────────────────────────

class TestPagesShared:
    def test_set_ticker_updates_session_state(self):
        _reset_session()
        pg_shared.set_ticker("TSLA")
        assert _ST.session_state["active_ticker"] == "TSLA"

    def test_set_ticker_uppercases(self):
        _reset_session()
        pg_shared.set_ticker("tsla")
        assert _ST.session_state["active_ticker"] == "TSLA"

    def test_on_sidebar_ticker_change_valid(self):
        _reset_session()
        _ST.session_state["_sidebar_ticker"] = "MSFT"
        pg_shared._on_sidebar_ticker_change()
        assert _ST.session_state["active_ticker"] == "MSFT"
        assert "_ticker_error" not in _ST.session_state

    def test_on_sidebar_ticker_change_invalid(self):
        _reset_session()
        _ST.session_state["_sidebar_ticker"] = "123INVALID"
        pg_shared._on_sidebar_ticker_change()
        assert "_ticker_error" in _ST.session_state

    def test_render_sidebar_returns_dict(self):
        _reset_session()
        with patch("pages.shared.get_watchlist", return_value=["AAPL", "MSFT"]), \
             patch("pages.shared.add_ticker"), \
             patch("pages.shared.remove_ticker"):
            result = pg_shared.render_sidebar()
        assert isinstance(result, dict)
        assert "ticker" in result
        assert "period" in result

    def test_render_sidebar_watchlist_populated(self):
        _reset_session()
        watchlist = ["AAPL", "MSFT", "NVDA"]
        with patch("pages.shared.get_watchlist", return_value=watchlist), \
             patch("pages.shared.add_ticker"), \
             patch("pages.shared.remove_ticker"):
            result = pg_shared.render_sidebar()
        assert result["watchlist"] == watchlist


# ── pages/backtest.py ─────────────────────────────────────────────────────────

class TestPagesBacktest:
    def test_render_no_run_does_not_crash(self):
        _reset_session()
        # button not clicked → no backtest runs
        _ST.button.return_value = False
        _ST.selectbox.return_value = "SMA Crossover (20/50)"
        _ST.date_input.return_value = date.today() - timedelta(days=30)
        pg_backtest.render()  # should not raise

    def test_render_with_backtest_run(self):
        _reset_session()
        _ST.button.return_value = True

        from backtester.engine import BacktestResult
        fake_result = BacktestResult(
            ticker="AAPL",
            strategy="sma_crossover",
            start_date=pd.Timestamp("2022-01-01"),
            end_date=pd.Timestamp("2024-01-01"),
            total_return_pct=15.0,
            buy_hold_return_pct=12.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            calmar_ratio=0.8,
            max_drawdown_pct=-10.0,
            win_rate_pct=55.0,
            num_trades=20,
            equity_curve=pd.DataFrame({"Equity": [1.0] * 10}),
            trades=[],
        )

        with patch("pages.backtest.fetch_ohlcv", return_value=_ohlcv()), \
             patch("pages.backtest.run_backtest", return_value=fake_result), \
             patch("pages.backtest.build_equity_chart", return_value=MagicMock()), \
             patch("pages.backtest.build_trade_log_df", return_value=pd.DataFrame()):
            pg_backtest.render()

    def test_render_handles_fetch_error(self):
        _reset_session()
        _ST.button.return_value = True

        with patch("pages.backtest.fetch_ohlcv", side_effect=ValueError("no data")):
            try:
                pg_backtest.render()
            except StopIteration:
                pass  # st.stop() was called — expected


# ── pages/efficient_frontier.py ───────────────────────────────────────────────

class TestPagesEfficientFrontier:
    def test_parse_holdings_empty(self):
        assert pg_ef._parse_holdings("") == {}

    def test_parse_holdings_valid(self):
        text = "AAPL:15000\nMSFT:8000"
        result = pg_ef._parse_holdings(text)
        assert result == {"AAPL": 15000.0, "MSFT": 8000.0}

    def test_parse_holdings_skips_malformed(self):
        text = "AAPL:15000\nbad_line\nMSFT:abc\nNVDA:5000"
        result = pg_ef._parse_holdings(text)
        assert "AAPL" in result
        assert "NVDA" in result
        assert "bad_line" not in result

    def test_parse_holdings_comma_in_value(self):
        result = pg_ef._parse_holdings("AAPL:15,000")
        assert result["AAPL"] == 15000.0

    def test_render_too_few_tickers_shows_warning(self):
        _reset_session()
        _ST.text_input.return_value = "AAPL"  # only 1 ticker
        pg_ef.render()
        _ST.warning.assert_called()

    def test_render_with_two_tickers_runs(self):
        _reset_session()
        _ST.text_input.return_value = "AAPL, MSFT"
        _ST.selectbox.return_value = "2y"
        _ST.button.return_value = False

        close = pd.Series(np.linspace(100, 120, 200))
        price_data = {"AAPL": close, "MSFT": close * 1.1}

        from risk.markowitz import OptimalPortfolio
        fake_max_sharpe = OptimalPortfolio(
            weights={"AAPL": 0.6, "MSFT": 0.4},
            expected_return=0.10,
            expected_volatility=0.15,
            sharpe_ratio=0.67,
        )
        fake_min_vol = OptimalPortfolio(
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.08,
            expected_volatility=0.12,
            sharpe_ratio=0.67,
        )

        with patch("pages.efficient_frontier._fetch_price_data", return_value=price_data), \
             patch("pages.efficient_frontier.build_efficient_frontier_chart", return_value=MagicMock()), \
             patch("pages.efficient_frontier.get_max_sharpe_portfolio", return_value=fake_max_sharpe), \
             patch("pages.efficient_frontier.get_min_volatility_portfolio", return_value=fake_min_vol):
            pg_ef.render()


# ── pages/journal_tab.py ──────────────────────────────────────────────────────

class TestPagesJournal:
    def test_render_empty_journal(self):
        _reset_session()
        with patch("pages.journal_tab.get_journal", return_value=pd.DataFrame()), \
             patch("pages.journal_tab.win_rate_by_signal_source", return_value=pd.DataFrame()), \
             patch("pages.journal_tab.avg_pnl_by_regime", return_value=pd.DataFrame()):
            pg_journal.render()

    def test_render_with_journal_entries(self):
        _reset_session()
        journal_df = pd.DataFrame(
            {
                "id": [1, 2],
                "ticker": ["AAPL", "MSFT"],
                "side": ["buy", "buy"],
                "qty": [10.0, 5.0],
                "entry_price": [150.0, 300.0],
                "entry_time": ["2024-01-01", "2024-01-02"],
                "exit_price": [160.0, None],
                "exit_time": ["2024-01-10", None],
                "pnl": [100.0, None],
                "signal_source": ["RSI", "MACD"],
                "regime": ["trending_bull", "mean_reverting"],
                "exit_reason": ["target", None],
            }
        )
        win_rate_df = pd.DataFrame(
            {
                "signal_source": ["RSI"],
                "win_rate": [0.6],
                "total_trades": [5],
                "wins": [3],
                "avg_pnl": [50.0],
            }
        )
        pnl_df = pd.DataFrame(
            {
                "regime": ["trending_bull"],
                "avg_pnl": [75.0],
                "total_trades": [3],
                "win_rate": [0.67],
            }
        )

        with patch("pages.journal_tab.get_journal", return_value=journal_df), \
             patch("pages.journal_tab.win_rate_by_signal_source", return_value=win_rate_df), \
             patch("pages.journal_tab.avg_pnl_by_regime", return_value=pnl_df):
            pg_journal.render()


# ── pages/portfolio.py ────────────────────────────────────────────────────────

class TestPagesPortfolio:
    def _fake_account(self):
        return {"cash": 90_000.0, "realised_pnl": 1_000.0}

    def test_render_empty_portfolio(self):
        _reset_session()
        with patch("pages.portfolio.get_portfolio", return_value=pd.DataFrame()), \
             patch("pages.portfolio.get_account", return_value=self._fake_account()), \
             patch("pages.portfolio.get_trade_history", return_value=pd.DataFrame()), \
             patch("pages.portfolio.fetch_latest_price", return_value={"price": 150.0, "error": None}):
            pg_portfolio.render()

    def test_render_with_positions(self):
        _reset_session()
        port_df = pd.DataFrame(
            {
                "Ticker":       ["AAPL"],
                "Shares":       [10.0],
                "Avg Cost":     [140.0],
                "Market Value": [1500.0],
                "Unrealised P&L": [100.0],
            }
        )
        with patch("pages.portfolio.get_portfolio", return_value=port_df), \
             patch("pages.portfolio.get_account", return_value=self._fake_account()), \
             patch("pages.portfolio.get_trade_history", return_value=pd.DataFrame()), \
             patch("pages.portfolio.fetch_latest_price", return_value={"price": 150.0, "error": None}):
            pg_portfolio.render()

    def test_render_buy_button_clicked(self):
        _reset_session()
        _ST.button.return_value = True
        _ST.number_input.return_value = 5.0  # shares & price

        with patch("pages.portfolio.get_portfolio", return_value=pd.DataFrame()), \
             patch("pages.portfolio.get_account", return_value=self._fake_account()), \
             patch("pages.portfolio.get_trade_history", return_value=pd.DataFrame()), \
             patch("pages.portfolio.fetch_latest_price", return_value={"price": 150.0, "error": None}), \
             patch("pages.portfolio.pt_buy", return_value={"status": "ok"}), \
             patch("pages.portfolio.pt_sell", return_value={"status": "ok"}), \
             patch("pages.portfolio.reset_account"):
            pg_portfolio.render()


# ── pages/screener.py ─────────────────────────────────────────────────────────

class TestPagesScreener:
    def test_render_empty_results(self):
        _reset_session()
        with patch("pages.screener.run_screen", return_value=pd.DataFrame()), \
             patch("pages.screener.set_ticker"):
            pg_screener.render()

    def test_render_with_results(self):
        _reset_session()
        screen_df = pd.DataFrame(
            {
                "Ticker":      ["AAPL", "MSFT"],
                "Sector":      ["Technology", "Technology"],
                "Name":        ["Apple", "Microsoft"],
                "Last Price":  [150.0, 300.0],
                "RSI":         [45.0, 60.0],
                "Change 5d (%)": [1.5, -0.5],
                "Vol Ratio":   [1.2, 0.8],
                "Above SMA50": [True, True],
                "Signal":      ["Neutral", "Neutral"],
            }
        )
        with patch("pages.screener.run_screen", return_value=screen_df), \
             patch("pages.screener.set_ticker"):
            pg_screener.render()

    def test_render_run_screen_shows_results(self):
        """Cover lines 83-189: button clicked → results table and ticker buttons."""
        _reset_session()
        _ST.button.return_value = True  # simulate "Run Screen" clicked
        screen_df = pd.DataFrame(
            {
                "Ticker":       ["AAPL"],
                "Sector":       ["Technology"],
                "Name":         ["Apple"],
                "Last Price":   [150.0],
                "RSI":          [45.0],
                "Change 1d (%)": [1.0],  # matches sc_change_days=1 (first selectbox option)
                "Vol Ratio":    [1.2],
                "Above SMA50":  [True],
                "Signal":       ["Oversold"],
            }
        )
        with patch("pages.screener.run_screen", return_value=screen_df), \
             patch("pages.screener.set_ticker"):
            pg_screener.render()

    def test_render_run_screen_empty_results(self):
        """Cover lines 83-103: button clicked but no tickers match filters."""
        _reset_session()
        _ST.button.return_value = True
        with patch("pages.screener.run_screen", return_value=pd.DataFrame()), \
             patch("pages.screener.set_ticker"):
            pg_screener.render()


# ── pages/alerts.py ───────────────────────────────────────────────────────────

class TestPagesAlerts:
    def test_render_no_action(self):
        _reset_session()
        _ST.form_submit_button.return_value = False
        _ST.button.return_value = False
        with patch("pages.alerts.get_alerts", return_value=pd.DataFrame(
            columns=["ID", "Ticker", "Type", "Threshold", "Enabled", "Created", "Last Triggered"]
        )), \
             patch("pages.alerts.add_alert"), \
             patch("pages.alerts.delete_alert"), \
             patch("pages.alerts.toggle_alert"), \
             patch("pages.alerts.check_alerts", return_value=[]):
            pg_alerts.render()

    def test_render_add_alert_submitted(self):
        _reset_session()
        _ST.form_submit_button.return_value = True
        _ST.text_input.return_value = "AAPL"
        _ST.number_input.return_value = 200.0

        with patch("pages.alerts.get_alerts", return_value=pd.DataFrame(
            columns=["ID", "Ticker", "Type", "Threshold", "Enabled", "Created", "Last Triggered"]
        )), \
             patch("pages.alerts.add_alert", return_value=1), \
             patch("pages.alerts.delete_alert"), \
             patch("pages.alerts.toggle_alert"), \
             patch("pages.alerts.check_alerts", return_value=[]):
            pg_alerts.render()

    def test_render_with_active_alerts(self):
        """Cover lines 74-114: non-empty alert table with enabled and disabled rows."""
        _reset_session()
        _ST.form_submit_button.return_value = False
        alerts_df = pd.DataFrame(
            {
                "ID":            [1, 2],
                "Ticker":        ["AAPL", "MSFT"],
                "Type":          ["price_above", "rsi_below"],
                "Threshold":     [150.0, 30.0],
                "Enabled":       [True, False],
                "Created":       ["2024-01-01 10:00", "2024-01-02 10:00"],
                "Last Triggered": ["—", "2024-01-10 09:30"],
            }
        )
        with patch("pages.alerts.get_alerts", return_value=alerts_df), \
             patch("pages.alerts.add_alert"), \
             patch("pages.alerts.delete_alert"), \
             patch("pages.alerts.toggle_alert"), \
             patch("pages.alerts.check_alerts", return_value=[]):
            pg_alerts.render()

    def test_render_check_now_with_active_alerts(self):
        """Cover lines 129-215: check-now path with enabled alerts triggering."""
        _reset_session()
        _ST.form_submit_button.return_value = False
        alerts_df = pd.DataFrame(
            {
                "ID":            [1],
                "Ticker":        ["AAPL"],
                "Type":          ["price_above"],
                "Threshold":     [150.0],
                "Enabled":       [True],
                "Created":       ["2024-01-01 10:00"],
                "Last Triggered": ["—"],
            }
        )
        triggered = [
            {
                "id": 1,
                "ticker": "AAPL",
                "alert_type": "price_above",
                "threshold": 150.0,
                "price": 160.0,
                "rsi": 55.0,
                "message": "AAPL: Price ≥ threshold 150.00 triggered — price=$160.00",
                "fired_at": "2024-01-15 10:30:00",
            }
        ]
        with patch("pages.alerts.get_alerts", return_value=alerts_df), \
             patch("pages.alerts.add_alert"), \
             patch("pages.alerts.delete_alert"), \
             patch("pages.alerts.toggle_alert"), \
             patch("pages.alerts.check_alerts", return_value=triggered), \
             patch("pages.alerts.fetch_latest_price",
                   return_value={"price": 160.0, "error": None}), \
             patch("pages.alerts.fetch_ohlcv", return_value=_ohlcv()), \
             patch("pages.alerts.compute_rsi", return_value=55.0):
            pg_alerts.render()

    def test_render_check_alerts_with_triggered(self):
        _reset_session()
        _ST.form_submit_button.return_value = False
        _ST.button.return_value = True  # "Check alerts now" button

        triggered = [
            {
                "id": 1,
                "ticker": "AAPL",
                "alert_type": "price_above",
                "threshold": 150.0,
                "price": 160.0,
                "rsi": 55.0,
                "message": "AAPL: Price ≥ threshold 150.00 triggered — price=$160.00",
                "fired_at": "2024-01-15 10:30:00",
            }
        ]

        with patch("pages.alerts.get_alerts", return_value=pd.DataFrame(
            columns=["ID", "Ticker", "Type", "Threshold", "Enabled", "Created", "Last Triggered"]
        )), \
             patch("pages.alerts.add_alert"), \
             patch("pages.alerts.delete_alert"), \
             patch("pages.alerts.toggle_alert"), \
             patch("pages.alerts.check_alerts", return_value=triggered), \
             patch("pages.alerts.fetch_latest_price", return_value={"price": 160.0, "error": None}), \
             patch("pages.alerts.fetch_ohlcv", return_value=_ohlcv()), \
             patch("pages.alerts.compute_rsi", return_value=55.0):
            pg_alerts.render()
