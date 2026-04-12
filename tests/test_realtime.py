"""
Tests for data/realtime.py.
All network I/O (yfinance, websockets) is mocked — no real connections.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from data.realtime import Quote, RealtimeFeed, create_feed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fast_info(last_price: float = 150.0, last_volume: int = 1_000_000):
    """Return a mock object that mimics yf.Ticker.fast_info."""
    fi = MagicMock()
    fi.last_price = last_price
    fi.last_volume = last_volume
    return fi


def _make_feed(poll_interval: float = 0.05) -> RealtimeFeed:
    """Polling feed with a very short poll interval, suitable for tests."""
    return create_feed(mode='polling')  # _poll_interval overridden below


# ---------------------------------------------------------------------------
# Test: create_feed returns correct type
# ---------------------------------------------------------------------------

class TestCreateFeed:
    def test_polling_mode_returns_realtime_feed(self):
        feed = create_feed(mode='polling')
        assert isinstance(feed, RealtimeFeed)
        assert feed._mode == 'polling'

    def test_auto_mode_no_creds_returns_polling(self):
        with patch('data.realtime.ALPACA_API_KEY', ''), \
             patch('data.realtime.ALPACA_SECRET_KEY', ''):
            feed = create_feed(mode='auto')
        assert feed._mode == 'polling'

    def test_auto_mode_with_creds_returns_alpaca(self):
        with patch('data.realtime.ALPACA_API_KEY', 'key'), \
             patch('data.realtime.ALPACA_SECRET_KEY', 'secret'):
            feed = create_feed(mode='auto')
        assert feed._mode == 'alpaca'

    def test_poll_interval_from_env(self, monkeypatch):
        monkeypatch.setenv('REALTIME_POLL_SECONDS', '30')
        feed = create_feed(mode='polling')
        assert feed._poll_interval == 30


# ---------------------------------------------------------------------------
# Test: subscribe + get_all_quotes (mocked yfinance polling)
# ---------------------------------------------------------------------------

class TestSubscribeAndQuotes:
    def test_subscribe_returns_dict_after_poll(self):
        """After subscribing and one poll cycle, get_all_quotes returns a dict."""
        feed = RealtimeFeed(_mode='polling', _poll_interval=0.01)

        mock_ticker = MagicMock()
        mock_ticker.fast_info = _make_fast_info(last_price=200.0)

        with patch('yfinance.Ticker', return_value=mock_ticker):
            feed.subscribe(['MSFT'])
            # Wait long enough for at least one poll to complete
            deadline = time.time() + 2.0
            while time.time() < deadline:
                if feed.get_all_quotes():
                    break
                time.sleep(0.02)

        quotes = feed.get_all_quotes()
        assert isinstance(quotes, dict)
        assert 'MSFT' in quotes
        assert quotes['MSFT'].last == 200.0

        feed.stop()

    def test_get_all_quotes_empty_before_subscribe(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        assert feed.get_all_quotes() == {}

    def test_get_quote_returns_none_for_unknown_ticker(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        assert feed.get_quote('UNKNOWN') is None


# ---------------------------------------------------------------------------
# Test: on_quote callback fires on synthetic quote injection
# ---------------------------------------------------------------------------

class TestOnQuoteCallback:
    def test_callback_fires_on_injected_quote(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        received: list[Quote] = []

        feed.on_quote(lambda q: received.append(q))

        synthetic = Quote(
            ticker='AAPL', bid=149.9, ask=150.1, last=150.0,
            volume=500_000, timestamp='2026-01-01T00:00:00+00:00',
        )
        feed._update_quote(synthetic)

        assert len(received) == 1
        assert received[0].ticker == 'AAPL'
        assert received[0].last == 150.0

    def test_multiple_callbacks_all_fire(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        results = []

        feed.on_quote(lambda q: results.append('cb1'))
        feed.on_quote(lambda q: results.append('cb2'))

        feed._update_quote(Quote('SPY', 400.0, 400.1, 400.05, 1_000, '2026-01-01T00:00:00+00:00'))

        assert results == ['cb1', 'cb2']

    def test_callback_exception_does_not_propagate(self):
        """A crashing callback must not break the feed."""
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        feed.on_quote(lambda q: (_ for _ in ()).throw(RuntimeError("boom")))

        # Should not raise
        feed._update_quote(Quote('QQQ', 300.0, 300.1, 300.05, 200, '2026-01-01T00:00:00+00:00'))


# ---------------------------------------------------------------------------
# Test: unsubscribe clears cache
# ---------------------------------------------------------------------------

class TestUnsubscribe:
    def test_unsubscribe_removes_cached_quote(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        feed._update_quote(Quote('TSLA', 180.0, 180.5, 180.2, 300_000, '2026-01-01T00:00:00+00:00'))
        assert feed.get_quote('TSLA') is not None

        feed.unsubscribe(['TSLA'])
        assert feed.get_quote('TSLA') is None


# ---------------------------------------------------------------------------
# Test: stop() exits cleanly without hanging
# ---------------------------------------------------------------------------

class TestStop:
    def test_stop_without_subscribe(self):
        """stop() on an idle feed should return quickly."""
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        start = time.time()
        feed.stop()
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_stop_with_active_polling_thread(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=0.05)

        mock_ticker = MagicMock()
        mock_ticker.fast_info = _make_fast_info()

        with patch('yfinance.Ticker', return_value=mock_ticker):
            feed.subscribe(['AAPL'])
            time.sleep(0.1)  # let the thread start

        start = time.time()
        feed.stop()
        elapsed = time.time() - start
        assert elapsed < 3.0  # well within the 5-second join timeout

        if feed._thread is not None:
            assert not feed._thread.is_alive()

    def test_stop_is_idempotent(self):
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        feed.stop()
        feed.stop()  # second call must not raise


# ---------------------------------------------------------------------------
# Test: thread-safety — concurrent reads and writes don't raise
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_get_and_update(self):
        """Concurrent get_quote / _update_quote calls must not raise."""
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        errors: list[Exception] = []

        def writer():
            for i in range(200):
                try:
                    feed._update_quote(Quote(
                        ticker='GOOG',
                        bid=float(100 + i),
                        ask=float(100 + i + 0.1),
                        last=float(100 + i),
                        volume=i,
                        timestamp='2026-01-01T00:00:00+00:00',
                    ))
                except Exception as exc:
                    errors.append(exc)

        def reader():
            for _ in range(200):
                try:
                    feed.get_quote('GOOG')
                    feed.get_all_quotes()
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == [], f"Thread-safety errors: {errors}"

    def test_subscribe_from_multiple_threads(self):
        """Subscribing from many threads must not corrupt internal state."""
        feed = RealtimeFeed(_mode='polling', _poll_interval=60)
        errors: list[Exception] = []

        def subscriber(ticker):
            try:
                feed.subscribe([ticker])
                feed.unsubscribe([ticker])
            except Exception as exc:
                errors.append(exc)

        tickers = [f'T{i}' for i in range(20)]
        threads = [threading.Thread(target=subscriber, args=(t,)) for t in tickers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
