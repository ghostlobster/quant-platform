"""
Real-time data feed via Alpaca WebSocket (equity quotes) and polling fallback.
When Alpaca credentials are absent, falls back to yfinance polling.
"""
import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import yfinance as yf

from broker.alpaca_bridge import _get_config as _alpaca_get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Sentinels — set to None so credentials are NOT read at import time.
# Tests may patch these directly; None means "read from alpaca_bridge lazily".
ALPACA_API_KEY = None
ALPACA_SECRET_KEY = None

_ALPACA_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
_DEFAULT_POLL_SECONDS = 60


@dataclass
class Quote:
    ticker: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: str  # ISO8601


@dataclass
class RealtimeFeed:
    """Thread-safe quote cache. Register callbacks for live updates."""
    _quotes: dict = field(default_factory=dict)       # {ticker: Quote}
    _callbacks: list = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _tickers: set = field(default_factory=set)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _mode: str = field(default='polling')
    _poll_interval: int = field(default=_DEFAULT_POLL_SECONDS)
    _thread: Optional[threading.Thread] = field(default=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _notify(self, quote: Quote) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(quote)
            except Exception:
                logger.exception("Error in quote callback for %s", quote.ticker)

    def _update_quote(self, quote: Quote) -> None:
        """Write a quote to the cache and fire callbacks. Thread-safe."""
        with self._lock:
            self._quotes[quote.ticker] = quote
        self._notify(quote)

    def _ensure_thread(self) -> None:
        """Start the background feed thread if it isn't already running."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if self._stop_event.is_set():
                return
            target = self._run_alpaca_ws if self._mode == 'alpaca' else self._run_polling
            t = threading.Thread(target=target, daemon=True, name=f"RealtimeFeed-{self._mode}")
            self._thread = t
        t.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def subscribe(self, tickers: list[str]) -> None:
        """Start streaming quotes for given tickers. Non-blocking."""
        upper = [t.upper() for t in tickers]
        with self._lock:
            self._tickers.update(upper)
        self._ensure_thread()

    def unsubscribe(self, tickers: list[str]) -> None:
        """Remove tickers from the subscription list and clear their cached quotes."""
        with self._lock:
            for t in tickers:
                self._tickers.discard(t.upper())
                self._quotes.pop(t.upper(), None)

    def get_quote(self, ticker: str) -> Optional[Quote]:
        """Returns latest cached quote, or None if not yet received."""
        with self._lock:
            return self._quotes.get(ticker.upper())

    def get_all_quotes(self) -> dict[str, Quote]:
        """Returns a snapshot of all current quotes."""
        with self._lock:
            return dict(self._quotes)

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """Register a callback invoked on every new quote received."""
        with self._lock:
            self._callbacks.append(callback)

    def stop(self) -> None:
        """Gracefully stop the feed."""
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=5)

    # ------------------------------------------------------------------
    # Polling backend
    # ------------------------------------------------------------------

    def _run_polling(self) -> None:
        logger.info("RealtimeFeed: polling mode started (interval=%ds)", self._poll_interval)
        while not self._stop_event.is_set():
            with self._lock:
                tickers = list(self._tickers)

            for ticker in tickers:
                if self._stop_event.is_set():
                    break
                try:
                    info = yf.Ticker(ticker).fast_info
                    last = float(getattr(info, 'last_price', 0) or 0)
                    quote = Quote(
                        ticker=ticker,
                        bid=last,
                        ask=last,
                        last=last,
                        volume=int(getattr(info, 'last_volume', 0) or 0),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    self._update_quote(quote)
                except Exception:
                    logger.exception("Polling error for %s", ticker)

            # Honour stop signal promptly even during a long sleep interval
            self._stop_event.wait(self._poll_interval)

        logger.info("RealtimeFeed: polling mode stopped")

    # ------------------------------------------------------------------
    # Alpaca WebSocket backend
    # ------------------------------------------------------------------

    def _run_alpaca_ws(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._alpaca_ws_loop())
        except Exception:
            logger.exception("Alpaca WS failed — falling back to polling")
            self._mode = 'polling'
            if not self._stop_event.is_set():
                self._run_polling()
        finally:
            loop.close()

    async def _alpaca_ws_loop(self) -> None:
        import websockets  # optional dependency

        subscribed: set = set()
        async with websockets.connect(_ALPACA_WS_URL) as ws:
            # Authenticate
            _cfg = _alpaca_get_config()
            _key = ALPACA_API_KEY if ALPACA_API_KEY is not None else _cfg["api_key"]
            _secret = ALPACA_SECRET_KEY if ALPACA_SECRET_KEY is not None else _cfg["secret_key"]
            await ws.send(json.dumps({
                "action": "auth",
                "key": _key,
                "secret": _secret,
            }))
            raw = await ws.recv()
            msg = json.loads(raw)
            if not self._alpaca_auth_ok(msg):
                raise ConnectionError(f"Alpaca auth failed: {msg}")
            logger.info("RealtimeFeed: Alpaca WS authenticated")

            while not self._stop_event.is_set():
                with self._lock:
                    current = set(self._tickers)

                to_add = current - subscribed
                if to_add:
                    await ws.send(json.dumps({"action": "subscribe", "quotes": list(to_add)}))
                    subscribed.update(to_add)

                to_remove = subscribed - current
                if to_remove:
                    await ws.send(json.dumps({"action": "unsubscribe", "quotes": list(to_remove)}))
                    subscribed -= to_remove

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    messages = json.loads(raw)
                    if not isinstance(messages, list):
                        messages = [messages]
                    for m in messages:
                        if m.get("T") == "q":
                            quote = Quote(
                                ticker=m.get("S", ""),
                                bid=float(m.get("bp", 0)),
                                ask=float(m.get("ap", 0)),
                                last=float(m.get("lp", m.get("bp", 0))),
                                volume=int(m.get("ls", 0)),
                                timestamp=m.get("t", datetime.now(timezone.utc).isoformat()),
                            )
                            self._update_quote(quote)
                except asyncio.TimeoutError:
                    continue

    @staticmethod
    def _alpaca_auth_ok(msg) -> bool:
        if isinstance(msg, list):
            return any(
                m.get("T") == "success" and m.get("msg") == "authenticated"
                for m in msg
            )
        return msg.get("T") == "success" and msg.get("msg") == "authenticated"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_feed(mode: str = 'auto') -> RealtimeFeed:
    """
    mode='auto': use Alpaca WS if credentials present, else yfinance polling
    mode='alpaca': force Alpaca WebSocket
    mode='polling': force yfinance polling (useful for testing / non-US hours)
    Polling interval controlled by REALTIME_POLL_SECONDS env var (default: 60)
    """
    poll_interval = int(os.getenv("REALTIME_POLL_SECONDS", str(_DEFAULT_POLL_SECONDS)))

    if mode == 'auto':
        _cfg = _alpaca_get_config()
        _key = ALPACA_API_KEY if ALPACA_API_KEY is not None else _cfg["api_key"]
        _secret = ALPACA_SECRET_KEY if ALPACA_SECRET_KEY is not None else _cfg["secret_key"]
        has_creds = bool(_key and _secret)
        mode = 'alpaca' if has_creds else 'polling'
        logger.info("create_feed: auto-selected mode=%s", mode)

    logger.info("create_feed: mode=%s poll_interval=%ds", mode, poll_interval)
    return RealtimeFeed(_mode=mode, _poll_interval=poll_interval)
