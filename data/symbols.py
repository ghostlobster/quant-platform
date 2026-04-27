"""
data/symbols.py — symbol metadata registry for multi-asset routing.

Backed by ``quant.db``'s ``symbols`` table, this module records the asset
class, exchange, and quote currency for every ticker the platform routes
through ``broker.ibkr_bridge`` and friends. P1.8 (#146) needs this so the
IBKR contract factory can build the right ``ib_insync`` contract:

* ``stock`` → ``Stock(symbol, exchange, currency)`` — exchange ``SMART`` for
  US, ``LSE``/``HKEX`` for foreign equities.
* ``forex`` → ``Forex(symbol)`` — symbol formatted as base/quote (e.g.
  ``EURUSD``).
* ``future`` → ``Future(symbol, last_trade_date_or_contract_month, exchange,
  currency)`` — exchange ``GLOBEX`` (CME), ``NYMEX``, etc.
* ``etf`` → routed as a stock with a sentinel asset-class tag.

Public API
----------
    AssetClass                     enum-style constants
    SymbolMeta                     frozen dataclass
    register(meta)                 upsert
    get(ticker)                    Optional[SymbolMeta]
    list_by_class(asset_class)     list[SymbolMeta]
    default_for(asset_class)       SymbolMeta — sane defaults for unknown tickers
"""
from __future__ import annotations

from dataclasses import dataclass

from data.db import get_connection, init_db


class AssetClass:
    STOCK = "stock"
    ETF = "etf"
    FOREX = "forex"
    FUTURE = "future"
    OPTION = "option"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        return (cls.STOCK, cls.ETF, cls.FOREX, cls.FUTURE, cls.OPTION)


@dataclass(frozen=True)
class SymbolMeta:
    """Per-ticker routing metadata."""

    ticker: str
    asset_class: str             # one of AssetClass.*
    exchange: str                # SMART, IDEALPRO, GLOBEX, NYMEX, LSE, HKEX, ...
    currency: str = "USD"
    expiry: str | None = None    # YYYYMM for futures (ib_insync convention)
    multiplier: int | None = None  # contract multiplier (futures); 100 for options

    def __post_init__(self) -> None:
        if not self.ticker or not self.ticker.strip():
            raise ValueError("ticker must be non-empty")
        if self.asset_class not in AssetClass.all():
            raise ValueError(
                f"asset_class must be one of {AssetClass.all()}, "
                f"got {self.asset_class!r}",
            )
        if not self.exchange or not self.exchange.strip():
            raise ValueError("exchange must be non-empty")
        if not self.currency or len(self.currency) != 3:
            raise ValueError(
                f"currency must be a 3-letter ISO code, got {self.currency!r}",
            )
        if self.asset_class == AssetClass.FUTURE and not self.expiry:
            raise ValueError("future entries require an expiry (YYYYMM)")


_DEFAULTS: dict[str, SymbolMeta] = {
    AssetClass.STOCK:  SymbolMeta("__default_stock__",  AssetClass.STOCK,  "SMART",     "USD"),
    AssetClass.ETF:    SymbolMeta("__default_etf__",    AssetClass.ETF,    "SMART",     "USD"),
    AssetClass.FOREX:  SymbolMeta("__default_forex__",  AssetClass.FOREX,  "IDEALPRO",  "USD"),
    AssetClass.FUTURE: SymbolMeta(
        "__default_future__", AssetClass.FUTURE, "GLOBEX", "USD",
        expiry="202612", multiplier=50,
    ),
    AssetClass.OPTION: SymbolMeta("__default_option__", AssetClass.OPTION, "SMART",     "USD",
                                  multiplier=100),
}


def _ensure_table() -> None:
    init_db()
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS symbols (
                    ticker      TEXT PRIMARY KEY,
                    asset_class TEXT NOT NULL,
                    exchange    TEXT NOT NULL,
                    currency    TEXT NOT NULL,
                    expiry      TEXT,
                    multiplier  INTEGER
                )
                """
            )
    finally:
        conn.close()


def register(meta: SymbolMeta) -> None:
    """Upsert a single ticker's metadata."""
    _ensure_table()
    conn = get_connection()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO symbols (ticker, asset_class, exchange, currency, expiry, multiplier)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    asset_class = excluded.asset_class,
                    exchange    = excluded.exchange,
                    currency    = excluded.currency,
                    expiry      = excluded.expiry,
                    multiplier  = excluded.multiplier
                """,
                (
                    meta.ticker.upper(),
                    meta.asset_class,
                    meta.exchange,
                    meta.currency,
                    meta.expiry,
                    meta.multiplier,
                ),
            )
    finally:
        conn.close()


def get(ticker: str) -> SymbolMeta | None:
    """Return the registered metadata for ``ticker``, or ``None`` if absent."""
    _ensure_table()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM symbols WHERE ticker = ?", (ticker.upper(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return SymbolMeta(
        ticker=row["ticker"],
        asset_class=row["asset_class"],
        exchange=row["exchange"],
        currency=row["currency"],
        expiry=row["expiry"],
        multiplier=row["multiplier"],
    )


def list_by_class(asset_class: str) -> list[SymbolMeta]:
    if asset_class not in AssetClass.all():
        raise ValueError(f"unknown asset_class {asset_class!r}")
    _ensure_table()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM symbols WHERE asset_class = ? ORDER BY ticker ASC",
            (asset_class,),
        ).fetchall()
    finally:
        conn.close()
    return [
        SymbolMeta(
            ticker=r["ticker"],
            asset_class=r["asset_class"],
            exchange=r["exchange"],
            currency=r["currency"],
            expiry=r["expiry"],
            multiplier=r["multiplier"],
        )
        for r in rows
    ]


def default_for(asset_class: str) -> SymbolMeta:
    """Routing fallback when a ticker has no registered metadata.

    Returns a copy of the canonical default with the supplied ticker
    substituted in. Callers can pass the result straight to the IBKR
    contract factory.
    """
    if asset_class not in _DEFAULTS:
        raise ValueError(f"no default for asset_class {asset_class!r}")
    return _DEFAULTS[asset_class]


def resolve(ticker: str, fallback_class: str = AssetClass.STOCK) -> SymbolMeta:
    """Look up ``ticker`` in the registry, falling back to a sensible default.

    The fallback substitutes the requested ticker into the default metadata
    for ``fallback_class`` so callers always get a usable
    :class:`SymbolMeta`. Resolved entries are not auto-registered — caller
    decides whether to persist the lookup.
    """
    found = get(ticker)
    if found is not None:
        return found
    proto = default_for(fallback_class)
    return SymbolMeta(
        ticker=ticker.upper(),
        asset_class=proto.asset_class,
        exchange=proto.exchange,
        currency=proto.currency,
        expiry=proto.expiry,
        multiplier=proto.multiplier,
    )
