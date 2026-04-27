"""
strategies/options_legs.py — multi-leg options builders.

Returns ``list[OptionLeg]`` for the four most common indie defined-risk
structures: vertical spread, iron condor, straddle, calendar. The legs
are broker-agnostic; the Tradier multi-leg wrapper in
:mod:`broker.tradier_bridge` (P1.7 ``place_multi_leg``) translates
``OptionLeg`` into the Tradier order payload.

Each builder normalises strikes / expiries so the test surface is
deterministic. Validation lives in :func:`OptionLeg.__post_init__` —
strike > 0, qty > 0, side ∈ Tradier's ``buy_to_open``/``sell_to_open``/
``buy_to_close``/``sell_to_close``, option_type ∈ {"call", "put"}.
"""
from __future__ import annotations

from dataclasses import dataclass

_VALID_SIDES = ("buy_to_open", "sell_to_open", "buy_to_close", "sell_to_close")
_VALID_OPTION_TYPES = ("call", "put")


@dataclass(frozen=True)
class OptionLeg:
    """One leg of a multi-leg options order."""

    underlying: str
    expiry: str               # ISO date string YYYY-MM-DD
    strike: float
    option_type: str          # "call" | "put"
    side: str                 # buy_to_open | sell_to_open | buy_to_close | sell_to_close
    qty: int                  # contracts (always positive — direction encoded by side)
    option_symbol: str | None = None  # OCC symbol; populated by adapter when None

    def __post_init__(self) -> None:
        if not self.underlying or not self.underlying.strip():
            raise ValueError("underlying must be a non-empty ticker")
        if not self.expiry or len(self.expiry) != 10:
            raise ValueError(
                f"expiry must be ISO 'YYYY-MM-DD', got {self.expiry!r}",
            )
        if self.strike <= 0:
            raise ValueError(f"strike must be > 0, got {self.strike}")
        if self.option_type.lower() not in _VALID_OPTION_TYPES:
            raise ValueError(
                f"option_type must be 'call' or 'put', got {self.option_type!r}",
            )
        if self.side.lower() not in _VALID_SIDES:
            raise ValueError(
                f"side must be one of {_VALID_SIDES}, got {self.side!r}",
            )
        if self.qty <= 0:
            raise ValueError(f"qty must be > 0, got {self.qty}")

    @property
    def is_long(self) -> bool:
        return self.side.lower().startswith("buy")

    @property
    def signed_qty(self) -> int:
        """Contract count signed by direction (long positive, short negative)."""
        return self.qty if self.is_long else -self.qty


# ── Vertical spread ──────────────────────────────────────────────────────────

def vertical_spread(
    underlying: str,
    expiry: str,
    long_strike: float,
    short_strike: float,
    *,
    option_type: str = "call",
    qty: int = 1,
) -> list[OptionLeg]:
    """Long vertical spread (debit when call & long_strike < short_strike)."""
    if long_strike == short_strike:
        raise ValueError("long_strike and short_strike must differ")
    return [
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(long_strike), option_type=option_type,
            side="buy_to_open", qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(short_strike), option_type=option_type,
            side="sell_to_open", qty=qty,
        ),
    ]


# ── Iron condor ─────────────────────────────────────────────────────────────

def iron_condor(
    underlying: str,
    expiry: str,
    *,
    put_long_strike: float,
    put_short_strike: float,
    call_short_strike: float,
    call_long_strike: float,
    qty: int = 1,
) -> list[OptionLeg]:
    """Iron condor — short the inner strikes, long the wings.

    Strike order must be: ``put_long < put_short < call_short < call_long``.
    The structure is credit at open and defined-risk in both wings.
    """
    if not (put_long_strike < put_short_strike < call_short_strike < call_long_strike):
        raise ValueError(
            "iron condor strikes must satisfy "
            "put_long < put_short < call_short < call_long, got "
            f"{put_long_strike}, {put_short_strike}, "
            f"{call_short_strike}, {call_long_strike}",
        )
    return [
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(put_long_strike), option_type="put",
            side="buy_to_open", qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(put_short_strike), option_type="put",
            side="sell_to_open", qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(call_short_strike), option_type="call",
            side="sell_to_open", qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(call_long_strike), option_type="call",
            side="buy_to_open", qty=qty,
        ),
    ]


# ── Straddle ────────────────────────────────────────────────────────────────

def straddle(
    underlying: str,
    expiry: str,
    strike: float,
    *,
    long: bool = True,
    qty: int = 1,
) -> list[OptionLeg]:
    """Long straddle (volatility play) or short straddle (premium harvest)."""
    side = "buy_to_open" if long else "sell_to_open"
    return [
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(strike), option_type="call",
            side=side, qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=expiry,
            strike=float(strike), option_type="put",
            side=side, qty=qty,
        ),
    ]


# ── Calendar spread ─────────────────────────────────────────────────────────

def calendar(
    underlying: str,
    near_expiry: str,
    far_expiry: str,
    strike: float,
    *,
    option_type: str = "call",
    qty: int = 1,
) -> list[OptionLeg]:
    """Calendar spread — short the near-dated leg, long the far-dated."""
    if near_expiry >= far_expiry:
        raise ValueError(
            f"near_expiry must be earlier than far_expiry "
            f"(got {near_expiry!r} vs {far_expiry!r})",
        )
    return [
        OptionLeg(
            underlying=underlying, expiry=near_expiry,
            strike=float(strike), option_type=option_type,
            side="sell_to_open", qty=qty,
        ),
        OptionLeg(
            underlying=underlying, expiry=far_expiry,
            strike=float(strike), option_type=option_type,
            side="buy_to_open", qty=qty,
        ),
    ]


def closing_legs(legs: list[OptionLeg]) -> list[OptionLeg]:
    """Return the closing-side mirror of an open multi-leg structure.

    Each ``buy_to_open`` becomes ``sell_to_close``; each ``sell_to_open``
    becomes ``buy_to_close``. Useful for one-click "close the condor"
    flows in the Streamlit UI.
    """
    pair = {
        "buy_to_open":  "sell_to_close",
        "sell_to_open": "buy_to_close",
    }
    out: list[OptionLeg] = []
    for leg in legs:
        side_lower = leg.side.lower()
        if side_lower not in pair:
            raise ValueError(
                f"closing_legs only accepts opening sides, got {leg.side!r}",
            )
        out.append(
            OptionLeg(
                underlying=leg.underlying, expiry=leg.expiry,
                strike=leg.strike, option_type=leg.option_type,
                side=pair[side_lower], qty=leg.qty,
                option_symbol=leg.option_symbol,
            )
        )
    return out
