"""
analysis/triple_barrier.py — López de Prado triple-barrier labeling.

Generates path-dependent labels for supervised ML: for every event timestamp,
three barriers are simultaneously active — a profit-taking target (upper),
a stop-loss target (lower), and a vertical time barrier. The first barrier
touched determines the label:

    +1   profit-take barrier hit first  (meaningful up-move)
    -1   stop-loss barrier hit first    (meaningful down-move)
     0   vertical barrier hit first     (time-out, neither move materialised)

Compared with fixed-horizon forward returns this method:

  * adapts barrier widths to each event's volatility regime
  * respects the path of returns, not just the endpoint
  * produces ternary labels suitable for classification models

Reference
---------
    López de Prado, *Advances in Financial Machine Learning*, Chapter 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def daily_volatility(prices: pd.Series, span: int = 20) -> pd.Series:
    """Exponentially-weighted daily return volatility (EWM std with the given span).

    Returns a Series aligned to ``prices`` with the first observation NaN.
    """
    returns = prices.pct_change()
    return returns.ewm(span=span, adjust=False).std()


def _vertical_barriers(
    prices: pd.Series,
    events: pd.DatetimeIndex,
    num_days: int,
) -> pd.Series:
    """Map each event index to the timestamp ``num_days`` bars later (clipped)."""
    locs = prices.index.searchsorted(events + pd.Timedelta(days=num_days))
    locs = np.clip(locs, 0, len(prices) - 1)
    return pd.Series(prices.index[locs], index=events)


def triple_barrier_labels(
    prices: pd.Series,
    events: pd.DatetimeIndex | None = None,
    pt_sl: tuple[float, float] = (1.0, 1.0),
    num_days: int = 5,
    vol_span: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Compute triple-barrier labels for each event timestamp.

    Parameters
    ----------
    prices    : pandas Series of close prices indexed by datetime (ascending).
    events    : DatetimeIndex of bar timestamps where a signal fires. Defaults
                to every index of *prices*.
    pt_sl     : (profit_take, stop_loss) multipliers expressed in units of
                daily volatility. Either side may be 0 to disable that barrier.
    num_days  : vertical barrier horizon (days).
    vol_span  : EWM span used when estimating per-event volatility.
    min_ret   : events whose volatility drops below this threshold are
                dropped (noise filter). ``0`` disables filtering.

    Returns
    -------
    DataFrame indexed by event timestamp with columns:
        * ``t1``     : timestamp of the first barrier touched
        * ``ret``    : realised return from event → first-touch
        * ``bin``    : label in {-1, 0, +1}
        * ``target`` : volatility used to scale barriers
    """
    if prices.empty:
        return pd.DataFrame(columns=["t1", "ret", "bin", "target"])
    if not prices.index.is_monotonic_increasing:
        prices = prices.sort_index()

    evts = pd.DatetimeIndex(events) if events is not None else prices.index

    target = daily_volatility(prices, span=vol_span).reindex(evts).ffill()
    if min_ret > 0.0:
        keep = target.fillna(0.0) > min_ret
        evts = evts[keep.values]
        target = target.loc[evts]
    if len(evts) == 0:
        return pd.DataFrame(columns=["t1", "ret", "bin", "target"])

    verticals = _vertical_barriers(prices, evts, num_days)

    pt_mult, sl_mult = pt_sl
    records = []
    for ts in evts:
        if ts not in prices.index:
            continue
        t1 = verticals.loc[ts]
        window = prices.loc[ts:t1]
        if len(window) < 2:
            records.append({"t1": t1, "ret": 0.0, "bin": 0, "target": float(target.loc[ts])})
            continue

        entry = float(window.iloc[0])
        vol = float(target.loc[ts]) if pd.notna(target.loc[ts]) else 0.0
        path_ret = window / entry - 1.0

        pt_hit: pd.Timestamp | None = None
        sl_hit: pd.Timestamp | None = None
        if pt_mult > 0 and vol > 0:
            touches = path_ret[path_ret >= pt_mult * vol]
            if not touches.empty:
                pt_hit = touches.index[0]
        if sl_mult > 0 and vol > 0:
            touches = path_ret[path_ret <= -sl_mult * vol]
            if not touches.empty:
                sl_hit = touches.index[0]

        candidates = {k: v for k, v in {"pt": pt_hit, "sl": sl_hit, "t1": t1}.items()
                      if v is not None}
        first_key = min(candidates, key=lambda k: candidates[k])
        first_ts = candidates[first_key]
        realised = float(path_ret.loc[first_ts])

        if first_key == "pt":
            label = 1
        elif first_key == "sl":
            label = -1
        else:
            # Time-out: sign of realised return if significant, else 0.
            label = 0 if abs(realised) < 1e-9 else int(np.sign(realised))

        records.append({"t1": first_ts, "ret": realised, "bin": label, "target": vol})

    return pd.DataFrame(records, index=evts[: len(records)])
