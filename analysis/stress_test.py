"""
analysis/stress_test.py — Portfolio stress testing with historical crisis scenarios.

Applies percentage shocks to an open portfolio and computes projected NAV impact.
Optionally calls the LLM provider to generate plausible adverse scenarios.

ENV vars
--------
    LLM_STRESS_SCENARIOS    set to '1' to enable LLM scenario generation (default: 0)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StressScenario:
    """Description of a historical or synthetic stress scenario."""

    name: str
    description: str
    equity_shock: float      # fractional change, e.g. -0.52 = -52%
    vol_multiplier: float = 1.0
    rates_bps: float = 0.0   # interest rate change in basis points (informational)


@dataclass
class StressResult:
    """Outcome of applying a scenario to the current portfolio."""

    scenario_name: str
    description: str
    pre_nav: float
    post_nav: float
    nav_change: float          # absolute dollar change
    nav_change_pct: float      # percentage change, e.g. -0.52
    worst_position: str        # ticker with largest absolute loss
    position_impacts: dict[str, float] = field(default_factory=dict)  # ticker → dollar impact


# ── Pre-built historical scenarios ────────────────────────────────────────────

HISTORICAL_SCENARIOS: list[StressScenario] = [
    StressScenario(
        name="2008_gfc",
        description="2008 Global Financial Crisis — peak-to-trough S&P 500 drawdown of ~55%.",
        equity_shock=-0.52,
        vol_multiplier=4.0,
        rates_bps=-200,
    ),
    StressScenario(
        name="2020_covid",
        description="2020 COVID-19 crash — S&P 500 fell ~34% in 33 days (Feb–Mar 2020).",
        equity_shock=-0.34,
        vol_multiplier=5.0,
        rates_bps=-150,
    ),
    StressScenario(
        name="2022_rate_hike",
        description="2022 Fed rate-hike cycle — S&P 500 -25%, bonds -15%, VIX doubled.",
        equity_shock=-0.25,
        vol_multiplier=2.0,
        rates_bps=450,
    ),
]


# ── Core logic ────────────────────────────────────────────────────────────────

def apply_scenario(portfolio_df: pd.DataFrame, scenario: StressScenario) -> StressResult:
    """
    Apply a stress scenario to the portfolio and return projected impact.

    Parameters
    ----------
    portfolio_df : DataFrame with at minimum columns 'Ticker' and 'Market Value'.
                   (matches the output of broker/paper_trader.get_portfolio())
    scenario     : StressScenario to apply

    Returns
    -------
    StressResult with pre/post NAV and per-position impacts.
    """
    if portfolio_df.empty:
        return StressResult(
            scenario_name=scenario.name,
            description=scenario.description,
            pre_nav=0.0,
            post_nav=0.0,
            nav_change=0.0,
            nav_change_pct=0.0,
            worst_position="N/A",
            position_impacts={},
        )

    # Normalise column name variations
    mv_col = next(
        (c for c in portfolio_df.columns if c.lower().replace(" ", "_") == "market_value"),
        None,
    )
    ticker_col = next(
        (c for c in portfolio_df.columns if c.lower() == "ticker"),
        None,
    )
    if mv_col is None or ticker_col is None:
        raise ValueError(
            "portfolio_df must contain 'Ticker' and 'Market Value' columns"
        )

    pre_nav = float(portfolio_df[mv_col].fillna(0).sum())
    impacts: dict[str, float] = {}

    for _, row in portfolio_df.iterrows():
        ticker = str(row[ticker_col])
        mv = float(row[mv_col] or 0)
        dollar_impact = mv * scenario.equity_shock
        impacts[ticker] = dollar_impact

    total_impact = sum(impacts.values())
    post_nav = pre_nav + total_impact
    nav_change_pct = total_impact / pre_nav if pre_nav != 0 else 0.0

    worst = min(impacts, key=impacts.get) if impacts else "N/A"

    return StressResult(
        scenario_name=scenario.name,
        description=scenario.description,
        pre_nav=round(pre_nav, 2),
        post_nav=round(post_nav, 2),
        nav_change=round(total_impact, 2),
        nav_change_pct=round(nav_change_pct, 4),
        worst_position=worst,
        position_impacts={k: round(v, 2) for k, v in impacts.items()},
    )


def apply_custom_shock(
    portfolio_df: pd.DataFrame,
    equity_pct: float,
    vol_mult: float = 1.0,
    name: str = "custom",
) -> StressResult:
    """
    Apply a custom equity shock percentage to the portfolio.

    Parameters
    ----------
    portfolio_df : portfolio DataFrame (same format as apply_scenario)
    equity_pct   : shock as a fraction, e.g. -0.30 = -30%, 0.10 = +10%
    vol_mult     : volatility multiplier (informational only)
    name         : label for the result
    """
    scenario = StressScenario(
        name=name,
        description=f"Custom shock: equity {equity_pct:+.1%}, vol ×{vol_mult:.1f}",
        equity_shock=equity_pct,
        vol_multiplier=vol_mult,
    )
    return apply_scenario(portfolio_df, scenario)


def run_stress_tests(
    portfolio_df: pd.DataFrame,
    scenarios: list[StressScenario] | None = None,
) -> list[StressResult]:
    """
    Run a list of stress scenarios against the portfolio.

    Parameters
    ----------
    portfolio_df : portfolio DataFrame
    scenarios    : list of StressScenario objects; defaults to HISTORICAL_SCENARIOS

    Returns
    -------
    List of StressResult objects, one per scenario.
    """
    to_run = scenarios if scenarios is not None else HISTORICAL_SCENARIOS
    results = []
    for scenario in to_run:
        try:
            results.append(apply_scenario(portfolio_df, scenario))
        except Exception as exc:
            logger.warning("Stress test failed for %s: %s", scenario.name, exc)
    return results


def generate_llm_scenarios(
    portfolio_summary: str,
    regime: str = "unknown",
    n_scenarios: int = 3,
) -> list[StressScenario]:
    """
    Ask the LLM to generate plausible adverse scenarios for the current regime.

    Returns an empty list when LLM_STRESS_SCENARIOS != '1' or when the LLM
    is unavailable.

    Parameters
    ----------
    portfolio_summary : human-readable summary of portfolio holdings
    regime            : current market regime label
    n_scenarios       : number of scenarios to request (default 3)
    """
    if os.environ.get("LLM_STRESS_SCENARIOS", "0") != "1":
        return []

    prompt = (
        f"You are a risk manager. The current market regime is '{regime}'.\n"
        f"Portfolio summary:\n{portfolio_summary}\n\n"
        f"Generate {n_scenarios} plausible adverse scenarios for this portfolio. "
        "For each scenario respond with a JSON object with keys: "
        '"name" (short slug), "description" (1 sentence), '
        '"equity_shock" (float, e.g. -0.30 for -30%), '
        '"vol_multiplier" (float ≥ 1.0), "rates_bps" (int).\n'
        f"Return a JSON array of exactly {n_scenarios} objects. No other text."
    )

    try:
        from providers.llm import get_llm
        import json
        llm = get_llm()
        raw = llm.complete(prompt)
        # Strip markdown code fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        items = json.loads(text)
        scenarios = []
        for item in items:
            scenarios.append(StressScenario(
                name=str(item.get("name", "llm_scenario")),
                description=str(item.get("description", "")),
                equity_shock=float(item.get("equity_shock", -0.20)),
                vol_multiplier=float(item.get("vol_multiplier", 1.5)),
                rates_bps=float(item.get("rates_bps", 0)),
            ))
        logger.info("LLM generated %d stress scenarios", len(scenarios))
        return scenarios
    except Exception as exc:
        logger.warning("LLM scenario generation failed: %s", exc)
        return []
