"""
risk/correlation.py — Portfolio correlation analysis and concentration monitoring.

Provides correlation matrix computation, heatmap visualisation, and automated
threshold-based alert checks for use by the scheduler.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_PERIODS = 20  # minimum trading-day observations for a reliable correlation matrix

# Alert thresholds (overridable by callers)
CORR_ALERT_AVG_THRESHOLD: float = 0.7
POSITION_WEIGHT_THRESHOLD: float = 0.25
SECTOR_WEIGHT_THRESHOLD: float = 0.40


@dataclass
class CorrelationAlert:
    """A fired threshold breach from the correlation/concentration monitor."""

    alert_type: str    # 'avg_correlation' | 'position_concentration' | 'sector_concentration'
    value: float       # measured value
    threshold: float   # threshold that was breached
    message: str       # human-readable description
    ticker: str = ""   # affected ticker (if applicable)


def rolling_correlation(
    price_data: dict[str, pd.Series], window: int = 20
) -> pd.DataFrame:
    """
    Compute rolling pairwise correlations and return the most recent snapshot.

    Parameters
    ----------
    price_data : dict of {ticker: price Series}
    window     : rolling window in trading days (default 20)

    Returns
    -------
    Square DataFrame of pairwise correlations at the most recent window endpoint.
    """
    if not price_data or len(price_data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(price_data).pct_change().dropna()
    if len(df) < _MIN_PERIODS:
        logger.warning(
            "rolling_correlation skipped: only %d periods available (min %d).",
            len(df), _MIN_PERIODS,
        )
        return pd.DataFrame()
    if len(df) < window:
        return df.corr()
    return df.rolling(window).corr().iloc[-len(price_data):]


def check_correlation_alerts(
    price_data: dict[str, pd.Series],
    positions: dict[str, float],
    sector_map: dict[str, str] | None = None,
    avg_corr_threshold: float = CORR_ALERT_AVG_THRESHOLD,
    position_weight_threshold: float = POSITION_WEIGHT_THRESHOLD,
    sector_weight_threshold: float = SECTOR_WEIGHT_THRESHOLD,
) -> list[CorrelationAlert]:
    """
    Evaluate correlation and concentration thresholds, returning a list of alerts.

    Parameters
    ----------
    price_data               : dict of {ticker: price Series} for correlation calc
    positions                : dict of {ticker: market_value} for concentration calc
    sector_map               : optional dict of {ticker: sector_name}
    avg_corr_threshold       : alert if mean pairwise correlation > this value
    position_weight_threshold: alert if any position > this fraction of portfolio
    sector_weight_threshold  : alert if any sector > this fraction of portfolio

    Returns
    -------
    List of CorrelationAlert objects for each breached threshold.
    """
    alerts: list[CorrelationAlert] = []
    total_nav = sum(positions.values()) if positions else 0.0

    # 1. Average pairwise correlation
    if len(price_data) >= 2:
        corr_matrix = compute_correlation_matrix(price_data)
        if not corr_matrix.empty:
            # Exclude diagonal (self-correlation = 1.0)
            mask = ~pd.DataFrame(
                index=corr_matrix.index, columns=corr_matrix.columns
            ).apply(lambda col: col.index == col.name, axis=0)
            off_diag = corr_matrix.where(mask)
            avg_corr = float(off_diag.stack().mean())
            if avg_corr > avg_corr_threshold:
                alerts.append(CorrelationAlert(
                    alert_type="avg_correlation",
                    value=round(avg_corr, 4),
                    threshold=avg_corr_threshold,
                    message=(
                        f"Portfolio average pairwise correlation is {avg_corr:.2f}, "
                        f"exceeding threshold of {avg_corr_threshold:.2f}. "
                        "Consider diversifying across uncorrelated assets."
                    ),
                ))

    # 2. Position concentration
    if total_nav > 0:
        for ticker, mv in positions.items():
            weight = mv / total_nav
            if weight > position_weight_threshold:
                alerts.append(CorrelationAlert(
                    alert_type="position_concentration",
                    value=round(weight, 4),
                    threshold=position_weight_threshold,
                    ticker=ticker,
                    message=(
                        f"{ticker} represents {weight:.1%} of portfolio NAV "
                        f"(threshold: {position_weight_threshold:.0%}). "
                        "Consider trimming or hedging."
                    ),
                ))

    # 3. Sector concentration
    if sector_map and total_nav > 0:
        sector_values: dict[str, float] = {}
        for ticker, mv in positions.items():
            sector = sector_map.get(ticker, "Unknown")
            sector_values[sector] = sector_values.get(sector, 0.0) + mv
        for sector, sv in sector_values.items():
            weight = sv / total_nav
            if weight > sector_weight_threshold:
                alerts.append(CorrelationAlert(
                    alert_type="sector_concentration",
                    value=round(weight, 4),
                    threshold=sector_weight_threshold,
                    ticker=sector,
                    message=(
                        f"Sector '{sector}' is {weight:.1%} of portfolio NAV "
                        f"(threshold: {sector_weight_threshold:.0%}). "
                        "Reduce sector exposure."
                    ),
                ))

    return alerts


def compute_correlation_matrix(price_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise correlation of returns from a dict of price series.

    Returns an empty DataFrame when fewer than _MIN_PERIODS observations are
    available, since the matrix would be statistically unreliable.
    """
    if not price_data or len(price_data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(price_data).pct_change().dropna()
    if len(df) < _MIN_PERIODS:
        logger.warning(
            "Correlation matrix skipped: only %d periods available (min %d). "
            "Results would be unreliable.",
            len(df), _MIN_PERIODS,
        )
        return pd.DataFrame()
    return df.corr()


def build_heatmap(corr_matrix: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    """Build a Plotly heatmap from a correlation DataFrame."""
    if corr_matrix.empty:
        return go.Figure()
    labels = list(corr_matrix.columns)
    z = corr_matrix.values.tolist()
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        title=title,
        xaxis_nticks=len(labels),
        yaxis_nticks=len(labels),
        height=max(400, len(labels) * 60),
    )
    return fig
