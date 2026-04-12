"""Portfolio correlation analysis."""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Optional


def compute_correlation_matrix(price_data: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute pairwise correlation of returns from a dict of price series."""
    if not price_data or len(price_data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(price_data).pct_change().dropna()
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
