"""
strategies/gnn_signal.py — Graph Neural Network cross-asset momentum signal.

Uses a 2-layer Graph Attention Network (GAT) with GICS sector co-membership as
graph edges. Node features: technical indicators + regime one-hot + sentiment.
Output: graph-aware momentum score per ticker in [-1.0, 1.0].

Requires (optional): pip install torch torch-geometric

ENV vars
--------
    GNN_ENABLED         set to '1' to activate in screener (default: 0)
    GNN_MODEL_PATH      path to saved GAT checkpoint (default: models/gnn_signal.pt)
    GNN_HIDDEN_DIM      hidden layer dimension (default: 32)
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

_MODEL_PATH = os.environ.get("GNN_MODEL_PATH", "models/gnn_signal.pt")
_HIDDEN_DIM = int(os.environ.get("GNN_HIDDEN_DIM", "32"))

# ── GICS sector mapping for the platform's default 30-ticker universe ─────────

TICKER_SECTOR: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials",
    "V": "Financials", "MA": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "HON": "Industrials", "BA": "Industrials",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "WMT": "Consumer Staples",
    "NEE": "Utilities",
    "AMT": "Real Estate",
    "SPY": "ETF", "QQQ": "ETF",
}


def build_sector_adjacency(tickers: list[str]) -> np.ndarray:
    """
    Build a binary adjacency matrix where 1 = same GICS sector.

    Parameters
    ----------
    tickers : list of ticker symbols

    Returns
    -------
    np.ndarray of shape (N, N), dtype float32, diagonal = 0.
    """
    n = len(tickers)
    adj = np.zeros((n, n), dtype=np.float32)
    sectors = [TICKER_SECTOR.get(t, "Unknown") for t in tickers]
    for i in range(n):
        for j in range(n):
            if i != j and sectors[i] == sectors[j] and sectors[i] != "Unknown":
                adj[i, j] = 1.0
    return adj


# ── Node feature builder ─────────────────────────────────────────────────────

def build_node_features(
    tickers: list[str],
    indicators: dict[str, dict] | None = None,
    regime: str = "trending_bull",
    sentiments: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Build a node feature matrix of shape (N, feature_dim).

    Feature vector per node (ticker):
        [rsi/100, momentum_5d, sma_cross_signal, regime_oh(4), sentiment]
        = 8 features

    Parameters
    ----------
    tickers    : ordered list of tickers (defines node ordering)
    indicators : dict of {ticker: {rsi, momentum, sma_signal}} — fetched if None
    regime     : current market regime label
    sentiments : dict of {ticker: sentiment_score} — defaults to 0.0

    Returns
    -------
    np.ndarray of shape (N, 8), dtype float32
    """
    from analysis.regime import REGIME_STATES

    regime_oh = [1.0 if r == regime else 0.0 for r in REGIME_STATES]
    features = []

    for ticker in tickers:
        ind = (indicators or {}).get(ticker, {})
        rsi = float(ind.get("rsi", 50)) / 100.0
        momentum = float(ind.get("momentum", 0.0))
        sma_signal = float(ind.get("sma_signal", 0.0))
        sentiment = float((sentiments or {}).get(ticker, 0.0))
        node_feat = [rsi, momentum, sma_signal] + regime_oh + [sentiment]
        features.append(node_feat)

    return np.array(features, dtype=np.float32)


# ── GAT model (torch-geometric) ──────────────────────────────────────────────

def _build_gat_model(in_channels: int, hidden_dim: int, out_channels: int = 1):
    """Build a 2-layer GAT model. Raises ImportError if torch/torch_geometric absent."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATConv  # type: ignore[import]

    class GATSignalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(in_channels, hidden_dim, heads=2, concat=True, dropout=0.1)
            self.conv2 = GATConv(hidden_dim * 2, out_channels, heads=1, concat=False, dropout=0.1)

        def forward(self, x, edge_index):
            import torch.nn.functional as F
            x = F.elu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return torch.tanh(x)  # output in [-1, 1]

    return GATSignalModel()


def _adj_to_edge_index(adj: np.ndarray):
    """Convert dense adjacency matrix to COO edge_index for torch-geometric."""
    import torch
    src, dst = np.where(adj > 0)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


# ── Public API ────────────────────────────────────────────────────────────────

class GNNSignal:
    """
    Graph-aware momentum scorer using a 2-layer GAT.

    When torch/torch_geometric are not installed or the model checkpoint is
    absent, falls back to a simple RSI + momentum composite score.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model_path = model_path or _MODEL_PATH
        self._model = None
        self._load_if_available()

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists():
            logger.info("GNNSignal: no checkpoint at %s, using fallback scorer", path)
            return
        try:
            import torch
            in_channels = 8
            model = _build_gat_model(in_channels, _HIDDEN_DIM)
            model.load_state_dict(torch.load(str(path), map_location="cpu"))
            model.eval()
            self._model = model
            logger.info("GNNSignal: loaded checkpoint from %s", path)
        except ImportError:
            logger.info("torch/torch-geometric not installed; GNN using fallback scorer")
        except Exception as exc:
            logger.warning("GNNSignal: failed to load checkpoint: %s", exc)

    def score(
        self,
        tickers: list[str],
        indicators: dict[str, dict] | None = None,
        regime: str = "trending_bull",
        sentiments: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Return a momentum score in [-1.0, 1.0] for each ticker.

        Parameters
        ----------
        tickers    : list of tickers to score
        indicators : {ticker: {rsi, momentum, sma_signal}} — fetched if None
        regime     : current regime label
        sentiments : {ticker: score} — defaults to 0.0

        Returns
        -------
        dict of {ticker: score} where score ∈ [-1.0, 1.0]
        """
        if not tickers:
            return {}

        if self._model is not None:
            return self._score_with_gat(tickers, indicators, regime, sentiments)
        return self._fallback_score(tickers, indicators, sentiments)

    def _score_with_gat(
        self,
        tickers: list[str],
        indicators: dict[str, dict] | None,
        regime: str,
        sentiments: dict[str, float] | None,
    ) -> dict[str, float]:
        import torch
        x_np = build_node_features(tickers, indicators, regime, sentiments)
        adj = build_sector_adjacency(tickers)
        edge_index = _adj_to_edge_index(adj)
        x = torch.tensor(x_np)
        with torch.no_grad():
            out = self._model(x, edge_index).squeeze(-1).numpy()
        return {ticker: float(out[i]) for i, ticker in enumerate(tickers)}

    def _fallback_score(
        self,
        tickers: list[str],
        indicators: dict[str, dict] | None,
        sentiments: dict[str, float] | None,
    ) -> dict[str, float]:
        """Simple RSI + momentum + sentiment composite when GNN is unavailable."""
        scores = {}
        for ticker in tickers:
            ind = (indicators or {}).get(ticker, {})
            rsi = float(ind.get("rsi", 50))
            momentum = float(ind.get("momentum", 0.0))
            sentiment = float((sentiments or {}).get(ticker, 0.0))
            # Normalise RSI to [-1, 1]: RSI=30 → -0.4, RSI=70 → +0.4
            rsi_score = (rsi - 50) / 50.0
            score = 0.5 * rsi_score + 0.3 * momentum + 0.2 * sentiment
            scores[ticker] = float(np.clip(score, -1.0, 1.0))
        return scores
