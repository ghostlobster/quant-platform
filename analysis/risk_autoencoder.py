"""
analysis/risk_autoencoder.py — Autoencoder-based latent risk factors.

Implements the conditional-autoencoder pipeline from Jansen,
*Machine Learning for Algorithmic Trading* (2nd ed.) Chapter 20: fit a
small fully-connected autoencoder over a panel of returns, treat the
bottleneck activations as latent risk factors (an ML-native sibling of
PCA), and expose reconstruction error as a stress signal.

torch is gated through the same try/except pattern used by
:mod:`strategies.dl_signal`.

Public surface
--------------
``AEResult``               — dataclass bundling the trained model
``train_autoencoder``      — fit on a returns DataFrame
``latent_factors``         — per-row latent activations (DataFrame)
``reconstruction_error``   — per-row L2 distance, indexed like input

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 20.5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger(__name__)

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Run: pip install 'torch>=2.0.0'"
        )


if _TORCH_AVAILABLE:

    class _Autoencoder(nn.Module):  # type: ignore[misc]
        """Fully-connected encoder/decoder with a single hidden layer per side."""

        def __init__(self, n_features: int, latent_dim: int, hidden: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_features),
            )

        def forward(self, x):  # type: ignore[override]
            z = self.encoder(x)
            return self.decoder(z), z
else:
    _Autoencoder = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AEResult:
    """Trained autoencoder artefacts."""

    model: object
    latent_dim: int
    feature_names: tuple[str, ...]


def _to_tensor(panel: pd.DataFrame) -> "torch.Tensor":  # type: ignore[name-defined]
    return torch.from_numpy(panel.fillna(0.0).to_numpy(dtype=np.float32))


def train_autoencoder(
    returns: pd.DataFrame,
    latent_dim: int = 4,
    hidden: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> AEResult:
    """Fit an autoencoder on ``returns`` (rows = dates, cols = tickers).

    Raises
    ------
    RuntimeError
        If torch is not installed.
    ValueError
        If ``returns`` is empty or has fewer columns than ``latent_dim``.
    """
    _require_torch()
    if returns.empty:
        raise ValueError("train_autoencoder: empty returns panel")
    n_features = returns.shape[1]
    if n_features < latent_dim:
        raise ValueError(
            f"train_autoencoder: latent_dim ({latent_dim}) exceeds n_features ({n_features})"
        )

    torch.manual_seed(int(seed))
    model = _Autoencoder(n_features=n_features, latent_dim=int(latent_dim), hidden=int(hidden))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = _to_tensor(returns)
    model.train()
    for _epoch in range(int(epochs)):
        opt.zero_grad()
        recon, _ = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
    model.eval()

    log.info(
        "risk_autoencoder.train complete",
        n_dates=len(returns), n_features=n_features,
        latent_dim=latent_dim, final_loss=float(loss.item()),
    )
    return AEResult(
        model=model,
        latent_dim=int(latent_dim),
        feature_names=tuple(str(c) for c in returns.columns),
    )


def latent_factors(result: AEResult, returns: pd.DataFrame) -> pd.DataFrame:
    """Per-row encoder activations as a ``(n_rows, latent_dim)`` DataFrame."""
    _require_torch()
    aligned = returns.reindex(columns=list(result.feature_names))
    X = _to_tensor(aligned)
    with torch.no_grad():
        _, z = result.model(X)  # type: ignore[union-attr]
    columns = [f"f{i+1}" for i in range(result.latent_dim)]
    return pd.DataFrame(z.cpu().numpy(), index=aligned.index, columns=columns)


def reconstruction_error(result: AEResult, returns: pd.DataFrame) -> pd.Series:
    """Per-row L2 reconstruction distance.

    Useful as a regime-shift / outlier signal: rows whose error spikes
    above a recent rolling baseline indicate the panel has moved into a
    regime the autoencoder hasn't seen.
    """
    _require_torch()
    aligned = returns.reindex(columns=list(result.feature_names))
    X = _to_tensor(aligned)
    with torch.no_grad():
        recon, _ = result.model(X)  # type: ignore[union-attr]
    err = torch.norm(recon - X, dim=1).cpu().numpy()
    return pd.Series(err, index=aligned.index, name="ae_recon_err")
