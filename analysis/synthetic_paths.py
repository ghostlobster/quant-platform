"""
analysis/synthetic_paths.py — Minimal GAN for synthetic return time series.

Implements a small adversarial generator inspired by the TimeGAN
sketch in Jansen, *Machine Learning for Algorithmic Trading*
(2nd ed.) Chapter 21. The full TimeGAN includes supervised + recovery
heads in addition to the adversarial loss; this module ships only the
adversarial pair (LSTM generator + LSTM discriminator) so the moving
parts stay readable and the training loop fits in a single test.

The intended consumer is :mod:`backtester.monte_carlo` (which already
exposes a bootstrap-based path generator); ``sample_paths`` here gives
backtests an alternative source of synthetic stress paths whose
auto-correlation structure is learned, not block-bootstrapped.

torch is gated through the same try/except pattern used by
:mod:`strategies.dl_signal`.

Public surface
--------------
``GeneratorResult``      — dataclass bundling the trained generator
``train_generator``      — fit on a 1-D return series
``sample_paths``         — draw fresh ``(n_paths, horizon)`` paths

Reference
---------
    Jansen, *Machine Learning for Algorithmic Trading* (2nd ed.) Ch 21.4.
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

    class _LSTMGenerator(nn.Module):  # type: ignore[misc]
        """Maps Gaussian noise sequences to scalar return paths."""

        def __init__(self, latent_dim: int, hidden: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(latent_dim, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, z):  # type: ignore[override]
            out, _ = self.lstm(z)
            return self.head(out).squeeze(-1)  # (batch, horizon)

    class _LSTMDiscriminator(nn.Module):  # type: ignore[misc]
        """Scores a return path as real (→ 1) or synthetic (→ 0)."""

        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):  # type: ignore[override]
            out, _ = self.lstm(x.unsqueeze(-1))
            return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)
else:
    _LSTMGenerator = None  # type: ignore[assignment]
    _LSTMDiscriminator = None  # type: ignore[assignment]


@dataclass(frozen=True)
class GeneratorResult:
    """Trained generator artefacts."""

    generator: object
    latent_dim: int
    horizon: int
    train_mean: float
    train_std: float


def _windows(returns: np.ndarray, horizon: int) -> np.ndarray:
    if len(returns) < horizon:
        raise ValueError(f"need at least {horizon} returns, got {len(returns)}")
    n = len(returns) - horizon + 1
    out = np.stack([returns[i : i + horizon] for i in range(n)], axis=0)
    return out.astype(np.float32)


def train_generator(
    returns: pd.Series | np.ndarray,
    horizon: int = 20,
    latent_dim: int = 8,
    hidden: int = 16,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
) -> GeneratorResult:
    """Fit a tiny LSTM-GAN on rolling windows of ``returns``.

    Raises
    ------
    RuntimeError
        If torch is not installed.
    ValueError
        If ``returns`` is shorter than ``horizon``.
    """
    _require_torch()
    arr = np.asarray(pd.Series(returns).dropna().to_numpy(dtype=float))
    if arr.size < horizon:
        raise ValueError(f"train_generator: need ≥{horizon} returns, got {arr.size}")

    mean = float(arr.mean())
    std = float(arr.std()) or 1.0
    standardised = (arr - mean) / std

    real_windows = _windows(standardised, horizon)

    torch.manual_seed(int(seed))
    G = _LSTMGenerator(latent_dim=int(latent_dim), hidden=int(hidden))
    D = _LSTMDiscriminator(hidden=int(hidden))
    g_opt = torch.optim.Adam(G.parameters(), lr=lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr)
    bce = nn.BCELoss()

    real_t = torch.from_numpy(real_windows)
    n = len(real_t)
    for _epoch in range(int(epochs)):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            real_batch = real_t[idx]
            bs = real_batch.shape[0]

            # Discriminator step
            d_opt.zero_grad()
            z = torch.randn(bs, horizon, latent_dim)
            fake = G(z).detach()
            d_real = D(real_batch)
            d_fake = D(fake)
            d_loss = bce(d_real, torch.ones(bs)) + bce(d_fake, torch.zeros(bs))
            d_loss.backward()
            d_opt.step()

            # Generator step
            g_opt.zero_grad()
            z = torch.randn(bs, horizon, latent_dim)
            fake = G(z)
            g_loss = bce(D(fake), torch.ones(bs))
            g_loss.backward()
            g_opt.step()

    G.eval()
    log.info(
        "synthetic_paths.train complete",
        n_windows=int(n), horizon=horizon, latent_dim=latent_dim,
    )
    return GeneratorResult(
        generator=G, latent_dim=int(latent_dim), horizon=int(horizon),
        train_mean=mean, train_std=std,
    )


def sample_paths(
    result: GeneratorResult,
    n_paths: int,
    horizon: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Draw ``n_paths`` synthetic return paths.

    Parameters
    ----------
    result :
        Output of :func:`train_generator`.
    n_paths :
        Number of paths to draw.
    horizon :
        Length of each path. Defaults to the generator's training
        horizon; callers may shorten freely (LSTM unrolls to any length).
    seed :
        Optional seed for reproducible noise.

    Returns
    -------
    np.ndarray
        ``(n_paths, horizon)`` array of returns on the original
        (un-standardised) scale.
    """
    _require_torch()
    h = int(horizon if horizon is not None else result.horizon)
    if seed is not None:
        torch.manual_seed(int(seed))

    z = torch.randn(int(n_paths), h, result.latent_dim)
    with torch.no_grad():
        synth = result.generator(z).cpu().numpy()  # type: ignore[union-attr]
    return synth * result.train_std + result.train_mean
