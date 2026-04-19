"""
strategies/cnn_signal.py — CNN over OHLC chart images (Jansen Ch 18).

Predicts ``fwd_ret_5d`` from a small Conv2d head over Gramian Angular
Field images of the rolling close-price window.  Same public surface
as :class:`strategies.linear_signal.LinearSignal` /
:class:`strategies.ml_signal.MLSignal`: ``train()``, ``predict()``,
plus a fallback to momentum when the model isn't available.

Torch is gated through the same try/except pattern used by
:mod:`strategies.dl_signal`, so this module imports cleanly even on
machines without torch installed.

ENV vars
--------
    CNN_ALPHA_MODEL_PATH    path to model checkpoint (default: models/cnn_alpha.pt)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.chart_images import to_gramian_angular_field
from analysis.factor_ic import _spearman_corr
from data.features import build_feature_matrix
from data.fetcher import fetch_ohlcv
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


_DEFAULT_MODEL_PATH = os.environ.get("CNN_ALPHA_MODEL_PATH", "models/cnn_alpha.pt")
_TARGET_COL = "fwd_ret_5d"
_DEFAULT_WINDOW = 16

from agents.knowledge_registry import ModelEntry  # noqa: E402

MODEL_ENTRY = ModelEntry(
    name="cnn_alpha",
    artefact_env="CNN_ALPHA_MODEL_PATH",
    artefact_default="models/cnn_alpha.pt",
    metadata_name="cnn_alpha",
)


if _TORCH_AVAILABLE:

    class _ChartCNN(nn.Module):  # type: ignore[misc]
        """Tiny 2-conv CNN reading a single-channel ``(window, window)`` GAF."""

        def __init__(self, window: int = _DEFAULT_WINDOW) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x):  # type: ignore[override]
            return self.head(self.features(x)).squeeze(-1)
else:
    _ChartCNN = None  # type: ignore[assignment]


def _build_image_dataset(
    fm: pd.DataFrame,
    closes_by_ticker: dict[str, pd.Series],
    window: int,
) -> tuple[np.ndarray | None, np.ndarray | None, pd.MultiIndex | None]:
    """Build aligned ``(N, 1, window, window)`` image tensor + targets.

    For each row of ``fm`` we look up the trailing ``window`` close
    prices for that ticker and convert to a GAF image. Rows whose
    trailing window is not yet available, or whose target is NaN, are
    dropped.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    idx_rows: list[tuple] = []

    for (date, ticker), row in fm.iterrows():
        target = float(row.get(_TARGET_COL, np.nan))
        if not np.isfinite(target):
            continue
        closes = closes_by_ticker.get(ticker)
        if closes is None or len(closes) < window:
            continue
        sub = closes.loc[:date].iloc[-window:]
        if len(sub) < window:
            continue
        img = to_gramian_angular_field(sub, window=window)
        xs.append(img[None, ...])  # add channel axis
        ys.append(target)
        idx_rows.append((date, ticker))

    if not xs:
        return None, None, None
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return X, y, pd.MultiIndex.from_tuples(idx_rows, names=["date", "ticker"])


class CNNSignal:
    """CNN-on-chart-image cross-sectional alpha model."""

    def __init__(
        self,
        model_path: str | None = None,
        window: int = _DEFAULT_WINDOW,
    ) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._window: int = int(window)
        self._model = None  # _ChartCNN | None
        self._load_if_available()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists() or not _TORCH_AVAILABLE:
            return
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = _ChartCNN(window=self._window)
            model.load_state_dict(state)
            model.eval()
            self._model = model
            log.info("cnn_signal: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning("cnn_signal: failed to load checkpoint", path=str(path), error=str(exc))

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 64,
    ) -> dict[str, float]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. Run: pip install 'torch>=2.0.0'"
            )

        log.info("cnn_signal: building feature matrix", tickers=len(tickers), period=period)
        fm = build_feature_matrix(tickers, period=period)
        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")
        fm = fm.dropna(subset=[_TARGET_COL])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{_TARGET_COL}'")

        closes = {t: fetch_ohlcv(t, period)["Close"].astype(float)
                  for t in tickers if fetch_ohlcv(t, period) is not None}

        X, y, idx = _build_image_dataset(fm, closes, window=self._window)
        if X is None:
            raise ValueError("cnn_signal: no usable image samples")

        all_dates = sorted({d for d, _ in idx})
        split = int(len(all_dates) * (1 - test_size))
        train_mask = np.array([d in set(all_dates[:split]) for d, _ in idx])
        test_mask = ~train_mask

        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)

        model = _ChartCNN(window=self._window)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _epoch in range(int(epochs)):
            model.train()
            perm = torch.randperm(len(X_t))
            for start in range(0, len(perm), batch_size):
                batch = perm[start : start + batch_size]
                opt.zero_grad()
                pred = model(X_t[batch])
                loss = loss_fn(pred, y_t[batch])
                loss.backward()
                opt.step()

        model.eval()
        self._model = model
        with torch.no_grad():
            preds = model(X_t).cpu().numpy()
        train_ic = _spearman_corr(preds[train_mask], y[train_mask]) if train_mask.any() else 0.0
        test_ic = _spearman_corr(preds[test_mask], y[test_mask]) if test_mask.any() else 0.0

        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self._model_path)
        self._write_metadata(
            n_tickers=len(tickers), period=period,
            train_ic=train_ic, test_ic=test_ic,
        )
        return {
            "train_ic": float(train_ic),
            "test_ic": float(test_ic),
            "train_icir": float(train_ic),
            "test_icir": float(test_ic),
            "n_train_samples": int(train_mask.sum()),
            "n_test_samples": int(test_mask.sum()),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, tickers: list[str], period: str = "6mo") -> dict[str, float]:
        if not _TORCH_AVAILABLE or self._model is None:
            return self._momentum_fallback(tickers, period)

        try:
            scores: dict[str, float] = {}
            for ticker in tickers:
                df = fetch_ohlcv(ticker, period)
                if df is None or len(df) < self._window:
                    scores[ticker] = 0.0
                    continue
                img = to_gramian_angular_field(df["Close"], window=self._window)
                tensor = torch.from_numpy(img[None, None, ...].astype(np.float32))
                with torch.no_grad():
                    raw = float(self._model(tensor).cpu().item())
                scores[ticker] = float(np.clip(raw, -1.0, 1.0))
            return scores
        except Exception as exc:
            log.warning("cnn_signal.predict: error, falling back to momentum", error=str(exc))
            return self._momentum_fallback(tickers, period)

    @staticmethod
    def _momentum_fallback(tickers: list[str], period: str) -> dict[str, float]:
        from strategies.momentum import compute_momentum_score

        scores: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, period)
                if df is not None and not df.empty:
                    mom = compute_momentum_score(df)
                    last_val = mom.dropna().iloc[-1] if not mom.dropna().empty else 0.0
                    scores[ticker] = float(np.clip(last_val, -1.0, 1.0))
                else:
                    scores[ticker] = 0.0
            except Exception:
                scores[ticker] = 0.0
        return scores

    def _write_metadata(
        self,
        n_tickers: int,
        period: str,
        train_ic: float,
        test_ic: float,
    ) -> None:
        try:
            from data.db import get_connection
            conn = get_connection()
            with conn:
                conn.execute(
                    """
                    INSERT INTO model_metadata
                        (model_name, trained_at, train_ic, test_ic, n_tickers, period)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("cnn_alpha", time.time(), train_ic, test_ic, n_tickers, period),
                )
            conn.close()
        except Exception as exc:
            log.warning("cnn_signal: could not write metadata", error=str(exc))
