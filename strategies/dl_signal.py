"""
strategies/dl_signal.py — LSTM alpha model (Jansen Ch 17-19).

Tree-based (LightGBM) and linear (Ridge / Bayesian) models in the
platform treat each sample as an independent cross-sectional
observation.  A sequence model — here a compact LSTM — instead looks
at the last ``window`` bars' features per ticker, picking up on
short-horizon path dependence that flat models miss.

Interface parity
----------------
:class:`DLSignal` mirrors :class:`strategies.ml_signal.MLSignal` and
:class:`strategies.linear_signal.LinearSignal`:

  * ``train(tickers, period) -> dict`` metrics
  * ``predict(tickers, period) -> dict[ticker, score ∈ [-1, 1]]``

so the ensemble blender picks it up as a fifth weighted source without
special-casing.

Optional dep
------------
    torch >= 2.0  (``pip install torch``)

When ``torch`` isn't installed, :class:`DLSignal` still imports and its
``predict`` falls back to the momentum composite — consistent with how
``MLSignal`` and ``LinearSignal`` degrade.

ENV vars
--------
    DL_ALPHA_MODEL_PATH   Checkpoint path (default: ``models/dl_alpha.pt``)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.factor_ic import _spearman_corr
from data.features import _FEATURE_COLS, build_feature_matrix
from utils.logger import get_logger

log = get_logger(__name__)

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None           # type: ignore[assignment]
    nn = None               # type: ignore[assignment]
    _TORCH_AVAILABLE = False


_DEFAULT_MODEL_PATH = os.environ.get("DL_ALPHA_MODEL_PATH", "models/dl_alpha.pt")
_TARGET_COL = "fwd_ret_5d"
_DEFAULT_WINDOW = 10
_DEFAULT_HIDDEN = 32


if _TORCH_AVAILABLE:

    class _LSTMRegressor(nn.Module):          # type: ignore[misc]
        """Tiny single-layer LSTM followed by a linear readout."""

        def __init__(self, n_features: int, hidden: int = _DEFAULT_HIDDEN) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden,
                num_layers=1, batch_first=True,
            )
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):                   # type: ignore[override]
            # x: (batch, window, n_features)
            out, _ = self.lstm(x)
            last = out[:, -1, :]                # final timestep's hidden state
            return self.head(last).squeeze(-1)
else:
    _LSTMRegressor = None                       # type: ignore[assignment]


def _build_windowed_tensors(
    fm: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int,
) -> tuple:
    """Shape ``fm`` into per-ticker sliding windows of features → target.

    Returns ``(X, y, idx)`` where:
      * ``X`` is shape ``(n_samples, window, n_features)``
      * ``y`` is shape ``(n_samples,)``
      * ``idx`` is a ``MultiIndex`` of ``(date, ticker)`` aligned with y
        — useful for the cross-sectional IC evaluation.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    idx_rows: list[tuple] = []

    for ticker, slice_ in fm.groupby(level="ticker", sort=False):
        slice_ = slice_.droplevel("ticker").sort_index()
        x_mat = slice_[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_vec = slice_[target_col].to_numpy(dtype=np.float32)
        if len(x_mat) <= window:
            continue
        for i in range(window, len(x_mat)):
            if np.isnan(y_vec[i]):
                continue
            xs.append(x_mat[i - window : i])
            ys.append(float(y_vec[i]))
            idx_rows.append((slice_.index[i], ticker))

    if not xs:
        return None, None, None

    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    idx = pd.MultiIndex.from_tuples(idx_rows, names=["date", "ticker"])
    return X, y, idx


class DLSignal:
    """LSTM-based cross-sectional alpha model with LinearSignal-parity API."""

    def __init__(
        self,
        model_path: str | None = None,
        window: int = _DEFAULT_WINDOW,
    ) -> None:
        self._model_path: str = model_path or _DEFAULT_MODEL_PATH
        self._window: int = int(window)
        self._model = None                  # _LSTMRegressor | None
        self._feature_cols: list[str] = list(_FEATURE_COLS)
        self._load_if_available()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_if_available(self) -> None:
        path = Path(self._model_path)
        if not path.exists() or not _TORCH_AVAILABLE:
            if not path.exists():
                log.info("dl_signal: no checkpoint found", path=str(path))
            return
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                log.warning("dl_signal: unexpected checkpoint format", path=str(path))
                return
            arch = state.get("arch", {})
            n_features = int(arch.get("n_features", len(self._feature_cols)))
            hidden = int(arch.get("hidden", _DEFAULT_HIDDEN))
            self._window = int(arch.get("window", self._window))
            self._feature_cols = list(
                arch.get("feature_cols", self._feature_cols)
            )
            model = _LSTMRegressor(n_features=n_features, hidden=hidden)
            model.load_state_dict(state["model_state"])
            model.eval()
            self._model = model
            log.info("dl_signal: loaded checkpoint", path=str(path))
        except Exception as exc:
            log.warning(
                "dl_signal: failed to load checkpoint",
                path=str(path), error=str(exc),
            )

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        tickers: list[str],
        period: str = "2y",
        test_size: float = 0.2,
        epochs: int = 20,
        lr: float = 1e-3,
        hidden: int = _DEFAULT_HIDDEN,
        batch_size: int = 256,
        seed: int = 42,
    ) -> dict[str, float]:
        """Fit the LSTM and return train / test Spearman IC metrics.

        Parameters mirror ``MLSignal.train`` where practical.  The
        heavy lifting (batching, optimisation) is deliberately minimal
        — AFML / Jansen aren't competing with transformer-scale
        recipes here.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "torch is not installed.  Run: pip install torch>=2.0"
            )

        log.info("dl_signal: building feature matrix", tickers=len(tickers), period=period)
        fm = build_feature_matrix(tickers, period=period)
        if fm.empty:
            raise ValueError("Feature matrix is empty — check tickers and period")
        fm = fm.dropna(subset=[_TARGET_COL])
        if fm.empty:
            raise ValueError(f"No rows with non-NaN target '{_TARGET_COL}'")

        feature_cols = [c for c in _FEATURE_COLS if c in fm.columns]
        self._feature_cols = feature_cols

        # Chronological train / test split on unique dates.
        dates = sorted(fm.index.get_level_values("date").unique())
        split_idx = int(len(dates) * (1 - test_size))
        train_dates = set(dates[:split_idx])
        test_dates = set(dates[split_idx:])

        train_fm = fm[fm.index.get_level_values("date").isin(train_dates)]
        test_fm = fm[fm.index.get_level_values("date").isin(test_dates)]

        X_train, y_train, _ = _build_windowed_tensors(
            train_fm, feature_cols, _TARGET_COL, self._window,
        )
        X_test, y_test, _ = _build_windowed_tensors(
            test_fm, feature_cols, _TARGET_COL, self._window,
        )
        if X_train is None or X_test is None:
            raise ValueError("Not enough bars per ticker to build training windows")

        torch.manual_seed(seed)
        model = _LSTMRegressor(n_features=len(feature_cols), hidden=hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        n_train = len(X_train_t)

        model.train()
        for epoch in range(int(epochs)):
            perm = torch.randperm(n_train)
            epoch_loss = 0.0
            for start in range(0, n_train, batch_size):
                batch_idx = perm[start : start + batch_size]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(batch_idx)
            log.debug(
                "dl_signal: epoch complete",
                epoch=epoch + 1, loss=epoch_loss / max(1, n_train),
            )
        model.eval()
        self._model = model

        # Evaluation: Spearman IC on train + test.
        with torch.no_grad():
            train_pred = model(X_train_t).cpu().numpy()
            test_pred = model(torch.from_numpy(X_test)).cpu().numpy()
        train_ic = _spearman_corr(train_pred, y_train)
        test_ic = _spearman_corr(test_pred, y_test)

        # Persist: state_dict + architecture metadata.
        Path(self._model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "arch": {
                    "n_features": len(feature_cols),
                    "hidden": int(hidden),
                    "window": self._window,
                    "feature_cols": feature_cols,
                },
            },
            self._model_path,
        )
        log.info(
            "dl_signal: checkpoint saved",
            path=self._model_path,
            train_ic=round(float(train_ic), 4),
            test_ic=round(float(test_ic), 4),
        )

        return {
            "train_ic": float(train_ic),
            "test_ic": float(test_ic),
            "train_icir": float(train_ic),
            "test_icir": float(test_ic),
            "n_train_samples": int(len(X_train)),
            "n_test_samples": int(len(X_test)),
        }

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(
        self, tickers: list[str], period: str = "6mo",
    ) -> dict[str, float]:
        """Return ``{ticker: score}`` z-scored + clipped to ``[-1, 1]``.

        Falls back to the momentum composite when torch isn't
        installed, no checkpoint is loaded, or the window-building
        step yields nothing.
        """
        if not _TORCH_AVAILABLE or self._model is None:
            return self._momentum_fallback(tickers, period)

        try:
            fm = build_feature_matrix(tickers, period=period)
            if fm.empty:
                return self._momentum_fallback(tickers, period)
            feature_cols = [c for c in self._feature_cols if c in fm.columns]

            # For predict we want the *latest* window per ticker.
            predictions: dict[str, float] = {}
            for ticker in tickers:
                try:
                    slice_ = fm.xs(ticker, level="ticker").sort_index()
                except KeyError:
                    continue
                if len(slice_) < self._window:
                    continue
                win = slice_.iloc[-self._window :][feature_cols].fillna(0.0)
                x = torch.from_numpy(win.to_numpy(dtype=np.float32)).unsqueeze(0)
                with torch.no_grad():
                    raw = float(self._model(x).item())
                predictions[ticker] = raw

            if not predictions:
                return self._momentum_fallback(tickers, period)

            # Cross-sectional z-score + clip.
            values = np.array(list(predictions.values()), dtype=float)
            std = float(values.std())
            if std == 0.0:
                return {t: 0.0 for t in predictions}
            z = np.clip((values - values.mean()) / std, -1.0, 1.0)
            return {t: float(z[i]) for i, t in enumerate(predictions.keys())}

        except Exception as exc:
            log.warning("dl_signal.predict: error, falling back", error=str(exc))
            return self._momentum_fallback(tickers, period)

    @staticmethod
    def _momentum_fallback(tickers: list[str], period: str) -> dict[str, float]:
        from data.fetcher import fetch_ohlcv
        from strategies.momentum import compute_momentum_score

        scores: dict[str, float] = {}
        for ticker in tickers:
            try:
                df = fetch_ohlcv(ticker, period)
                if df is not None and not df.empty:
                    mom = compute_momentum_score(df)
                    last_val = (
                        mom.dropna().iloc[-1] if not mom.dropna().empty else 0.0
                    )
                    scores[ticker] = float(np.clip(last_val, -1.0, 1.0))
                else:
                    scores[ticker] = 0.0
            except Exception:
                scores[ticker] = 0.0
        return scores

    # ── Accessors ────────────────────────────────────────────────────────────

    def is_trained(self) -> bool:
        return self._model is not None

    def info(self) -> Optional[dict]:
        """Return a small dict describing the currently-loaded model."""
        if self._model is None:
            return None
        return {
            "window": self._window,
            "n_features": len(self._feature_cols),
            "feature_cols": list(self._feature_cols),
        }
