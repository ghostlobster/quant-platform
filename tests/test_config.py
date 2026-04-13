"""
Tests for config.py and analysis/regime.get_live_regime().
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── config.py ─────────────────────────────────────────────────────────────────

class TestConfig:
    def test_alpaca_api_key_defaults_to_empty(self):
        with patch.dict(os.environ, {}, clear=False):
            import importlib

            import config
            importlib.reload(config)
            assert isinstance(config.ALPACA_API_KEY, str)

    def test_alpaca_secret_key_defaults_to_empty(self):
        import config
        assert isinstance(config.ALPACA_SECRET_KEY, str)

    def test_alpaca_base_url_default(self):
        import config
        assert "alpaca" in config.ALPACA_BASE_URL.lower()

    def test_app_env_default_is_development(self):
        with patch.dict(os.environ, {"APP_ENV": "development"}):
            import importlib

            import config
            importlib.reload(config)
            assert config.APP_ENV == "development"

    def test_log_level_default_is_info(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "INFO"}):
            import importlib

            import config
            importlib.reload(config)
            assert config.LOG_LEVEL == "INFO"

    def test_env_var_override(self):
        with patch.dict(os.environ, {"ALPACA_API_KEY": "test_key_123"}):
            import importlib

            import config
            importlib.reload(config)
            assert config.ALPACA_API_KEY == "test_key_123"


# ── analysis/regime.get_live_regime ──────────────────────────────────────────

class TestGetLiveRegime:
    def _make_ohlcv(self, n: int = 250) -> pd.DataFrame:
        prices = np.linspace(400, 450, n)
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.DataFrame({"Close": prices}, index=idx)

    def test_get_live_regime_trending_bull(self):
        spy_df = self._make_ohlcv(250)
        vix_df = pd.DataFrame({"Close": [15.0]})

        mock_yf = MagicMock()
        mock_yf.download.side_effect = [spy_df, vix_df]

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            import importlib

            import analysis.regime
            importlib.reload(analysis.regime)
            result = analysis.regime.get_live_regime()

        assert result["regime"] in ("trending_bull", "trending_bear", "mean_reverting", "high_vol")
        assert "spy_price" in result
        assert "vix" in result
        assert "description" in result
        assert "recommended_strategies" in result

    def test_get_live_regime_raises_on_empty_spy(self):
        mock_yf = MagicMock()
        mock_yf.download.return_value = pd.DataFrame()

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            import importlib

            import analysis.regime
            importlib.reload(analysis.regime)
            with pytest.raises(RuntimeError, match="SPY"):
                analysis.regime.get_live_regime()

    def test_get_live_regime_raises_on_empty_vix(self):
        spy_df = self._make_ohlcv(250)

        mock_yf = MagicMock()
        mock_yf.download.side_effect = [spy_df, pd.DataFrame()]

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            import importlib

            import analysis.regime
            importlib.reload(analysis.regime)
            with pytest.raises(RuntimeError, match="VIX"):
                analysis.regime.get_live_regime()

    def test_get_live_regime_short_series_warning(self):
        """get_live_regime with < 200 SPY points logs a warning but still works."""
        spy_df = self._make_ohlcv(50)  # short
        vix_df = pd.DataFrame({"Close": [25.0]})

        mock_yf = MagicMock()
        mock_yf.download.side_effect = [spy_df, vix_df]

        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            import importlib

            import analysis.regime
            importlib.reload(analysis.regime)
            result = analysis.regime.get_live_regime()
        assert result["regime"] == "mean_reverting"  # VIX=25 → mean_reverting
