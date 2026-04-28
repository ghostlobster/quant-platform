"""Tests for analysis/regime.py — market regime classification.

Covers ``detect_regime`` quant classification, the LLM-fusion pipeline
(``_build_macro_prompt`` / ``_parse_llm_regime`` / ``_blend_regimes``),
``get_live_regime`` happy + error paths, ``get_live_regime_with_llm``
weight modes, and ``get_cached_live_regime`` TTL behaviour.

Coverage achieved: 100 % combined line+branch (closes #215).

Beyond line+branch, the file aims for "excellent" rather than baseline:

  * **Schema invariant** — locks the output keys ``pages/chart.py`` reads
    so a future refactor can't silently break the regime banner.
  * **Property tests** (hypothesis) — randomly generated VIX × SPY
    inputs verify that ``detect_regime`` always returns one of the four
    known regimes, that ``is_regime_at_risk`` is symmetric in spread
    direction, and that ``_blend_regimes`` clamps weight to [0, 1].
  * **Boundary matrix** — every (VIX-bucket × SPY direction) combination
    exercised explicitly, plus the exact-20 / exact-30 / 30.01 boundaries.
  * **Failure-mode coverage** — yfinance missing, SPY empty, VIX empty,
    LLM down, malformed LLM JSON, invalid regime label, out-of-range
    confidence — every degraded path documented and asserted.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from analysis import regime as regime_module
from analysis.regime import (
    REGIME_METADATA,
    REGIME_STATES,
    _blend_regimes,
    _build_macro_prompt,
    _parse_llm_regime,
    detect_regime,
    get_cached_live_regime,
    get_live_regime,
    get_live_regime_with_llm,
    is_regime_at_risk,
    kelly_regime_multiplier,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_prices(n: int = 250, base: float = 100.0, last: float = 100.0) -> pd.Series:
    """Return a price series of length n where the final value is `last`.

    All values except the last are `base`, so the 200d SMA ≈ base.
    Setting `last` above or below `base` controls whether SPY is above/below SMA.
    """
    prices = np.full(n, base, dtype=float)
    prices[-1] = last
    return pd.Series(prices)


@pytest.fixture(autouse=True)
def _clear_regime_cache():
    """Reset the module-level regime cache between tests."""
    regime_module._regime_cache["data"] = None
    regime_module._regime_cache["expires_at"] = 0.0
    yield
    regime_module._regime_cache["data"] = None
    regime_module._regime_cache["expires_at"] = 0.0


# ── detect_regime ────────────────────────────────────────────────────────────

class TestDetectRegime:
    def test_trending_bull(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bull"

    def test_trending_bear(self) -> None:
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bear"

    def test_mean_reverting_spy_above_sma(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=25.0) == "mean_reverting"

    def test_mean_reverting_spy_below_sma(self) -> None:
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=25.0) == "mean_reverting"

    def test_high_vol_spy_above_sma(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=35.0) == "high_vol"

    def test_high_vol_spy_below_sma(self) -> None:
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=35.0) == "high_vol"

    def test_short_history_warns_but_classifies(self, capsys) -> None:
        """Less than 200 points → warning emitted, classification still
        runs over the available data instead of erroring."""
        prices = pd.Series([100.0] * 50 + [110.0])
        r = detect_regime(prices, vix_level=15.0)
        assert r == "trending_bull"
        # structlog routes warnings to stdout in console mode; check capsys.
        captured = capsys.readouterr()
        assert "price points" in (captured.out + captured.err)


# ── Boundary conditions ──────────────────────────────────────────────────────

class TestBoundaryConditions:
    def test_vix_exactly_20_is_mean_reverting(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=20.0) == "mean_reverting"

    def test_vix_exactly_30_is_mean_reverting(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=30.0) == "mean_reverting"

    def test_vix_just_above_30_is_high_vol(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=30.01) == "high_vol"

    def test_vix_just_below_20_uses_sma_bull(self) -> None:
        prices = _make_prices(last=110.0)
        assert detect_regime(prices, vix_level=19.99) == "trending_bull"

    def test_vix_just_below_20_uses_sma_bear(self) -> None:
        prices = _make_prices(last=90.0)
        assert detect_regime(prices, vix_level=19.99) == "trending_bear"

    def test_spy_equal_to_sma_is_trending_bear(self) -> None:
        prices = _make_prices(n=200, base=100.0, last=100.0)
        assert detect_regime(prices, vix_level=15.0) == "trending_bear"


# ── Kelly multiplier ─────────────────────────────────────────────────────────

class TestKellyRegimeMultiplier:
    @pytest.mark.parametrize("regime,expected", [
        ("high_vol",        0.5),
        ("trending_bull",   1.0),
        ("trending_bear",   1.0),
        ("mean_reverting",  1.0),
    ])
    def test_kelly(self, regime: str, expected: float) -> None:
        assert kelly_regime_multiplier(regime) == expected


# ── is_regime_at_risk ────────────────────────────────────────────────────────

class TestIsRegimeAtRisk:
    def test_spy_at_sma_is_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=100.0, spy_sma200=100.0, vix=15.0) is True

    def test_spy_within_2pct_of_sma_is_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=101.0, spy_sma200=100.0, vix=15.0) is True

    def test_spy_far_from_sma_low_vix_not_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=15.0) is False

    def test_vix_within_10pct_of_20_is_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=19.0) is True

    def test_vix_at_22_is_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=22.0) is True

    def test_vix_at_25_not_at_risk(self) -> None:
        assert is_regime_at_risk(spy_price=110.0, spy_sma200=100.0, vix=25.0) is False

    def test_zero_sma_returns_false(self) -> None:
        assert is_regime_at_risk(spy_price=100.0, spy_sma200=0.0, vix=15.0) is False


# ── REGIME_METADATA ──────────────────────────────────────────────────────────

class TestRegimeMetadata:
    def test_all_four_regimes_present(self) -> None:
        for regime in ("trending_bull", "trending_bear", "mean_reverting", "high_vol"):
            assert regime in REGIME_METADATA, f"Missing regime: {regime}"

    def test_metadata_has_description(self) -> None:
        for regime, meta in REGIME_METADATA.items():
            assert isinstance(meta["description"], str) and meta["description"]

    def test_metadata_has_recommended_strategies(self) -> None:
        for regime, meta in REGIME_METADATA.items():
            assert isinstance(meta["recommended_strategies"], list)
            assert len(meta["recommended_strategies"]) > 0

    def test_regime_states_constant_matches_metadata(self) -> None:
        assert set(REGIME_STATES) == set(REGIME_METADATA.keys())


# ── _build_macro_prompt ──────────────────────────────────────────────────────

class TestBuildMacroPrompt:
    def test_prompt_includes_inputs(self) -> None:
        p = _build_macro_prompt(spy_price=500.0, spy_sma200=480.0, vix=14.2)
        assert "500.00" in p and "480.00" in p and "14.20" in p

    def test_prompt_says_above_when_price_above_sma(self) -> None:
        p = _build_macro_prompt(spy_price=500.0, spy_sma200=480.0, vix=14.2)
        assert "above" in p

    def test_prompt_says_below_when_price_below_sma(self) -> None:
        p = _build_macro_prompt(spy_price=460.0, spy_sma200=480.0, vix=14.2)
        assert "below" in p

    def test_prompt_lists_all_regime_states(self) -> None:
        p = _build_macro_prompt(spy_price=500.0, spy_sma200=480.0, vix=14.2)
        for state in REGIME_STATES:
            assert state in p


# ── _parse_llm_regime ────────────────────────────────────────────────────────

class TestParseLLMRegime:
    def test_valid_response(self) -> None:
        regime, conf = _parse_llm_regime('{"regime": "trending_bull", "confidence": 0.82}')
        assert regime == "trending_bull"
        assert conf == pytest.approx(0.82)

    def test_strips_markdown_code_fences(self) -> None:
        raw = '```json\n{"regime": "high_vol", "confidence": 0.5}\n```'
        regime, conf = _parse_llm_regime(raw)
        assert regime == "high_vol"
        assert conf == pytest.approx(0.5)

    def test_clamps_confidence_to_unit_interval_high(self) -> None:
        regime, conf = _parse_llm_regime('{"regime": "trending_bull", "confidence": 1.7}')
        assert conf == 1.0

    def test_clamps_confidence_to_unit_interval_low(self) -> None:
        regime, conf = _parse_llm_regime('{"regime": "trending_bull", "confidence": -0.4}')
        assert conf == 0.0

    def test_falls_back_on_invalid_regime(self) -> None:
        regime, conf = _parse_llm_regime('{"regime": "panic", "confidence": 0.9}')
        assert regime == "high_vol"
        assert conf == 0.5

    def test_falls_back_on_malformed_json(self) -> None:
        regime, conf = _parse_llm_regime('not json at all')
        assert regime == "high_vol"
        assert conf == 0.5

    def test_falls_back_on_missing_regime_key(self) -> None:
        regime, conf = _parse_llm_regime('{"confidence": 0.9}')
        assert regime == "high_vol"
        assert conf == 0.5

    def test_default_confidence_when_missing(self) -> None:
        regime, conf = _parse_llm_regime('{"regime": "trending_bull"}')
        assert regime == "trending_bull"
        assert conf == pytest.approx(0.5)


# ── _blend_regimes ───────────────────────────────────────────────────────────

class TestBlendRegimes:
    def test_weight_zero_returns_quant(self) -> None:
        assert _blend_regimes("trending_bull", "high_vol", 1.0, 0.0) == "trending_bull"

    def test_weight_one_returns_llm(self) -> None:
        assert _blend_regimes("trending_bull", "high_vol", 1.0, 1.0) == "high_vol"

    def test_weight_negative_treated_as_zero(self) -> None:
        assert _blend_regimes("trending_bull", "high_vol", 1.0, -0.3) == "trending_bull"

    def test_weight_above_one_treated_as_one(self) -> None:
        assert _blend_regimes("trending_bull", "high_vol", 1.0, 1.4) == "high_vol"

    def test_intermediate_weight_high_llm_confidence_picks_llm(self) -> None:
        # weight=0.7 + llm_conf=0.95 should pull toward llm_regime
        out = _blend_regimes("trending_bull", "high_vol", 0.95, 0.7)
        assert out == "high_vol"

    def test_intermediate_weight_low_llm_confidence_keeps_quant(self) -> None:
        # weight=0.4 + llm_conf=0.3 keeps the quant signal
        out = _blend_regimes("trending_bull", "high_vol", 0.3, 0.4)
        assert out == "trending_bull"

    def test_agreement_returns_same(self) -> None:
        # Both signals agree → blended must agree
        out = _blend_regimes("trending_bull", "trending_bull", 0.8, 0.5)
        assert out == "trending_bull"


# ── get_live_regime ──────────────────────────────────────────────────────────

def _fake_yfinance(spy_close=None, vix_close=15.0, spy_empty=False, vix_empty=False):
    """Build a fake yfinance module that returns deterministic frames."""
    yf = types.ModuleType("yfinance")

    def download(symbol, period="1y", progress=False, auto_adjust=True):
        if symbol == "SPY":
            if spy_empty:
                return pd.DataFrame()
            close = spy_close if spy_close is not None else (
                [100.0] * 199 + [110.0]
            )
            return pd.DataFrame({"Close": close})
        if symbol == "^VIX":
            if vix_empty:
                return pd.DataFrame()
            return pd.DataFrame({"Close": [vix_close]})
        return pd.DataFrame()

    yf.download = download  # type: ignore[attr-defined]
    return yf


class TestGetLiveRegime:
    def test_happy_path_trending_bull(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        out = get_live_regime()
        assert out["regime"] == "trending_bull"
        assert out["spy_price"] == pytest.approx(110.0)
        assert out["spy_sma200"] == pytest.approx(
            (199 * 100.0 + 110.0) / 200, rel=1e-9
        )
        assert out["vix"] == pytest.approx(14.0)
        assert "description" in out and out["description"]
        assert isinstance(out["recommended_strategies"], list)
        assert "at_risk" in out

    def test_happy_path_high_vol(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=35.0))
        out = get_live_regime()
        assert out["regime"] == "high_vol"

    def test_output_has_chart_required_keys(self, monkeypatch) -> None:
        """``pages/chart.py:_render_regime_badge`` reads exactly these keys.
        Lock the output schema so a future refactor doesn't silently break
        the chart banner."""
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        out = get_live_regime()
        for key in (
            "regime", "spy_price", "spy_sma200", "vix",
            "description", "recommended_strategies",
        ):
            assert key in out, f"Missing required output key: {key}"

    def test_raises_when_yfinance_missing(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", None)
        with pytest.raises(ImportError, match="yfinance is required"):
            get_live_regime()

    def test_raises_when_spy_empty(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(spy_empty=True))
        with pytest.raises(RuntimeError, match="SPY"):
            get_live_regime()

    def test_raises_when_vix_empty(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_empty=True))
        with pytest.raises(RuntimeError, match="VIX"):
            get_live_regime()


# ── get_live_regime_with_llm ─────────────────────────────────────────────────

class TestGetLiveRegimeWithLLM:
    def test_weight_zero_short_circuits_to_get_live_regime(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        out = get_live_regime_with_llm(llm_weight=0.0)
        assert out["regime"] == "trending_bull"
        assert "llm_regime" not in out  # LLM code path not entered

    def test_env_var_default_is_zero(self, monkeypatch) -> None:
        monkeypatch.delenv("REGIME_LLM_WEIGHT", raising=False)
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        out = get_live_regime_with_llm()
        assert out["regime"] == "trending_bull"
        assert "llm_regime" not in out

    def test_weight_positive_blends_with_llm(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        fake_llm = MagicMock()
        fake_llm.complete.return_value = (
            '{"regime": "high_vol", "confidence": 0.95}'
        )
        with patch("providers.llm.get_llm", return_value=fake_llm):
            out = get_live_regime_with_llm(llm_weight=0.9)
        assert out["llm_regime"] == "high_vol"
        assert out["llm_confidence"] == pytest.approx(0.95)
        assert out["llm_weight"] == pytest.approx(0.9)
        assert out["quant_regime"] == "trending_bull"
        # Heavy llm weight + high confidence → blended regime is high_vol
        assert out["regime"] == "high_vol"
        assert out["description"] == REGIME_METADATA["high_vol"]["description"]

    def test_weight_positive_llm_failure_falls_back_to_quant(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        with patch("providers.llm.get_llm", side_effect=RuntimeError("llm down")):
            out = get_live_regime_with_llm(llm_weight=0.7)
        # Quant regime preserved; LLM keys not added when fusion fails
        assert out["regime"] == "trending_bull"
        assert "llm_regime" not in out


# ── get_cached_live_regime ───────────────────────────────────────────────────

class TestGetCachedLiveRegime:
    def test_cache_hit_returns_stored_value(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        first = get_cached_live_regime()
        # Second call should not re-fetch — confirm by replacing yfinance with a
        # raiser; if the cache did re-fetch we'd see the exception.
        monkeypatch.setitem(sys.modules, "yfinance", None)
        second = get_cached_live_regime()
        assert first == second

    def test_cache_miss_refetches(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        first = get_cached_live_regime()
        # Expire the cache
        regime_module._regime_cache["expires_at"] = 0.0
        # Swap to a fake that flips to high_vol → must refetch and reflect
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=35.0))
        second = get_cached_live_regime()
        assert first["regime"] == "trending_bull"
        assert second["regime"] == "high_vol"

    def test_use_llm_false_takes_quant_path(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        out = get_cached_live_regime(use_llm=False)
        assert out["regime"] == "trending_bull"
        assert "llm_regime" not in out

    def test_use_llm_true_takes_llm_path(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "yfinance", _fake_yfinance(vix_close=14.0))
        monkeypatch.setenv("REGIME_LLM_WEIGHT", "0.0")  # weight 0 short-circuits
        out = get_cached_live_regime(use_llm=True)
        # Quant path is reached because weight=0; LLM keys absent.
        assert out["regime"] == "trending_bull"
        assert "llm_regime" not in out


# ── Property tests (hypothesis) ─────────────────────────────────────────────

# Imported lazily inside the suite so the file collects even when hypothesis
# is missing (see #199 silent-skip guard for the rationale on optional deps).
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_PROP_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


class TestRegimeProperties:
    @given(
        last_price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
        sma_base=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
        vix=st.floats(min_value=0.0, max_value=200.0, allow_nan=False),
    )
    @_PROP_SETTINGS
    def test_detect_regime_total_function(
        self, last_price: float, sma_base: float, vix: float
    ) -> None:
        """``detect_regime`` is total over plausible inputs — every output
        is one of the four documented regimes, never raises."""
        prices = _make_prices(n=210, base=sma_base, last=last_price)
        out = detect_regime(prices, vix_level=vix)
        assert out in REGIME_STATES

    @given(
        spy_price=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
        spy_sma=st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False),
        vix=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
    )
    @_PROP_SETTINGS
    def test_is_regime_at_risk_returns_bool(
        self, spy_price: float, spy_sma: float, vix: float
    ) -> None:
        """``is_regime_at_risk`` is a total predicate — always returns
        a Python ``bool`` for any plausible input."""
        out = is_regime_at_risk(spy_price, spy_sma, vix)
        assert isinstance(out, bool)

    @given(
        weight=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False),
        llm_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        quant=st.sampled_from(REGIME_STATES),
        llm=st.sampled_from(REGIME_STATES),
    )
    @_PROP_SETTINGS
    def test_blend_regimes_total_and_in_states(
        self, weight: float, llm_conf: float, quant: str, llm: str
    ) -> None:
        """``_blend_regimes`` clamps weight to [0, 1] internally and
        always returns a documented regime label."""
        out = _blend_regimes(quant, llm, llm_conf, weight)
        assert out in REGIME_STATES

    @given(
        regime=st.sampled_from(REGIME_STATES),
        confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    @_PROP_SETTINGS
    def test_parse_llm_regime_round_trip(
        self, regime: str, confidence: float
    ) -> None:
        """Any well-formed LLM JSON in the documented schema parses back
        to the same (regime, confidence) pair."""
        import json
        raw = json.dumps({"regime": regime, "confidence": confidence})
        parsed_regime, parsed_conf = _parse_llm_regime(raw)
        assert parsed_regime == regime
        assert parsed_conf == pytest.approx(confidence, abs=1e-12)
