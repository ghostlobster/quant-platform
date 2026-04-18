"""
tests/test_ensemble_signal.py — Unit tests for strategies/ensemble_signal.py.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from strategies.ensemble_signal import blend_signals


def test_equal_weights_default():
    a = {"AAPL": 0.6, "MSFT": -0.4}
    b = {"AAPL": 0.2, "MSFT": 0.8}
    result = blend_signals(a, b)
    assert abs(result["AAPL"] - 0.4) < 1e-9
    assert abs(result["MSFT"] - 0.2) < 1e-9


def test_custom_weights_normalised():
    a = {"AAPL": 1.0}
    b = {"AAPL": 0.0}
    # 75% weight on a, 25% on b
    result = blend_signals(a, b, weights=[3.0, 1.0])
    assert abs(result["AAPL"] - 0.75) < 1e-9


def test_missing_ticker_treated_as_zero():
    a = {"AAPL": 0.8}
    b = {"MSFT": 0.6}
    result = blend_signals(a, b)
    # AAPL: (0.8 + 0.0) / 2 = 0.4
    assert abs(result["AAPL"] - 0.4) < 1e-9
    # MSFT: (0.0 + 0.6) / 2 = 0.3
    assert abs(result["MSFT"] - 0.3) < 1e-9


def test_output_clipped_to_minus_one_one():
    a = {"X": 0.9}
    b = {"X": 0.9}
    result = blend_signals(a, b)
    assert result["X"] <= 1.0

    c = {"X": -0.9}
    d = {"X": -0.9}
    result2 = blend_signals(c, d)
    assert result2["X"] >= -1.0


def test_empty_input_returns_empty():
    result = blend_signals()
    assert result == {}


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        blend_signals({"A": 0.5}, {"B": 0.3}, weights=[1.0])


def test_all_zero_weights_raises():
    with pytest.raises(ValueError, match="zero"):
        blend_signals({"A": 0.5}, {"B": 0.3}, weights=[0.0, 0.0])


def test_three_models_union_tickers():
    a = {"AAPL": 0.6}
    b = {"MSFT": 0.4}
    c = {"GOOG": 0.2}
    result = blend_signals(a, b, c)
    assert set(result.keys()) == {"AAPL", "MSFT", "GOOG"}
    # Each model covers only its own ticker; others get 0.0
    assert abs(result["AAPL"] - 0.6 / 3) < 1e-9


# ── Adoption-aware weighting (#117) ────────────────────────────────────────────

def test_blend_signals_unchanged_without_model_names(monkeypatch):
    # Regression guard: when model_names is not passed, behaviour must be identical
    # to the pre-#117 blend — even if the knowledge agent would return retrain.
    calls = []

    def _boom(name):
        calls.append(name)
        return "retrain"

    monkeypatch.setattr(
        "strategies.ensemble_signal._get_model_recommendation", _boom,
    )
    a = {"AAPL": 0.6, "MSFT": -0.4}
    b = {"AAPL": 0.2, "MSFT": 0.8}
    result = blend_signals(a, b)
    assert abs(result["AAPL"] - 0.4) < 1e-9
    assert abs(result["MSFT"] - 0.2) < 1e-9
    # Agent must not have been consulted
    assert calls == []


def test_blend_signals_downweights_stale_model(monkeypatch):
    # When one of two models is stale (retrain → 0.4 multiplier), it should
    # carry less weight in the blend.
    recs = {"lgbm_alpha": "retrain", "bayesian": "fresh"}
    monkeypatch.setattr(
        "strategies.ensemble_signal._get_model_recommendation",
        lambda name: recs[name],
    )
    a = {"X": 1.0}   # stale model
    b = {"X": -1.0}  # fresh model
    result = blend_signals(
        a, b, weights=[1.0, 1.0], model_names=["lgbm_alpha", "bayesian"],
    )
    # After multiplier: weights = [0.4, 1.0] → normalised [2/7, 5/7]
    # Score = (2/7)*1.0 + (5/7)*(-1.0) = -3/7 ≈ -0.4286
    assert abs(result["X"] - (-3 / 7)) < 1e-9


def test_blend_signals_renormalises_after_downweight(monkeypatch):
    # Two identical-score models: even if we downweight one, the blended
    # score on a ticker where both agree should equal that score (clipped).
    recs = {"lgbm_alpha": "monitor", "bayesian": "fresh"}
    monkeypatch.setattr(
        "strategies.ensemble_signal._get_model_recommendation",
        lambda name: recs[name],
    )
    a = {"X": 0.5}
    b = {"X": 0.5}
    result = blend_signals(
        a, b, weights=[1.0, 1.0], model_names=["lgbm_alpha", "bayesian"],
    )
    # Both agree on 0.5 → blended 0.5 regardless of relative weights
    assert abs(result["X"] - 0.5) < 1e-9


def test_blend_signals_model_names_length_mismatch_raises(monkeypatch):
    monkeypatch.setattr(
        "strategies.ensemble_signal._get_model_recommendation",
        lambda name: "fresh",
    )
    with pytest.raises(ValueError, match="model_names"):
        blend_signals(
            {"A": 0.5}, {"A": 0.3}, model_names=["only_one_name"],
        )


def test_blend_signals_caches_recommendation_per_call(monkeypatch):
    # The per-call cache should mean _get_model_recommendation is invoked
    # exactly once per unique model_name even when the same name repeats.
    call_counts: dict[str, int] = {}

    def _counting(name: str) -> str:
        call_counts[name] = call_counts.get(name, 0) + 1
        return "fresh"

    monkeypatch.setattr(
        "strategies.ensemble_signal._get_model_recommendation", _counting,
    )
    blend_signals(
        {"A": 0.1}, {"A": 0.2}, {"A": 0.3},
        model_names=["shared_model", "shared_model", "shared_model"],
    )
    assert call_counts == {"shared_model": 1}


def test_blend_signals_unknown_model_defaults_to_fresh(monkeypatch):
    # When the agent fails for any reason we fail open — recommendation
    # defaults to "fresh" so the blend matches the unadjusted path.
    def _raise(name):
        raise RuntimeError("agent unavailable")

    monkeypatch.setattr(
        "strategies.ensemble_signal.KnowledgeAdaptionAgent",
        lambda: (_ for _ in ()).throw(_raise("boom")),
        raising=False,
    )
    a = {"X": 0.5}
    b = {"X": -0.5}
    # Without agent, _get_model_recommendation returns "fresh" for both
    result = blend_signals(
        a, b, weights=[1.0, 1.0], model_names=["m1", "m2"],
    )
    # Both fresh → multipliers 1.0 → weights unchanged → result = 0.0
    assert abs(result["X"]) < 1e-9
