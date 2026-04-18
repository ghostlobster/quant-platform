"""Tests for providers/macro.py, adapters/macro/*, and data/macro.py."""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── Provider factory ──────────────────────────────────────────────────────────

def test_get_macro_defaults_to_mock_when_no_api_key(monkeypatch):
    monkeypatch.delenv("MACRO_PROVIDER", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from adapters.macro.mock_adapter import MockMacroAdapter
    from providers.macro import get_macro
    assert isinstance(get_macro(), MockMacroAdapter)


def test_get_macro_picks_fred_when_api_key_present(monkeypatch):
    monkeypatch.delenv("MACRO_PROVIDER", raising=False)
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    from adapters.macro.fred_adapter import FREDAdapter
    from providers.macro import get_macro
    assert isinstance(get_macro(), FREDAdapter)


def test_get_macro_explicit_override(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    from adapters.macro.mock_adapter import MockMacroAdapter
    from providers.macro import get_macro
    assert isinstance(get_macro("mock"), MockMacroAdapter)


def test_get_macro_unknown_raises():
    from providers.macro import get_macro
    with pytest.raises(ValueError, match="Unknown macro provider"):
        get_macro("not-a-thing")


# ── Mock adapter ──────────────────────────────────────────────────────────────

def test_mock_adapter_returns_deterministic_series():
    from adapters.macro.mock_adapter import MockMacroAdapter
    a, b = MockMacroAdapter(), MockMacroAdapter()
    s1 = a.get_series("VIXCLS", start="2024-01-01", end="2024-02-01")
    s2 = b.get_series("VIXCLS", start="2024-01-01", end="2024-02-01")
    assert not s1.empty
    pd.testing.assert_series_equal(s1, s2)


def test_mock_adapter_different_series_yield_different_values():
    from adapters.macro.mock_adapter import MockMacroAdapter
    a = MockMacroAdapter()
    s1 = a.get_series("VIXCLS", start="2024-01-01", end="2024-02-01")
    s2 = a.get_series("DGS10", start="2024-01-01", end="2024-02-01")
    assert not s1.equals(s2)


def test_mock_adapter_empty_window():
    from adapters.macro.mock_adapter import MockMacroAdapter
    a = MockMacroAdapter()
    s = a.get_series("X", start="2024-01-10", end="2024-01-09")
    assert s.empty


# ── FRED adapter (mocked HTTP, no network) ────────────────────────────────────

def _fake_response(payload: dict):
    body = json.dumps(payload).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda self: self
    mock_resp.__exit__ = lambda self, exc_type, exc, tb: False
    return mock_resp


def test_fred_adapter_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    from adapters.macro.fred_adapter import FREDAdapter
    s = FREDAdapter().get_series("VIXCLS")
    assert s.empty


def test_fred_adapter_parses_observations():
    from adapters.macro.fred_adapter import FREDAdapter
    payload = {
        "observations": [
            {"date": "2024-01-02", "value": "12.34"},
            {"date": "2024-01-03", "value": "13.50"},
            {"date": "2024-01-04", "value": "."},   # missing → skipped
            {"date": "2024-01-05", "value": "14.10"},
        ],
    }
    with patch(
        "adapters.macro.fred_adapter.urllib.request.urlopen",
        return_value=_fake_response(payload),
    ):
        adapter = FREDAdapter(api_key="test-key")
        s = adapter.get_series("VIXCLS", start="2024-01-01", end="2024-01-31")

    assert list(s.index.strftime("%Y-%m-%d")) == ["2024-01-02", "2024-01-03", "2024-01-05"]
    assert s.tolist() == [12.34, 13.50, 14.10]


def test_fred_adapter_handles_http_failure_gracefully():
    from adapters.macro.fred_adapter import FREDAdapter
    with patch(
        "adapters.macro.fred_adapter.urllib.request.urlopen",
        side_effect=RuntimeError("network down"),
    ):
        s = FREDAdapter(api_key="test-key").get_series("VIXCLS")
    assert s.empty


def test_fred_adapter_invalid_value_skipped():
    from adapters.macro.fred_adapter import FREDAdapter
    payload = {
        "observations": [
            {"date": "2024-01-02", "value": "not-a-number"},
            {"date": "2024-01-03", "value": "5.0"},
        ],
    }
    with patch(
        "adapters.macro.fred_adapter.urllib.request.urlopen",
        return_value=_fake_response(payload),
    ):
        s = FREDAdapter(api_key="test-key").get_series("VIXCLS")
    assert s.tolist() == [5.0]


# ── data/macro.py ─────────────────────────────────────────────────────────────

def test_fetch_macro_series_uses_provider_when_cache_empty():
    from data import macro as data_macro

    fake = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"]),
        name="X",
        dtype=float,
    )

    fake_provider = MagicMock()
    fake_provider.get_series.return_value = fake

    with (
        patch.object(data_macro, "_read_cache", return_value=pd.Series(dtype=float)),
        patch.object(data_macro, "_write_cache") as mock_write,
        patch("providers.macro.get_macro", return_value=fake_provider),
    ):
        s = data_macro.fetch_macro_series("X")

    pd.testing.assert_series_equal(s, fake)
    mock_write.assert_called_once()


def test_fetch_macro_series_returns_cache_when_available():
    from data import macro as data_macro
    cached = pd.Series([7.0], index=pd.DatetimeIndex(["2024-06-01"]), name="X", dtype=float)
    with patch.object(data_macro, "_read_cache", return_value=cached):
        s = data_macro.fetch_macro_series("X")
    pd.testing.assert_series_equal(s, cached)


def test_fetch_macro_series_handles_provider_failure():
    from data import macro as data_macro
    with (
        patch.object(data_macro, "_read_cache", return_value=pd.Series(dtype=float)),
        patch("providers.macro.get_macro", side_effect=RuntimeError("no provider")),
    ):
        s = data_macro.fetch_macro_series("X")
    assert s.empty


def test_macro_context_features_assembles_multi_series():
    from data import macro as data_macro

    def fake_fetch(series_id, start=None, end=None, provider_name=None, use_cache=True):
        idx = pd.date_range("2024-01-01", "2024-01-10", freq="B")
        if series_id == "MISSING":
            return pd.Series(dtype=float)
        return pd.Series(range(len(idx)), index=idx, name=series_id, dtype=float)

    with patch.object(data_macro, "fetch_macro_series", side_effect=fake_fetch):
        df = data_macro.macro_context_features(
            pd.date_range("2024-01-01", "2024-01-10", freq="B"),
            ["VIXCLS", "MISSING"],
        )

    assert "VIXCLS" in df.columns and "MISSING" in df.columns
    assert df["MISSING"].isna().all()
    assert df["VIXCLS"].notna().all()


def test_macro_context_features_empty_dates_returns_empty_df():
    from data.macro import macro_context_features
    out = macro_context_features([])
    assert out.empty
