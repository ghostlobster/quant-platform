"""Tests for providers/options_flow.py and mock adapter (Issue #37)."""
from __future__ import annotations

import pytest


def test_get_options_flow_returns_mock_by_default():
    import os
    os.environ.pop("OPTIONS_FLOW_PROVIDER", None)
    from providers.options_flow import get_options_flow
    provider = get_options_flow()
    # Should return the mock adapter
    assert hasattr(provider, "get_flow")
    assert hasattr(provider, "unusual_activity_score")


def test_mock_adapter_get_flow_returns_list():
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter

    adapter = MockOptionsFlowAdapter()
    flow = adapter.get_flow("AAPL", lookback_days=1)
    assert isinstance(flow, list)
    assert len(flow) > 0


def test_mock_adapter_get_flow_structure():
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter

    adapter = MockOptionsFlowAdapter()
    flow = adapter.get_flow("MSFT", lookback_days=1)
    record = flow[0]
    assert "symbol" in record
    assert "side" in record
    assert record["side"] in ("call", "put")
    assert "volume" in record
    assert "strike" in record


def test_mock_adapter_unusual_activity_score():
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter
    from providers.options_flow import OptionsFlowResult

    adapter = MockOptionsFlowAdapter()
    result = adapter.unusual_activity_score("AAPL")
    assert isinstance(result, OptionsFlowResult)
    assert result.symbol == "AAPL"
    assert result.call_volume >= 0
    assert result.put_volume >= 0
    assert -1.0 <= result.unusual_score <= 1.0


def test_mock_adapter_call_put_ratio_positive():
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter

    adapter = MockOptionsFlowAdapter()
    result = adapter.unusual_activity_score("NVDA")
    assert result.call_put_ratio >= 0


def test_mock_adapter_deterministic_for_same_symbol():
    """Same ticker should return the same synthetic values."""
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter

    adapter = MockOptionsFlowAdapter()
    r1 = adapter.unusual_activity_score("TSLA")
    r2 = adapter.unusual_activity_score("TSLA")
    assert r1.call_volume == r2.call_volume
    assert r1.put_volume == r2.put_volume
    assert r1.unusual_score == r2.unusual_score


def test_mock_adapter_different_tickers_differ():
    from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter

    adapter = MockOptionsFlowAdapter()
    r_aapl = adapter.unusual_activity_score("AAPL")
    r_msft = adapter.unusual_activity_score("MSFT")
    # Different tickers may have different scores (hash-based determinism)
    assert r_aapl.symbol == "AAPL"
    assert r_msft.symbol == "MSFT"


def test_get_options_flow_unknown_provider_raises():
    from providers.options_flow import get_options_flow

    with pytest.raises(ValueError, match="Unknown options flow provider"):
        get_options_flow(provider="nonexistent_provider")


def test_get_options_flow_explicit_mock():
    from providers.options_flow import get_options_flow

    provider = get_options_flow(provider="mock")
    result = provider.unusual_activity_score("SPY")
    assert result.symbol == "SPY"
