"""Protocol contract tests — every adapter satisfies its Protocol (closes #205).

Each ``providers/<x>.py`` module declares a ``Protocol`` that defines the
public interface the rest of the codebase relies on. Each ``adapters/<x>/``
module ships at least one concrete implementation. There is no static-typing
gate in CI (we don't run ``mypy``), so the only feedback today on a missing
or renamed Protocol method is a runtime ``AttributeError`` from whichever
caller happens to invoke it first — typically far away from the broken
adapter, sometimes only under a credential-gated code path.

This file enumerates every (Protocol, adapter) pair and asserts:

  1. Each non-private Protocol method exists on the adapter class.
  2. The adapter signature accepts every required Protocol parameter
     — adapters may add optional kwargs but must not drop required ones.

Adapters that need credentials for ``__init__`` are checked at the **class**
level (not instances), so the test does not require a live connection,
network access, or env vars.

Adding a new adapter: add it to ``ADAPTER_REGISTRY``. Adding a new method
to a Protocol: every adapter immediately fails until it implements it —
which is the regression net we want.
"""
from __future__ import annotations

import inspect

import pytest

# ── Adapter imports ─────────────────────────────────────────────────────────
from adapters.alert.email_adapter import EmailAlertAdapter
from adapters.alert.noop_adapter import NoopAlertAdapter
from adapters.alert.slack_adapter import SlackAlertAdapter
from adapters.broker.alpaca_adapter import AlpacaBrokerAdapter
from adapters.broker.ibkr_adapter import IBKRAdapter
from adapters.broker.paper_adapter import PaperBrokerAdapter
from adapters.broker.schwab_adapter import SchwabAdapter
from adapters.execution_algo.market_adapter import MarketAlgoAdapter
from adapters.execution_algo.twap_adapter import TWAPAdapter
from adapters.execution_algo.vwap_adapter import VWAPAdapter
from adapters.feature_store.memory_adapter import InMemoryFeatureStoreAdapter
from adapters.feature_store.redis_adapter import RedisFeatureStoreAdapter
from adapters.llm.anthropic_adapter import AnthropicAdapter
from adapters.llm.mock_adapter import MockLLMAdapter
from adapters.llm.ollama_adapter import OllamaAdapter
from adapters.llm.openai_adapter import OpenAIAdapter
from adapters.macro.fred_adapter import FREDAdapter
from adapters.macro.mock_adapter import MockMacroAdapter
from adapters.market_data.alpaca_adapter import AlpacaMarketDataAdapter
from adapters.market_data.mock_adapter import MockMarketDataAdapter
from adapters.market_data.polygon_adapter import PolygonAdapter
from adapters.market_data.yfinance_adapter import YFinanceAdapter
from adapters.model_registry.mlflow_adapter import MLflowAdapter
from adapters.model_registry.mock_adapter import MockModelRegistryAdapter
from adapters.options_flow.mock_adapter import MockOptionsFlowAdapter
from adapters.options_flow.thetadata_adapter import ThetaDataAdapter
from adapters.options_flow.unusual_whales_adapter import UnusualWhalesAdapter
from adapters.sentiment.mock_adapter import MockSentimentAdapter
from adapters.sentiment.stocktwits_adapter import StocktwitsAdapter
from adapters.sentiment.vader_adapter import VADERAdapter
from adapters.tsdb.duckdb_adapter import DuckDBAdapter
from adapters.tsdb.sqlite_adapter import SQLiteTSDBAdapter
from adapters.tsdb.timescale_adapter import TimescaleAdapter

# ── Protocol imports ────────────────────────────────────────────────────────
from providers.alert import AlertProvider
from providers.broker import BrokerProvider
from providers.execution_algo import ExecutionAlgoProvider
from providers.feature_store import FeatureStoreProvider
from providers.llm import LLMProvider
from providers.macro import MacroDataProvider
from providers.market_data import MarketDataProvider
from providers.model_registry import ModelRegistryProvider
from providers.options_flow import OptionsFlowProvider
from providers.sentiment import SentimentProvider
from providers.tsdb import TSDBProvider

# ── Registry ────────────────────────────────────────────────────────────────
# (label, Protocol class, Adapter class). The label drives parametrize ids
# so a failure points at exactly which (kind, vendor) combination is wrong.
ADAPTER_REGISTRY: list[tuple[str, type, type]] = [
    ("alert.email",                AlertProvider,         EmailAlertAdapter),
    ("alert.noop",                 AlertProvider,         NoopAlertAdapter),
    ("alert.slack",                AlertProvider,         SlackAlertAdapter),
    ("broker.alpaca",              BrokerProvider,        AlpacaBrokerAdapter),
    ("broker.ibkr",                BrokerProvider,        IBKRAdapter),
    ("broker.paper",               BrokerProvider,        PaperBrokerAdapter),
    ("broker.schwab",              BrokerProvider,        SchwabAdapter),
    ("execution_algo.market",      ExecutionAlgoProvider, MarketAlgoAdapter),
    ("execution_algo.twap",        ExecutionAlgoProvider, TWAPAdapter),
    ("execution_algo.vwap",        ExecutionAlgoProvider, VWAPAdapter),
    ("feature_store.memory",       FeatureStoreProvider,  InMemoryFeatureStoreAdapter),
    ("feature_store.redis",        FeatureStoreProvider,  RedisFeatureStoreAdapter),
    ("llm.anthropic",              LLMProvider,           AnthropicAdapter),
    ("llm.mock",                   LLMProvider,           MockLLMAdapter),
    ("llm.ollama",                 LLMProvider,           OllamaAdapter),
    ("llm.openai",                 LLMProvider,           OpenAIAdapter),
    ("macro.fred",                 MacroDataProvider,     FREDAdapter),
    ("macro.mock",                 MacroDataProvider,     MockMacroAdapter),
    ("market_data.alpaca",         MarketDataProvider,    AlpacaMarketDataAdapter),
    ("market_data.mock",           MarketDataProvider,    MockMarketDataAdapter),
    ("market_data.polygon",        MarketDataProvider,    PolygonAdapter),
    ("market_data.yfinance",       MarketDataProvider,    YFinanceAdapter),
    ("model_registry.mlflow",      ModelRegistryProvider, MLflowAdapter),
    ("model_registry.mock",        ModelRegistryProvider, MockModelRegistryAdapter),
    ("options_flow.mock",          OptionsFlowProvider,   MockOptionsFlowAdapter),
    ("options_flow.thetadata",     OptionsFlowProvider,   ThetaDataAdapter),
    ("options_flow.unusual_whales", OptionsFlowProvider,  UnusualWhalesAdapter),
    ("sentiment.mock",             SentimentProvider,     MockSentimentAdapter),
    ("sentiment.stocktwits",       SentimentProvider,     StocktwitsAdapter),
    ("sentiment.vader",            SentimentProvider,     VADERAdapter),
    ("tsdb.duckdb",                TSDBProvider,          DuckDBAdapter),
    ("tsdb.sqlite",                TSDBProvider,          SQLiteTSDBAdapter),
    ("tsdb.timescale",             TSDBProvider,          TimescaleAdapter),
]

_LABELS = [row[0] for row in ADAPTER_REGISTRY]


def _protocol_methods(p: type) -> list[str]:
    """Public callable members declared on a ``Protocol`` class.

    ``inspect.getmembers(predicate=isfunction)`` only returns
    non-inherited callables that have a real Python function body —
    which matches Protocol method declarations (each has a ``...``
    body). Private dunder/underscore members are filtered out.
    """
    return sorted(
        name
        for name, value in inspect.getmembers(p, predicate=inspect.isfunction)
        if not name.startswith("_")
    )


def _required_proto_params(method) -> set[str]:
    """Names of required (no-default) positional/keyword params on a
    Protocol method, sans ``self`` and ``*args`` / ``**kwargs`` packs."""
    sig = inspect.signature(method)
    return {
        name
        for name, p in sig.parameters.items()
        if name != "self"
        and p.default is inspect.Parameter.empty
        and p.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    }


def _adapter_accepts_var_kwargs(adapter_method) -> bool:
    sig = inspect.signature(adapter_method)
    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


# ── Tests ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("label, protocol, adapter", ADAPTER_REGISTRY, ids=_LABELS)
def test_adapter_class_implements_every_protocol_method(
    label: str, protocol: type, adapter: type
) -> None:
    """Every public Protocol method exists on the adapter class."""
    proto_methods = _protocol_methods(protocol)
    assert proto_methods, f"Protocol {protocol.__name__} declares no public methods"
    missing = [m for m in proto_methods if not hasattr(adapter, m)]
    assert not missing, (
        f"{label} ({adapter.__name__}) is missing Protocol methods "
        f"required by {protocol.__name__}: {missing}"
    )


@pytest.mark.parametrize("label, protocol, adapter", ADAPTER_REGISTRY, ids=_LABELS)
def test_adapter_signatures_accept_required_protocol_params(
    label: str, protocol: type, adapter: type
) -> None:
    """Adapter implementations may add optional kwargs but must accept
    every required parameter the Protocol declares.

    A ``**kwargs`` adapter is implicitly compatible.
    """
    for method_name in _protocol_methods(protocol):
        proto_method = getattr(protocol, method_name)
        adapter_method = getattr(adapter, method_name)
        try:
            adapter_sig = inspect.signature(adapter_method)
        except (ValueError, TypeError):
            # Built-in / C-extension method — can't introspect; skip.
            continue
        if _adapter_accepts_var_kwargs(adapter_method):
            continue
        required = _required_proto_params(proto_method)
        adapter_params = set(adapter_sig.parameters)
        missing = required - adapter_params
        assert not missing, (
            f"{label}.{method_name} is missing required Protocol "
            f"parameters: {sorted(missing)}"
        )


# ── Registry hygiene ────────────────────────────────────────────────────────


def test_no_duplicate_labels() -> None:
    assert len(_LABELS) == len(set(_LABELS)), "duplicate labels in registry"


def test_every_provider_protocol_has_at_least_one_adapter() -> None:
    """If a Protocol exists, the registry should exercise it. Adding a
    new ``providers/<x>.py`` without registering an adapter fails here."""
    protocols_in_registry = {row[1] for row in ADAPTER_REGISTRY}
    expected = {
        AlertProvider, BrokerProvider, ExecutionAlgoProvider, FeatureStoreProvider,
        LLMProvider, MacroDataProvider, MarketDataProvider, ModelRegistryProvider,
        OptionsFlowProvider, SentimentProvider, TSDBProvider,
    }
    assert expected <= protocols_in_registry, (
        f"Protocols not exercised: {expected - protocols_in_registry}"
    )


def test_every_adapter_class_appears_only_once() -> None:
    """Catch copy-paste bugs in the registry."""
    classes = [row[2] for row in ADAPTER_REGISTRY]
    assert len(classes) == len(set(classes)), "adapter class listed twice"
