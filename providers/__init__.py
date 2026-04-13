"""
providers — vendor-flexible interface layer.

Import protocols and factory functions from here; never import vendor SDKs
directly in business logic.

Usage
-----
    from providers import get_llm, get_broker, get_market_data
    llm    = get_llm()          # reads LLM_PROVIDER env var
    broker = get_broker()       # reads BROKER_PROVIDER env var
    md     = get_market_data()  # reads MARKET_DATA_PROVIDER env var
"""
from providers.llm import LLMProvider, get_llm
from providers.market_data import MarketDataProvider, get_market_data
from providers.broker import BrokerProvider, get_broker
from providers.alert import AlertProvider, get_alert_channel
from providers.sentiment import SentimentProvider, get_sentiment
from providers.execution_algo import ExecutionAlgoProvider, get_execution_algo
from providers.tsdb import TSDBProvider, get_tsdb
from providers.feature_store import FeatureStoreProvider, get_feature_store

__all__ = [
    "LLMProvider", "get_llm",
    "MarketDataProvider", "get_market_data",
    "BrokerProvider", "get_broker",
    "AlertProvider", "get_alert_channel",
    "SentimentProvider", "get_sentiment",
    "ExecutionAlgoProvider", "get_execution_algo",
    "TSDBProvider", "get_tsdb",
    "FeatureStoreProvider", "get_feature_store",
]
