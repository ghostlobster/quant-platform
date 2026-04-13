"""Vendor-flexible provider factory. Import from here, not from adapters directly."""
from providers.alert import AlertProvider, get_alert_channel
from providers.broker import BrokerProvider, get_broker
from providers.execution_algo import ExecutionAlgoProvider, get_execution_algo
from providers.llm import LLMProvider, get_llm
from providers.market_data import MarketDataProvider, get_market_data
from providers.sentiment import SentimentProvider, get_sentiment
from providers.tsdb import TSDBProvider, get_tsdb

__all__ = [
    "LLMProvider", "get_llm",
    "MarketDataProvider", "get_market_data",
    "BrokerProvider", "get_broker",
    "AlertProvider", "get_alert_channel",
    "SentimentProvider", "get_sentiment",
    "ExecutionAlgoProvider", "get_execution_algo",
    "TSDBProvider", "get_tsdb",
]
