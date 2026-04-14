"""
tests/test_providers.py — Integration-style tests for the provider protocol layer.

All tests use only mock/paper/noop adapters — no network calls, no credentials.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure project root is importable when running pytest from any cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ── LLM ──────────────────────────────────────────────────────────────────────

class TestLLMProvider:
    def test_mock_adapter_satisfies_protocol(self):
        from providers.llm import LLMProvider, get_llm
        llm = get_llm("mock")
        assert isinstance(llm, LLMProvider)

    def test_mock_complete_returns_string(self):
        from providers.llm import get_llm
        llm = get_llm("mock")
        result = llm.complete("hello world")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mock_embed_returns_floats(self):
        from providers.llm import get_llm
        llm = get_llm("mock")
        vec = llm.embed("test text")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_mock_model_name_property(self):
        from providers.llm import get_llm
        llm = get_llm("mock")
        assert isinstance(llm.model_name, str)
        assert len(llm.model_name) > 0

    def test_unknown_provider_raises_value_error(self):
        from providers.llm import get_llm
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm("notarealllm")

    def test_env_var_wiring(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        from providers.llm import LLMProvider, get_llm
        llm = get_llm()  # no args — reads env var
        assert isinstance(llm, LLMProvider)

    def test_complete_preview_truncation(self):
        from providers.llm import get_llm
        llm = get_llm("mock")
        long_prompt = "x" * 200
        result = llm.complete(long_prompt)
        assert "mock::" in result

    def test_anthropic_importable_without_sdk(self):
        """Adapter module must be importable even if anthropic SDK is absent."""
        import importlib
        mod = importlib.import_module("adapters.llm.anthropic_adapter")
        assert hasattr(mod, "AnthropicAdapter")

    def test_openai_importable_without_sdk(self):
        import importlib
        mod = importlib.import_module("adapters.llm.openai_adapter")
        assert hasattr(mod, "OpenAIAdapter")


# ── Market Data ───────────────────────────────────────────────────────────────

class TestMarketDataProvider:
    def test_mock_adapter_satisfies_protocol(self):
        from providers.market_data import MarketDataProvider, get_market_data
        md = get_market_data("mock")
        assert isinstance(md, MarketDataProvider)

    def test_mock_get_bars_returns_list(self):
        from providers.market_data import get_market_data
        md = get_market_data("mock")
        bars = md.get_bars("AAPL", "1Day", "2024-01-01", "2024-01-05")
        assert isinstance(bars, list)
        assert len(bars) > 0
        first = bars[0]
        for key in ("t", "o", "h", "l", "c", "v"):
            assert key in first, f"Missing key {key!r} in bar"

    def test_mock_get_quote_returns_dict(self):
        from providers.market_data import get_market_data
        md = get_market_data("mock")
        q = md.get_quote("SPY")
        assert isinstance(q, dict)
        assert q["symbol"] == "SPY"

    def test_mock_get_quotes_multi(self):
        from providers.market_data import get_market_data
        md = get_market_data("mock")
        qs = md.get_quotes(["AAPL", "MSFT"])
        assert set(qs.keys()) == {"AAPL", "MSFT"}

    def test_unknown_provider_raises_value_error(self):
        from providers.market_data import get_market_data
        with pytest.raises(ValueError, match="Unknown market data provider"):
            get_market_data("bloomberg")

    def test_env_var_wiring(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "mock")
        from providers.market_data import MarketDataProvider, get_market_data
        md = get_market_data()
        assert isinstance(md, MarketDataProvider)


# ── Broker ────────────────────────────────────────────────────────────────────

class TestBrokerProvider:
    def test_paper_adapter_satisfies_protocol(self):
        from providers.broker import BrokerProvider, get_broker
        broker = get_broker("paper")
        assert isinstance(broker, BrokerProvider)

    def test_paper_get_account_info_returns_dict(self):
        from providers.broker import get_broker
        broker = get_broker("paper")
        info = broker.get_account_info()
        assert isinstance(info, dict)

    def test_paper_get_positions_returns_list(self):
        from providers.broker import get_broker
        broker = get_broker("paper")
        positions = broker.get_positions()
        assert isinstance(positions, list)

    def test_paper_place_order_returns_dict_with_order_id(self):
        from providers.broker import get_broker
        broker = get_broker("paper")
        result = broker.place_order("AAPL", 1.0, "buy", limit_price=100.0)
        assert isinstance(result, dict)
        assert "order_id" in result

    def test_paper_get_orders_returns_list(self):
        from providers.broker import get_broker
        broker = get_broker("paper")
        orders = broker.get_orders("all")
        assert isinstance(orders, list)

    def test_unknown_provider_raises_value_error(self):
        from providers.broker import get_broker
        with pytest.raises(ValueError, match="Unknown broker provider"):
            get_broker("robinhood")

    def test_env_var_wiring(self, monkeypatch):
        monkeypatch.setenv("BROKER_PROVIDER", "paper")
        from providers.broker import BrokerProvider, get_broker
        broker = get_broker()
        assert isinstance(broker, BrokerProvider)


# ── Alert ─────────────────────────────────────────────────────────────────────

class TestAlertProvider:
    def test_noop_adapter_satisfies_protocol(self):
        from providers.alert import AlertProvider, get_alert_channel
        alert = get_alert_channel("noop")
        assert isinstance(alert, AlertProvider)

    def test_noop_send_returns_true(self):
        from providers.alert import get_alert_channel
        alert = get_alert_channel("noop")
        assert alert.send("test message") is True

    def test_noop_send_with_level_and_channel(self):
        from providers.alert import get_alert_channel
        alert = get_alert_channel("noop")
        assert alert.send("boom", level="error", channel="#ops") is True

    def test_unknown_provider_raises_value_error(self):
        from providers.alert import get_alert_channel
        with pytest.raises(ValueError, match="Unknown alert provider"):
            get_alert_channel("pager")

    def test_env_var_wiring(self, monkeypatch):
        monkeypatch.setenv("ALERT_PROVIDER", "noop")
        from providers.alert import AlertProvider, get_alert_channel
        alert = get_alert_channel()
        assert isinstance(alert, AlertProvider)


# ── Sentiment ─────────────────────────────────────────────────────────────────

class TestSentimentProvider:
    def test_mock_adapter_satisfies_protocol(self):
        from providers.sentiment import SentimentProvider, get_sentiment
        s = get_sentiment("mock")
        assert isinstance(s, SentimentProvider)

    def test_mock_score_returns_float_in_range(self):
        from providers.sentiment import get_sentiment
        s = get_sentiment("mock")
        score = s.score("The stock is surging on earnings beat")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_mock_batch_score_length_matches(self):
        from providers.sentiment import get_sentiment
        s = get_sentiment("mock")
        texts = ["good", "bad", "neutral"]
        scores = s.batch_score(texts)
        assert len(scores) == len(texts)

    def test_mock_ticker_sentiment_returns_float(self):
        from providers.sentiment import get_sentiment
        s = get_sentiment("mock")
        score = s.ticker_sentiment("AAPL")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    @pytest.mark.skipif(
        "vaderSentiment" not in sys.modules
        and not __import__("importlib.util", fromlist=["find_spec"]).find_spec("vaderSentiment"),
        reason="vaderSentiment not installed",
    )
    def test_vader_adapter_satisfies_protocol(self):
        from providers.sentiment import SentimentProvider, get_sentiment
        s = get_sentiment("vader")
        assert isinstance(s, SentimentProvider)

    def test_vader_adapter_falls_back_gracefully(self):
        """VADERAdapter must work even without vaderSentiment (uses lexicon fallback)."""
        from adapters.sentiment.vader_adapter import VADERAdapter
        adapter = VADERAdapter()
        score = adapter.score("stocks rally on strong earnings beat")
        assert isinstance(score, float)

    def test_unknown_provider_raises_value_error(self):
        from providers.sentiment import get_sentiment
        with pytest.raises(ValueError, match="Unknown sentiment provider"):
            get_sentiment("finbert")


# ── Execution Algo ────────────────────────────────────────────────────────────

class TestExecutionAlgoProvider:
    def _paper_broker(self):
        from providers.broker import get_broker
        return get_broker("paper")

    def test_market_algo_satisfies_protocol(self):
        from providers.execution_algo import ExecutionAlgoProvider, get_execution_algo
        algo = get_execution_algo("market")
        assert isinstance(algo, ExecutionAlgoProvider)

    def test_market_algo_execute_returns_execution_result(self):
        from adapters.execution_algo.result import ExecutionResult
        from providers.execution_algo import get_execution_algo
        algo = get_execution_algo("market")
        broker = self._paper_broker()
        result = algo.execute("AAPL", 5.0, "buy", broker)
        assert isinstance(result, ExecutionResult)
        assert result.symbol == "AAPL"
        assert result.total_qty == pytest.approx(5.0)

    def test_unknown_algo_raises_value_error(self):
        from providers.execution_algo import get_execution_algo
        with pytest.raises(ValueError, match="Unknown execution algo"):
            get_execution_algo("iceberg")

    def test_env_var_wiring(self, monkeypatch):
        monkeypatch.setenv("EXECUTION_ALGO", "market")
        from providers.execution_algo import ExecutionAlgoProvider, get_execution_algo
        algo = get_execution_algo()
        assert isinstance(algo, ExecutionAlgoProvider)


# ── TSDB ──────────────────────────────────────────────────────────────────────

class TestTSDBProvider:
    def test_sqlite_adapter_satisfies_protocol(self, tmp_path):
        from adapters.tsdb.sqlite_adapter import SQLiteTSDBAdapter
        from providers.tsdb import TSDBProvider
        db = SQLiteTSDBAdapter(path=str(tmp_path / "test.db"))
        assert isinstance(db, TSDBProvider)
        db.close()

    def test_sqlite_write_and_query(self, tmp_path):
        # Each test needs its own connection (module-level singleton would bleed across)
        import adapters.tsdb.sqlite_adapter as mod
        from adapters.tsdb.sqlite_adapter import SQLiteTSDBAdapter
        orig = mod._connection
        mod._connection = None
        try:
            db = SQLiteTSDBAdapter(path=str(tmp_path / "test2.db"))
            db.create_table("prices", "symbol TEXT, ts TEXT, close REAL")
            db.write("prices", [
                {"symbol": "AAPL", "ts": "2024-01-01", "close": 190.5},
                {"symbol": "MSFT", "ts": "2024-01-01", "close": 375.0},
            ])
            rows = db.query("SELECT * FROM prices ORDER BY symbol")
            assert len(rows) == 2
            assert rows[0]["symbol"] == "AAPL"
            db.close()
        finally:
            mod._connection = orig

    def test_unknown_tsdb_raises_value_error(self):
        from providers.tsdb import get_tsdb
        with pytest.raises(ValueError, match="Unknown TSDB provider"):
            get_tsdb("influxdb")


# ── Feature Store ─────────────────────────────────────────────────────────────

class TestFeatureStoreProvider:
    def test_memory_adapter_satisfies_protocol(self):
        from providers.feature_store import FeatureStoreProvider, get_feature_store
        fs = get_feature_store("memory")
        assert isinstance(fs, FeatureStoreProvider)

    def test_memory_set_and_get(self):
        import adapters.feature_store.memory_adapter as mod
        from providers.feature_store import get_feature_store
        # Isolate from other tests by clearing the shared store
        mod._store.clear()
        fs = get_feature_store("memory")
        fs.set_features("AAPL", {"rsi_14": 65.3, "macd": 0.42})
        result = fs.get_features("AAPL", ["rsi_14", "macd", "missing"])
        assert result["rsi_14"] == 65.3
        assert result["macd"] == 0.42
        assert "missing" not in result

    def test_memory_list_features(self):
        import adapters.feature_store.memory_adapter as mod
        from providers.feature_store import get_feature_store
        mod._store.clear()
        fs = get_feature_store("memory")
        fs.set_features("SPY", {"vol_20": 0.15})
        fs.set_features("QQQ", {"vol_20": 0.18, "beta": 1.1})
        names = fs.list_features()
        assert "vol_20" in names
        assert "beta" in names

    def test_unknown_feature_store_raises_value_error(self):
        from providers.feature_store import get_feature_store
        with pytest.raises(ValueError, match="Unknown feature store provider"):
            get_feature_store("feast")


# ── Protocol isinstance checks (runtime_checkable) ───────────────────────────

class TestProtocolRuntimeCheck:
    """Verify runtime_checkable isinstance works for all 7 protocols."""

    def test_all_mock_adapters_pass_isinstance(self):
        from providers.alert import AlertProvider, get_alert_channel
        from providers.broker import BrokerProvider, get_broker
        from providers.execution_algo import ExecutionAlgoProvider, get_execution_algo
        from providers.feature_store import FeatureStoreProvider, get_feature_store
        from providers.llm import LLMProvider, get_llm
        from providers.market_data import MarketDataProvider, get_market_data
        from providers.sentiment import SentimentProvider, get_sentiment

        assert isinstance(get_llm("mock"), LLMProvider)
        assert isinstance(get_market_data("mock"), MarketDataProvider)
        assert isinstance(get_broker("paper"), BrokerProvider)
        assert isinstance(get_alert_channel("noop"), AlertProvider)
        assert isinstance(get_sentiment("mock"), SentimentProvider)
        assert isinstance(get_execution_algo("market"), ExecutionAlgoProvider)
        assert isinstance(get_feature_store("memory"), FeatureStoreProvider)
