import os
from unittest.mock import patch

import pytest


def test_get_llm_mock():
    from providers import LLMProvider, get_llm
    llm = get_llm("mock")
    assert isinstance(llm, LLMProvider)
    result = llm.complete("hello")
    assert isinstance(result, str)


def test_get_llm_env_var():
    with patch.dict(os.environ, {"LLM_PROVIDER": "mock"}):
        from providers import get_llm
        llm = get_llm()
        assert llm.model_name == "mock-llm"


def test_get_llm_unknown_raises():
    from providers import get_llm
    with pytest.raises(ValueError, match="Unknown"):
        get_llm("nonexistent_provider")


def test_get_broker_paper():
    from providers import BrokerProvider, get_broker
    broker = get_broker("paper")
    assert isinstance(broker, BrokerProvider)
    info = broker.get_account_info()
    assert isinstance(info, dict)
    assert "equity" in info or "cash" in info or "buying_power" in info


def test_get_alert_noop():
    from providers import AlertProvider, get_alert_channel
    alert = get_alert_channel("noop")
    assert isinstance(alert, AlertProvider)
    result = alert.send("test message", level="info")
    assert result is True


def test_get_sentiment_mock():
    from providers import SentimentProvider, get_sentiment
    s = get_sentiment("mock")
    assert isinstance(s, SentimentProvider)
    score = s.score("This stock is amazing!")
    assert isinstance(score, float)


def test_get_execution_algo_market():
    from providers import ExecutionAlgoProvider, get_execution_algo
    algo = get_execution_algo("market")
    assert isinstance(algo, ExecutionAlgoProvider)


def test_get_tsdb_sqlite(tmp_path):
    with patch.dict(os.environ, {"SQLITE_DB_PATH": str(tmp_path / "test.db")}):
        import importlib

        import adapters.tsdb.sqlite_adapter as mod
        importlib.reload(mod)
        from providers import TSDBProvider, get_tsdb
        db = get_tsdb("sqlite")
        assert isinstance(db, TSDBProvider)
        db.create_table("test_tbl", "id INTEGER PRIMARY KEY, val TEXT")
        db.write("test_tbl", [{"id": 1, "val": "hello"}])
        rows = db.query("SELECT * FROM test_tbl WHERE id = ?", (1,))
        assert rows[0]["val"] == "hello"
        db.close()


def test_vader_adapter_skip_if_not_installed():
    pytest.importorskip("vaderSentiment", reason="vaderSentiment not installed")
    from adapters.sentiment.vader_adapter import VaderSentimentAdapter
    v = VaderSentimentAdapter()
    assert -1.0 <= v.score("Great earnings beat!") <= 1.0


def test_get_market_data_mock():
    from providers import MarketDataProvider, get_market_data
    md = get_market_data("mock")
    assert isinstance(md, MarketDataProvider)
    bars = md.get_bars("AAPL", "1Day", "2024-01-01", "2024-01-31")
    assert isinstance(bars, list)


def test_paper_broker_place_order():
    from providers import get_broker
    broker = get_broker("paper")
    order = broker.place_order("AAPL", 10, "buy")
    assert isinstance(order, dict)


def test_paper_broker_get_positions():
    from providers import get_broker
    broker = get_broker("paper")
    positions = broker.get_positions()
    assert isinstance(positions, list)


def test_paper_broker_get_orders():
    from providers import get_broker
    broker = get_broker("paper")
    orders = broker.get_orders()
    assert isinstance(orders, list)


def test_execution_algo_market_order():
    from providers import ExecutionAlgoProvider, get_broker, get_execution_algo
    algo = get_execution_algo("market")
    assert isinstance(algo, ExecutionAlgoProvider)
    broker = get_broker("paper")
    results = algo.execute("AAPL", 5, "buy", broker)
    assert isinstance(results, list)
    assert len(results) >= 1


def test_unknown_providers_raise_value_error():
    from providers import (
        get_alert_channel,
        get_broker,
        get_execution_algo,
        get_llm,
        get_market_data,
        get_sentiment,
        get_tsdb,
    )
    for factory, name in [
        (get_llm, "bad_llm"),
        (get_broker, "bad_broker"),
        (get_market_data, "bad_md"),
        (get_alert_channel, "bad_alert"),
        (get_sentiment, "bad_sent"),
        (get_execution_algo, "bad_algo"),
        (get_tsdb, "bad_db"),
    ]:
        with pytest.raises(ValueError):
            factory(name)


def test_mock_llm_embed_shape():
    from providers import get_llm
    llm = get_llm("mock")
    emb = llm.embed("hello world")
    assert len(emb) > 0 and all(isinstance(x, float) for x in emb)


def test_noop_alert_all_levels():
    from providers import get_alert_channel
    alert = get_alert_channel("noop")
    for level in ("info", "warning", "error"):
        assert alert.send("msg", level=level) is True


def test_mock_sentiment_full():
    from providers import get_sentiment
    s = get_sentiment("mock")
    assert isinstance(s.score("great"), float)
    assert len(s.batch_score(["a", "b"])) == 2
    assert isinstance(s.ticker_sentiment("AAPL"), float)
