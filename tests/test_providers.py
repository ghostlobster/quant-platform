import os

import pytest
from unittest.mock import patch


def test_get_llm_mock():
    from providers import get_llm, LLMProvider
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
    from providers import get_broker, BrokerProvider
    broker = get_broker("paper")
    assert isinstance(broker, BrokerProvider)
    info = broker.get_account_info()
    assert isinstance(info, dict)
    assert "equity" in info or "cash" in info or "buying_power" in info


def test_get_alert_noop():
    from providers import get_alert_channel, AlertProvider
    alert = get_alert_channel("noop")
    assert isinstance(alert, AlertProvider)
    result = alert.send("test message", level="info")
    assert result is True


def test_get_sentiment_mock():
    from providers import get_sentiment, SentimentProvider
    s = get_sentiment("mock")
    assert isinstance(s, SentimentProvider)
    score = s.score("This stock is amazing!")
    assert isinstance(score, float)


def test_get_execution_algo_market():
    from providers import get_execution_algo, ExecutionAlgoProvider
    algo = get_execution_algo("market")
    assert isinstance(algo, ExecutionAlgoProvider)


def test_get_tsdb_sqlite(tmp_path):
    with patch.dict(os.environ, {"SQLITE_DB_PATH": str(tmp_path / "test.db")}):
        import importlib
        import adapters.tsdb.sqlite_adapter as mod
        importlib.reload(mod)
        from providers import get_tsdb, TSDBProvider
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
