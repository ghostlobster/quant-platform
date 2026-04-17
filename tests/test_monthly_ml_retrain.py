"""
tests/test_monthly_ml_retrain.py — Unit tests for cron/monthly_ml_retrain.py.

All MLSignal.train() calls are mocked — no network access, no LightGBM required.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


_MOCK_RESULT = {
    "train_ic": 0.12,
    "test_ic": 0.09,
    "train_icir": 0.45,
    "test_icir": 0.38,
    "n_train_samples": 800,
    "n_test_samples": 200,
}


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_calls_train_with_default_tickers(mock_cls):
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    from cron.monthly_ml_retrain import main
    main()

    mock_instance.train.assert_called_once()
    call_args = mock_instance.train.call_args
    tickers = call_args[0][0]
    assert isinstance(tickers, list)
    assert len(tickers) > 0
    assert all(isinstance(t, str) for t in tickers)


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_respects_wf_tickers_env(mock_cls, monkeypatch=None):
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    os.environ["WF_TICKERS"] = "SPY,GLD,TLT"
    try:
        import cron.monthly_ml_retrain as mod
        # Reload to pick up fresh env; call main directly with patched env
        with patch.dict(os.environ, {"WF_TICKERS": "SPY,GLD,TLT"}):
            # Call main which reads the env var at runtime
            mod.main()
        call_args = mock_instance.train.call_args
        tickers = call_args[0][0]
        assert "SPY" in tickers
        assert "GLD" in tickers
        assert "TLT" in tickers
    finally:
        os.environ.pop("WF_TICKERS", None)


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_uses_ml_train_period_env(mock_cls):
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    with patch.dict(os.environ, {"ML_TRAIN_PERIOD": "1y"}):
        from cron.monthly_ml_retrain import main
        main()

    call_kwargs = mock_instance.train.call_args[1]
    assert call_kwargs.get("period") == "1y"


@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", False)
def test_main_exits_when_lgbm_unavailable():
    from cron.monthly_ml_retrain import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_exits_on_runtime_error(mock_cls):
    mock_instance = MagicMock()
    mock_instance.train.side_effect = RuntimeError("lgbm missing")
    mock_cls.return_value = mock_instance

    from cron.monthly_ml_retrain import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_exits_on_unexpected_exception(mock_cls):
    mock_instance = MagicMock()
    mock_instance.train.side_effect = ValueError("empty feature matrix")
    mock_cls.return_value = mock_instance

    from cron.monthly_ml_retrain import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("strategies.ml_tuning.load_best_params")
@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_passes_tuned_params_when_available(mock_cls, mock_load):
    """If load_best_params returns a dict, train() must receive it as lgbm_params."""
    tuned = {"n_estimators": 500, "learning_rate": 0.03}
    mock_load.return_value = tuned
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    from cron.monthly_ml_retrain import main
    main()

    call_kwargs = mock_instance.train.call_args[1]
    assert call_kwargs.get("lgbm_params") == tuned


@patch("strategies.ml_tuning.load_best_params", return_value=None)
@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_passes_none_when_no_tuned_params(mock_cls, mock_load):
    """No persisted params → train() is called with lgbm_params=None (defaults)."""
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    from cron.monthly_ml_retrain import main
    main()

    call_kwargs = mock_instance.train.call_args[1]
    assert call_kwargs.get("lgbm_params") is None


@patch("cron.monthly_ml_retrain.MLSignal")
@patch("cron.monthly_ml_retrain._LGBM_AVAILABLE", True)
def test_main_strips_whitespace_from_tickers(mock_cls):
    mock_instance = MagicMock()
    mock_instance.train.return_value = _MOCK_RESULT
    mock_cls.return_value = mock_instance

    with patch.dict(os.environ, {"WF_TICKERS": " AAPL , MSFT , TSLA "}):
        from cron.monthly_ml_retrain import main
        main()

    tickers = mock_instance.train.call_args[0][0]
    assert "AAPL" in tickers
    assert "MSFT" in tickers
    assert "TSLA" in tickers
    # Ensure no whitespace in ticker names
    assert all(t == t.strip() for t in tickers)
