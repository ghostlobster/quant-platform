"""
tests/test_daily_ml_execute.py — Unit tests for cron/daily_ml_execute.py.

All MLSignal and broker calls are mocked — no network, no LightGBM required.
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _mock_model(scores=None):
    model = MagicMock()
    model._model = object()  # truthy — "model loaded"
    model.predict.return_value = scores or {"AAPL": 0.7, "MSFT": 0.4, "GOOG": -0.5}
    return model


@patch("cron.daily_ml_execute.execute_ml_signals")
@patch("cron.daily_ml_execute.MLSignal")
@patch("cron.daily_ml_execute._LGBM_AVAILABLE", True)
def test_main_happy_path(mock_cls, mock_exec):
    mock_cls.return_value = _mock_model()
    mock_exec.return_value = ["BUY AAPL x10", "SELL GOOG x5"]

    from cron.daily_ml_execute import main
    main()

    mock_cls.return_value.predict.assert_called_once()
    mock_exec.assert_called_once()
    # Default threshold and max positions pulled from env defaults.
    kwargs = mock_exec.call_args.kwargs
    assert kwargs["threshold"] == pytest.approx(0.3)
    assert kwargs["max_positions"] == 5


@patch("cron.daily_ml_execute.execute_ml_signals", return_value=[])
@patch("cron.daily_ml_execute.MLSignal")
@patch("cron.daily_ml_execute._LGBM_AVAILABLE", True)
def test_main_respects_env_overrides(mock_cls, mock_exec):
    mock_cls.return_value = _mock_model()

    env = {
        "WF_TICKERS": "AAPL,TSLA",
        "ML_SCORE_THRESHOLD": "0.5",
        "ML_MAX_POSITIONS": "2",
    }
    with patch.dict(os.environ, env):
        from cron.daily_ml_execute import main
        main()

    predict_args = mock_cls.return_value.predict.call_args
    assert predict_args[0][0] == ["AAPL", "TSLA"]
    exec_kwargs = mock_exec.call_args.kwargs
    assert exec_kwargs["threshold"] == pytest.approx(0.5)
    assert exec_kwargs["max_positions"] == 2


@patch("cron.daily_ml_execute._LGBM_AVAILABLE", False)
def test_main_exits_when_lgbm_unavailable():
    from cron.daily_ml_execute import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("cron.daily_ml_execute.MLSignal")
@patch("cron.daily_ml_execute._LGBM_AVAILABLE", True)
def test_main_exits_when_no_trained_model(mock_cls):
    model = MagicMock()
    model._model = None  # simulates no checkpoint
    mock_cls.return_value = model

    from cron.daily_ml_execute import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("cron.daily_ml_execute.execute_ml_signals")
@patch("cron.daily_ml_execute.MLSignal")
@patch("cron.daily_ml_execute._LGBM_AVAILABLE", True)
def test_main_exits_on_unexpected_exception(mock_cls, mock_exec):
    mock_cls.return_value = _mock_model()
    mock_exec.side_effect = RuntimeError("broker unreachable")

    from cron.daily_ml_execute import main
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


@patch("cron.daily_ml_execute.execute_ml_signals", return_value=[])
@patch("cron.daily_ml_execute.MLSignal")
@patch("cron.daily_ml_execute._LGBM_AVAILABLE", True)
def test_main_strips_whitespace_from_tickers(mock_cls, mock_exec):
    mock_cls.return_value = _mock_model()

    with patch.dict(os.environ, {"WF_TICKERS": " AAPL , MSFT , TSLA "}):
        from cron.daily_ml_execute import main
        main()

    tickers = mock_cls.return_value.predict.call_args[0][0]
    assert tickers == ["AAPL", "MSFT", "TSLA"]
