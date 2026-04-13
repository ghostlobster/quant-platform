"""Tests for structured logging configuration (structlog)."""
import structlog


def test_configure_logging_console_mode(monkeypatch):
    """configure_logging() runs without error in console mode."""
    monkeypatch.setenv("LOG_FORMAT", "console")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    # Reset structlog state so configure is re-applied
    structlog.reset_defaults()
    from config import configure_logging
    configure_logging()  # must not raise


def test_configure_logging_json_mode(monkeypatch):
    """configure_logging() runs without error in json mode."""
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    structlog.reset_defaults()
    from config import configure_logging
    configure_logging()  # must not raise


def test_logger_methods_do_not_raise(monkeypatch, capsys):
    """structlog logger .info/.warning/.error/.debug all run without raising."""
    monkeypatch.setenv("LOG_FORMAT", "console")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    structlog.reset_defaults()
    from config import configure_logging
    configure_logging()

    logger = structlog.get_logger("test.logger")
    logger.debug("debug message", key="value")
    logger.info("info message", count=1)
    logger.warning("warning message")
    logger.error("error message", code=500)


def test_context_vars_bind_and_clear():
    """bind_contextvars and clear_contextvars work correctly."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(run_id="abc123", component="test")

    # Bound vars are accessible
    ctx = structlog.contextvars.get_contextvars()
    assert ctx["run_id"] == "abc123"
    assert ctx["component"] == "test"

    # Clear removes them
    structlog.contextvars.clear_contextvars()
    ctx_after = structlog.contextvars.get_contextvars()
    assert ctx_after == {}


def test_utils_logger_get_logger_returns_structlog_logger():
    """utils.logger.get_logger returns a structlog BoundLogger."""
    from utils.logger import get_logger
    logger = get_logger("test.module")
    # structlog bound loggers have an .info method
    assert callable(logger.info)
    assert callable(logger.warning)
    assert callable(logger.error)
