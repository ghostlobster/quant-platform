"""
tests/test_channels.py — Unit tests for alerts/channels.py

All network I/O is mocked; no real requests are sent.
"""
import os
import smtplib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from alerts.channels import (
    EmailChannel,
    TelegramChannel,
    WebhookChannel,
    broadcast,
    get_configured_channels,
)

# ── get_configured_channels ───────────────────────────────────────────────────

def test_get_configured_channels_empty_when_no_env_vars(monkeypatch):
    """Returns empty list when no relevant env vars are set."""
    for var in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_TO",
        "WEBHOOK_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    channels = get_configured_channels()
    assert channels == []


def test_get_configured_channels_telegram_only(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok123")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat456")
    for var in ["EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_TO", "WEBHOOK_URL"]:
        monkeypatch.delenv(var, raising=False)

    channels = get_configured_channels()
    assert len(channels) == 1
    assert isinstance(channels[0], TelegramChannel)


def test_get_configured_channels_email_only(monkeypatch):
    monkeypatch.setenv("EMAIL_USERNAME", "user@example.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "secret")
    monkeypatch.setenv("EMAIL_TO", "dest@example.com")
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "WEBHOOK_URL"]:
        monkeypatch.delenv(var, raising=False)

    channels = get_configured_channels()
    assert len(channels) == 1
    assert isinstance(channels[0], EmailChannel)


def test_get_configured_channels_webhook_only(monkeypatch):
    monkeypatch.setenv("WEBHOOK_URL", "https://hooks.example.com/notify")
    for var in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_TO",
    ]:
        monkeypatch.delenv(var, raising=False)

    channels = get_configured_channels()
    assert len(channels) == 1
    assert isinstance(channels[0], WebhookChannel)


def test_get_configured_channels_all_three(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "cid")
    monkeypatch.setenv("EMAIL_USERNAME", "u@x.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("EMAIL_TO", "t@x.com")
    monkeypatch.setenv("WEBHOOK_URL", "https://example.com/hook")

    channels = get_configured_channels()
    assert len(channels) == 3


def test_email_channel_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("EMAIL_USERNAME", "u@x.com")
    monkeypatch.setenv("EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("EMAIL_TO", "t@x.com")
    monkeypatch.delenv("EMAIL_SMTP_HOST", raising=False)
    monkeypatch.delenv("EMAIL_SMTP_PORT", raising=False)
    for var in ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "WEBHOOK_URL"]:
        monkeypatch.delenv(var, raising=False)

    channels = get_configured_channels()
    ch = channels[0]
    assert isinstance(ch, EmailChannel)
    assert ch.smtp_host == "smtp.gmail.com"
    assert ch.smtp_port == 587


# ── TelegramChannel.send ──────────────────────────────────────────────────────

class _MockHTTPResponse:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_telegram_send_returns_true_on_200(monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(200),
    )
    ch = TelegramChannel(bot_token="tok", chat_id="cid")
    assert ch.send("Subject", "Body") is True


def test_telegram_send_returns_false_on_non_200(monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(400),
    )
    ch = TelegramChannel(bot_token="tok", chat_id="cid")
    assert ch.send("Subject", "Body") is False


def test_telegram_send_returns_false_on_network_error(monkeypatch):
    import urllib.request

    def _raise(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    ch = TelegramChannel(bot_token="tok", chat_id="cid")
    assert ch.send("Subject", "Body") is False


# ── EmailChannel.send ─────────────────────────────────────────────────────────

class _MockSMTP:
    """Minimal SMTP stub that records calls."""

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def ehlo(self):
        pass

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def sendmail(self, from_addr, to_addrs, msg):
        pass


def test_email_send_returns_true_on_success(monkeypatch):
    monkeypatch.setattr(smtplib, "SMTP", _MockSMTP)
    ch = EmailChannel(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="u@g.com",
        password="pw",
        to_address="dest@g.com",
    )
    assert ch.send("Subject", "Body") is True


def test_email_send_returns_false_on_smtp_error(monkeypatch):
    def _bad_smtp(*args, **kwargs):
        raise smtplib.SMTPException("connection refused")

    monkeypatch.setattr(smtplib, "SMTP", _bad_smtp)
    ch = EmailChannel(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="u@g.com",
        password="pw",
        to_address="dest@g.com",
    )
    assert ch.send("Subject", "Body") is False


def test_email_send_returns_false_on_network_error(monkeypatch):
    def _bad_smtp(*args, **kwargs):
        raise OSError("network unreachable")

    monkeypatch.setattr(smtplib, "SMTP", _bad_smtp)
    ch = EmailChannel(
        smtp_host="smtp.gmail.com",
        smtp_port=587,
        username="u@g.com",
        password="pw",
        to_address="dest@g.com",
    )
    assert ch.send("Subject", "Body") is False


# ── WebhookChannel.send ───────────────────────────────────────────────────────

def test_webhook_send_returns_true_on_2xx(monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(200),
    )
    ch = WebhookChannel(url="https://hooks.example.com/notify")
    assert ch.send("Subject", "Body") is True


def test_webhook_send_returns_false_on_non_2xx(monkeypatch):
    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda req, timeout=None: _MockHTTPResponse(500),
    )
    ch = WebhookChannel(url="https://hooks.example.com/notify")
    assert ch.send("Subject", "Body") is False


def test_webhook_send_returns_false_on_network_error(monkeypatch):
    import urllib.request

    def _raise(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    ch = WebhookChannel(url="https://hooks.example.com/notify")
    assert ch.send("Subject", "Body") is False


# ── broadcast ─────────────────────────────────────────────────────────────────

class _OkChannel:
    def send(self, subject, body):
        return True


class _FailChannel:
    def send(self, subject, body):
        return False


def test_broadcast_counts_sent_and_failed():
    result = broadcast("S", "B", channels=[_OkChannel(), _OkChannel(), _FailChannel()])
    assert result["sent"] == 2
    assert result["failed"] == 1
    assert result["channels"] == ["_OkChannel", "_OkChannel", "_FailChannel"]


def test_broadcast_all_sent():
    result = broadcast("S", "B", channels=[_OkChannel(), _OkChannel()])
    assert result["sent"] == 2
    assert result["failed"] == 0


def test_broadcast_all_failed():
    result = broadcast("S", "B", channels=[_FailChannel(), _FailChannel()])
    assert result["sent"] == 0
    assert result["failed"] == 2


def test_broadcast_continues_after_one_channel_fails():
    """A failing channel must not prevent subsequent channels from being called."""
    calls = []

    class _TrackedOk:
        def send(self, subject, body):
            calls.append("ok")
            return True

    class _TrackedFail:
        def send(self, subject, body):
            calls.append("fail")
            return False

    result = broadcast("S", "B", channels=[_TrackedFail(), _TrackedOk(), _TrackedFail(), _TrackedOk()])
    assert calls == ["fail", "ok", "fail", "ok"]
    assert result["sent"] == 2
    assert result["failed"] == 2


def test_broadcast_continues_if_channel_raises():
    """Even if send() raises unexpectedly, broadcast() keeps going."""

    class _RaisingChannel:
        def send(self, subject, body):
            raise RuntimeError("unexpected!")

    result = broadcast("S", "B", channels=[_RaisingChannel(), _OkChannel()])
    assert result["sent"] == 1
    assert result["failed"] == 1


def test_broadcast_empty_channels_list():
    result = broadcast("S", "B", channels=[])
    assert result == {"sent": 0, "failed": 0, "channels": []}


def test_broadcast_uses_get_configured_channels_when_none_passed(monkeypatch):
    """When channels=None, broadcast() calls get_configured_channels()."""
    for var in [
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "EMAIL_USERNAME", "EMAIL_PASSWORD", "EMAIL_TO",
        "WEBHOOK_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    result = broadcast("S", "B")
    assert result == {"sent": 0, "failed": 0, "channels": []}
