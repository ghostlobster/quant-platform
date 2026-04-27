"""
tests/test_safe_requests.py — defensive HTTP wrapper.

Mocks out ``requests.request`` so no network fires. Asserts the
wrapper applies the documented safe defaults (timeout, redirect policy,
max-content-length cap) and surfaces ``ResponseTooLarge`` on bombs.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import safe_requests
from utils.safe_requests import (
    ResponseTooLarge,
    default_timeout,
    max_response_bytes,
)

# ── Env helpers ──────────────────────────────────────────────────────────────

def test_max_response_bytes_default(monkeypatch):
    monkeypatch.delenv("HTTP_MAX_RESPONSE_BYTES", raising=False)
    assert max_response_bytes() == 16 * 1024 * 1024


def test_max_response_bytes_env_override(monkeypatch):
    monkeypatch.setenv("HTTP_MAX_RESPONSE_BYTES", "1024")
    assert max_response_bytes() == 1024


def test_max_response_bytes_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("HTTP_MAX_RESPONSE_BYTES", "garbage")
    assert max_response_bytes() == 16 * 1024 * 1024


def test_default_timeout_default(monkeypatch):
    monkeypatch.delenv("HTTP_DEFAULT_TIMEOUT_SECONDS", raising=False)
    assert default_timeout() == 15.0


def test_default_timeout_env_override(monkeypatch):
    monkeypatch.setenv("HTTP_DEFAULT_TIMEOUT_SECONDS", "2.5")
    assert default_timeout() == 2.5


def test_default_timeout_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("HTTP_DEFAULT_TIMEOUT_SECONDS", "junk")
    assert default_timeout() == 15.0


# ── Mock requests.request ────────────────────────────────────────────────────

def _fake_response(content: bytes, *, content_length: int | None = None):
    headers = {}
    if content_length is not None:
        headers["Content-Length"] = str(content_length)
    return SimpleNamespace(content=content, headers=headers)


@pytest.fixture
def captured_request(monkeypatch):
    captured: dict = {}

    def _fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured.update(kwargs)
        body = captured.pop("_body", b"hi")
        cl = captured.pop("_content_length", None)
        return _fake_response(body, content_length=cl)

    fake_requests = SimpleNamespace(request=_fake_request)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    return captured


# ── GET / POST / DELETE round-trip ───────────────────────────────────────────

def test_get_passes_through_with_defaults(captured_request, monkeypatch):
    monkeypatch.delenv("HTTP_MAX_RESPONSE_BYTES", raising=False)
    monkeypatch.delenv("HTTP_DEFAULT_TIMEOUT_SECONDS", raising=False)

    r = safe_requests.get("https://example.test", params={"x": 1})
    assert captured_request["method"] == "GET"
    assert captured_request["url"] == "https://example.test"
    assert captured_request["timeout"] == 15.0
    assert captured_request["allow_redirects"] is True
    assert r.content == b"hi"


def test_post_disables_redirects_by_default(captured_request):
    safe_requests.post("https://example.test", data={"k": "v"})
    assert captured_request["method"] == "POST"
    assert captured_request["allow_redirects"] is False


def test_post_can_opt_into_redirects(captured_request):
    safe_requests.post("https://example.test", allow_redirects=True)
    assert captured_request["allow_redirects"] is True


def test_delete_round_trips(captured_request):
    safe_requests.delete("https://example.test/order/123")
    assert captured_request["method"] == "DELETE"


# ── Size enforcement ─────────────────────────────────────────────────────────

def test_oversized_declared_content_length_rejected(monkeypatch):
    """Header-based size check rejects before the body is read."""
    captured: dict = {}

    def _fake_request(method, url, **kwargs):
        captured["called"] = True
        return _fake_response(b"", content_length=10_000_000)

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_fake_request))
    with pytest.raises(ResponseTooLarge, match="declared"):
        safe_requests.get("https://example.test", max_bytes=1024)


def test_oversized_materialised_body_rejected(monkeypatch):
    """No Content-Length header → materialised body still capped."""
    big = b"a" * 5000

    def _fake_request(method, url, **kwargs):
        return _fake_response(big, content_length=None)

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_fake_request))
    with pytest.raises(ResponseTooLarge, match="materialised"):
        safe_requests.get("https://example.test", max_bytes=1024)


def test_under_cap_returns_response(monkeypatch):
    body = b"a" * 100

    def _fake_request(method, url, **kwargs):
        return _fake_response(body, content_length=100)

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_fake_request))
    r = safe_requests.get("https://example.test", max_bytes=1024)
    assert r.content == body


def test_max_bytes_zero_disables_check(monkeypatch):
    big = b"x" * 10_000_000

    def _fake_request(method, url, **kwargs):
        return _fake_response(big, content_length=10_000_000)

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_fake_request))
    # max_bytes=0 turns the cap off — size mitigation explicitly opted out.
    r = safe_requests.get("https://example.test", max_bytes=0)
    assert len(r.content) == 10_000_000


def test_invalid_declared_content_length_uses_body(monkeypatch):
    """Garbage Content-Length header → falls through to body-length check."""
    body = b"x" * 200
    headers = {"Content-Length": "abc"}
    response = SimpleNamespace(content=body, headers=headers)

    def _fake_request(method, url, **kwargs):
        return response

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(request=_fake_request))
    r = safe_requests.get("https://example.test", max_bytes=1024)
    assert r.content == body


# ── Custom timeout / kwargs ──────────────────────────────────────────────────

def test_custom_timeout_kwarg_overrides_env(captured_request, monkeypatch):
    monkeypatch.setenv("HTTP_DEFAULT_TIMEOUT_SECONDS", "5")
    safe_requests.get("https://example.test", timeout=1.5)
    assert captured_request["timeout"] == 1.5


def test_extra_kwargs_pass_through(captured_request):
    headers = {"Authorization": "Bearer x"}
    safe_requests.get("https://example.test", headers=headers, params={"a": 1})
    assert captured_request["headers"] == headers
    assert captured_request["params"] == {"a": 1}
