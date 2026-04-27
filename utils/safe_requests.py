"""
utils/safe_requests.py — defensive HTTP wrapper.

Wraps :mod:`requests` with three operator-controlled mitigations against
the open ``urllib3`` HIGH-severity advisories (CVE-2026-21441,
CVE-2025-66471, CVE-2025-66418) that have no upstream fix yet:

* **Max content-length cap** — every response is checked against
  :func:`max_response_bytes` (env ``HTTP_MAX_RESPONSE_BYTES``, default
  16 MB) before its body is materialised; oversized responses raise
  ``ResponseTooLarge`` rather than streaming through urllib3's
  decompression path.
* **Timeout floor** — every call carries a default timeout
  (``HTTP_DEFAULT_TIMEOUT_SECONDS``, default 15s) so a hung endpoint
  cannot pin a worker.
* **Optional redirect cap** — pass ``allow_redirects=False`` (the default
  for ``post``) to dodge the redirect-decompression-bomb chain entirely
  on flows that do not need redirects.

This is intentionally a thin wrapper. It does **not** add automatic
retries, session pooling, or auth — adapters that need those keep using
``requests`` directly. The wrapper is for the trusted-API call path
(Polygon, Alpaca, Tradier, Slack, Ollama) where the only attacker is a
compromised upstream.
"""
from __future__ import annotations

import os

import structlog

logger = structlog.get_logger(__name__)

_DEFAULT_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB
_DEFAULT_TIMEOUT_S = 15.0


class ResponseTooLarge(RuntimeError):
    """Raised when a response exceeds the configured size cap."""


def max_response_bytes() -> int:
    raw = os.environ.get("HTTP_MAX_RESPONSE_BYTES")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MAX_BYTES
    try:
        return int(raw)
    except ValueError:
        logger.warning("safe_requests: bad HTTP_MAX_RESPONSE_BYTES", value=raw)
        return _DEFAULT_MAX_BYTES


def default_timeout() -> float:
    raw = os.environ.get("HTTP_DEFAULT_TIMEOUT_SECONDS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_TIMEOUT_S
    try:
        return float(raw)
    except ValueError:
        logger.warning("safe_requests: bad HTTP_DEFAULT_TIMEOUT_SECONDS", value=raw)
        return _DEFAULT_TIMEOUT_S


def _enforce_size(response, *, max_bytes: int) -> None:
    """Raise ``ResponseTooLarge`` if the response is over ``max_bytes``.

    Checks the declared ``Content-Length`` first (cheap), then falls back
    to ``len(response.content)`` after the body is materialised — useful
    when servers omit the header.
    """
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise ResponseTooLarge(
                    f"declared Content-Length {declared} > {max_bytes}",
                )
        except ValueError:
            pass
    body = response.content
    if len(body) > max_bytes:
        raise ResponseTooLarge(
            f"materialised body {len(body)} > {max_bytes} bytes",
        )


def _request(
    method: str,
    url: str,
    *,
    timeout: float | None = None,
    max_bytes: int | None = None,
    allow_redirects: bool | None = None,
    **kwargs,
):
    """Make a request with safe defaults.

    Parameters
    ----------
    method, url, **kwargs
        Forwarded to :func:`requests.request`.
    timeout
        Override the env-driven default.
    max_bytes
        Override :func:`max_response_bytes`. Set to ``None`` to keep the
        env default; set to ``0`` to disable the cap (not recommended).
    allow_redirects
        ``None`` keeps the per-method default (``True`` for ``GET``,
        ``False`` for ``POST``). Override explicitly when the caller
        needs redirects on a POST flow.
    """
    import requests

    effective_timeout = timeout if timeout is not None else default_timeout()
    effective_max = max_bytes if max_bytes is not None else max_response_bytes()
    if allow_redirects is None:
        allow_redirects = method.upper() == "GET"

    response = requests.request(
        method, url,
        timeout=effective_timeout,
        allow_redirects=allow_redirects,
        **kwargs,
    )
    if effective_max > 0:
        _enforce_size(response, max_bytes=effective_max)
    return response


def get(url: str, **kwargs):
    return _request("GET", url, **kwargs)


def post(url: str, **kwargs):
    return _request("POST", url, **kwargs)


def delete(url: str, **kwargs):
    return _request("DELETE", url, **kwargs)
