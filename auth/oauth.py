"""
auth/oauth.py — OAuth2 Authorization Code flow orchestration.

Manages CSRF state tokens in the oauth_states table (quant.db) and delegates
protocol-specific URL construction / token exchange to the provider adapters.

Public API
----------
    get_auth_url(provider_name)          → str
    handle_callback(code, state)         → dict | None
    get_enabled_provider_names()         → list[str]
"""
from __future__ import annotations

import secrets
import sqlite3
import time
from typing import Optional

import structlog

from data.db import get_connection
from providers.auth import OAuthProvider, get_oauth_providers

log = structlog.get_logger(__name__)

_STATE_TTL_SECONDS = 600  # 10 minutes


# ── State store helpers (all accept an open connection) ────────────────────────

def _store_state(conn: sqlite3.Connection, state: str, provider_name: str) -> None:
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO oauth_states (state, provider, created_at) VALUES (?, ?, ?)",
            (state, provider_name, time.time()),
        )


def _validate_and_consume_state(conn: sqlite3.Connection, state: str) -> Optional[str]:
    """Return provider name if state is valid and not expired; delete it on success."""
    row = conn.execute(
        "SELECT provider, created_at FROM oauth_states WHERE state = ?", (state,)
    ).fetchone()
    if row is None:
        log.warning("oauth: unknown state", state=state[:8])
        return None
    provider_name, created_at = row["provider"], row["created_at"]
    if time.time() - created_at > _STATE_TTL_SECONDS:
        log.warning("oauth: expired state", provider=provider_name)
        _delete_state(conn, state)
        return None
    _delete_state(conn, state)
    return provider_name


def _delete_state(conn: sqlite3.Connection, state: str) -> None:
    with conn:
        conn.execute("DELETE FROM oauth_states WHERE state = ?", (state,))


def _purge_expired_states(conn: sqlite3.Connection) -> None:
    """Remove stale rows; called opportunistically on each auth URL generation."""
    with conn:
        conn.execute(
            "DELETE FROM oauth_states WHERE created_at < ?",
            (time.time() - _STATE_TTL_SECONDS,),
        )


# ── Adapter lookup ─────────────────────────────────────────────────────────────

def _get_adapter(provider_name: str) -> OAuthProvider:
    providers = get_oauth_providers()
    if provider_name not in providers:
        raise ValueError(
            f"OAuth provider {provider_name!r} is not enabled. "
            f"Set {provider_name.upper()}_CLIENT_ID in the environment."
        )
    return providers[provider_name]


# ── Public API ─────────────────────────────────────────────────────────────────

def get_enabled_provider_names() -> list[str]:
    """Return names of providers that have credentials configured."""
    return list(get_oauth_providers().keys())


def get_auth_url(provider_name: str) -> str:
    """
    Generate a provider authorization URL embedding a fresh CSRF state token.

    The state is persisted to quant.db and validated in handle_callback.
    Expired states are pruned opportunistically on each call.

    Raises ValueError if the provider is not enabled.
    """
    conn = get_connection()
    try:
        _purge_expired_states(conn)
        state = secrets.token_urlsafe(32)
        _store_state(conn, state, provider_name)
    finally:
        conn.close()

    adapter = _get_adapter(provider_name)
    url = adapter.get_auth_url(state)
    log.info("oauth: auth URL generated", provider=provider_name)
    return url


def handle_callback(code: str, state: str) -> Optional[dict]:
    """
    Complete the OAuth flow after the provider redirects back.

    Validates and consumes the state, exchanges the code for a token,
    and fetches normalized user info.

    Returns a user dict ``{id, name, email, avatar_url, provider}`` on
    success, or ``None`` on any failure (invalid state, network error, etc.).
    """
    conn = get_connection()
    try:
        provider_name = _validate_and_consume_state(conn, state)
    finally:
        conn.close()

    if provider_name is None:
        return None

    try:
        adapter = _get_adapter(provider_name)
    except ValueError:
        log.warning("oauth: adapter not found for provider", provider=provider_name)
        return None

    token = adapter.exchange_code(code)
    if not token:
        log.warning("oauth: token exchange returned no token", provider=provider_name)
        return None

    user = adapter.get_user_info(token)
    if user:
        log.info("oauth: user authenticated", provider=provider_name, email=user.get("email"))
    return user
