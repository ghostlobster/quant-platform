"""
tests/test_auth_oauth.py — Unit tests for auth/oauth.py flow.

All external dependencies (requests, SQLite) are mocked so tests run
offline and in-process without touching quant.db.
"""
from __future__ import annotations

import sqlite3
import time
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────

class _PersistentConn:
    """Thin wrapper around sqlite3.Connection that turns close() into a no-op.

    Production code calls close() after each operation; this keeps the
    in-memory connection alive so tests can inspect state afterward.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __getattr__(self, name: str):
        return getattr(self._conn, name)

    def __enter__(self):
        return self._conn.__enter__()

    def __exit__(self, *args):
        return self._conn.__exit__(*args)

    def close(self) -> None:
        pass  # keep alive for test inspection


def _make_in_memory_conn() -> _PersistentConn:
    """Return an in-memory SQLite connection with the oauth_states table."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.row_factory = sqlite3.Row
    raw.execute("""
        CREATE TABLE oauth_states (
            state      TEXT PRIMARY KEY,
            provider   TEXT NOT NULL,
            created_at REAL NOT NULL
        )
    """)
    raw.commit()
    return _PersistentConn(raw)


def _seed_state(conn, state: str, provider: str, age_seconds: float = 0.0) -> None:
    conn.execute(
        "INSERT INTO oauth_states (state, provider, created_at) VALUES (?, ?, ?)",
        (state, provider, time.time() - age_seconds),
    )
    conn.commit()


def _make_adapter(auth_url="https://provider.example/auth", token="tok123", user=None):
    """Build a minimal fake OAuthProvider adapter."""
    adapter = MagicMock()
    adapter.get_auth_url.return_value = auth_url
    adapter.exchange_code.return_value = token
    adapter.get_user_info.return_value = user or {
        "id": "1",
        "name": "Alice",
        "email": "alice@example.com",
        "avatar_url": "",
        "provider": "google",
    }
    return adapter


# ── get_auth_url ───────────────────────────────────────────────────────────────

class TestGetAuthUrl:
    def test_returns_provider_auth_url(self):
        conn = _make_in_memory_conn()
        adapter = _make_adapter(auth_url="https://accounts.google.com/o/oauth2/v2/auth?state=x")
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import get_auth_url
            url = get_auth_url("google")
        assert "accounts.google.com" in url

    def test_stores_state_in_db(self):
        conn = _make_in_memory_conn()
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import get_auth_url
            get_auth_url("google")
        row = conn.execute("SELECT * FROM oauth_states").fetchone()
        assert row is not None
        assert row["provider"] == "google"

    def test_raises_for_unknown_provider(self):
        conn = _make_in_memory_conn()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={}),
        ):
            from auth.oauth import get_auth_url
            with pytest.raises(ValueError, match="not enabled"):
                get_auth_url("google")

    def test_purges_expired_states_on_call(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "old-state", "google", age_seconds=700)
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import get_auth_url
            get_auth_url("google")
        # expired row should be gone; only the fresh row should remain
        rows = conn.execute("SELECT state FROM oauth_states").fetchall()
        states = [r["state"] for r in rows]
        assert "old-state" not in states


# ── handle_callback ────────────────────────────────────────────────────────────

class TestHandleCallback:
    def test_happy_path_returns_user(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "valid-state", "google")
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import handle_callback
            user = handle_callback("auth-code-123", "valid-state")
        assert user is not None
        assert user["email"] == "alice@example.com"
        assert user["provider"] == "google"

    def test_state_consumed_after_use(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "one-shot", "github")
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"github": adapter}),
        ):
            from auth.oauth import handle_callback
            handle_callback("code", "one-shot")
        row = conn.execute("SELECT * FROM oauth_states WHERE state='one-shot'").fetchone()
        assert row is None

    def test_unknown_state_returns_none(self):
        conn = _make_in_memory_conn()
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import handle_callback
            result = handle_callback("code", "nonexistent-state")
        assert result is None

    def test_expired_state_returns_none(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "stale", "google", age_seconds=700)
        adapter = _make_adapter()
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import handle_callback
            result = handle_callback("code", "stale")
        assert result is None

    def test_token_exchange_failure_returns_none(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "st8", "google")
        adapter = _make_adapter(token=None)
        adapter.exchange_code.return_value = None
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"google": adapter}),
        ):
            from auth.oauth import handle_callback
            result = handle_callback("code", "st8")
        assert result is None

    def test_userinfo_failure_returns_none(self):
        conn = _make_in_memory_conn()
        _seed_state(conn, "st9", "github")
        adapter = _make_adapter()
        adapter.get_user_info.return_value = None
        with (
            patch("auth.oauth.get_connection", return_value=conn),
            patch("auth.oauth.get_oauth_providers", return_value={"github": adapter}),
        ):
            from auth.oauth import handle_callback
            result = handle_callback("code", "st9")
        assert result is None


# ── get_enabled_provider_names ─────────────────────────────────────────────────

class TestGetEnabledProviderNames:
    def test_empty_when_no_credentials(self):
        with patch("auth.oauth.get_oauth_providers", return_value={}):
            from auth.oauth import get_enabled_provider_names
            assert get_enabled_provider_names() == []

    def test_returns_configured_providers(self):
        providers = {"google": MagicMock(), "github": MagicMock()}
        with patch("auth.oauth.get_oauth_providers", return_value=providers):
            from auth.oauth import get_enabled_provider_names
            names = get_enabled_provider_names()
        assert set(names) == {"google", "github"}
