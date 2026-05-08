"""
tests/test_auth_session.py — Unit tests for auth/session.py helpers.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

# ── Streamlit session_state mock ───────────────────────────────────────────────

class _FakeSessionState(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            return None

    def __setattr__(self, k, v):
        self[k] = v

    def pop(self, k, *args):
        return dict.pop(self, k, *args)


def _make_st_mock(state: dict | None = None):
    st = MagicMock()
    ss = _FakeSessionState(state or {})
    st.session_state = ss
    return st


# ── is_authenticated ───────────────────────────────────────────────────────────

class TestIsAuthenticated:
    def test_false_when_no_user(self):
        st_mock = _make_st_mock({})
        with patch("auth.session.st", st_mock):
            from auth.session import is_authenticated
            assert is_authenticated() is False

    def test_false_when_user_is_none(self):
        st_mock = _make_st_mock({"user": None})
        with patch("auth.session.st", st_mock):
            from auth.session import is_authenticated
            assert is_authenticated() is False

    def test_true_when_user_present(self):
        user = {"id": "42", "name": "Bob", "email": "bob@example.com", "provider": "github"}
        st_mock = _make_st_mock({"user": user})
        with patch("auth.session.st", st_mock):
            from auth.session import is_authenticated
            assert is_authenticated() is True


# ── get_user ───────────────────────────────────────────────────────────────────

class TestGetUser:
    def test_returns_none_when_not_authenticated(self):
        st_mock = _make_st_mock({})
        with patch("auth.session.st", st_mock):
            from auth.session import get_user
            assert get_user() is None

    def test_returns_user_dict(self):
        user = {"id": "1", "name": "Alice", "email": "alice@example.com", "provider": "google"}
        st_mock = _make_st_mock({"user": user})
        with patch("auth.session.st", st_mock):
            from auth.session import get_user
            assert get_user() == user


# ── logout ─────────────────────────────────────────────────────────────────────

class TestLogout:
    def test_clears_user_from_session(self):
        user = {"id": "1", "name": "Alice", "email": "alice@example.com", "provider": "google"}
        st_mock = _make_st_mock({"user": user})
        with patch("auth.session.st", st_mock):
            from auth.session import logout
            logout()
            assert st_mock.session_state.get("user") is None

    def test_logout_is_idempotent_when_no_user(self):
        st_mock = _make_st_mock({})
        with patch("auth.session.st", st_mock):
            from auth.session import logout
            logout()  # should not raise
