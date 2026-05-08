"""
tests/test_auth_adapters.py — Unit tests for OAuth adapter classes and provider factory.

Covers adapters/auth/google_adapter.py, adapters/auth/github_adapter.py,
and providers/auth.py so the changed-module coverage gate (≥ 85%) passes.
All HTTP calls are mocked; no network access required.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# ── GoogleOAuthAdapter ─────────────────────────────────────────────────────────

class TestGoogleOAuthAdapter:
    def _make_adapter(self, client_id="gid", client_secret="gsec", base_url="http://localhost:8501"):
        env = {
            "GOOGLE_CLIENT_ID": client_id,
            "GOOGLE_CLIENT_SECRET": client_secret,
            "APP_BASE_URL": base_url,
        }
        with patch.dict(os.environ, env, clear=False):
            from adapters.auth.google_adapter import GoogleOAuthAdapter
            return GoogleOAuthAdapter()

    def test_get_auth_url_contains_google_domain(self):
        adapter = self._make_adapter()
        url = adapter.get_auth_url("test-state")
        assert "accounts.google.com" in url
        assert "test-state" in url
        assert "gid" in url

    def test_get_auth_url_includes_redirect_uri(self):
        adapter = self._make_adapter(base_url="http://localhost:8501")
        url = adapter.get_auth_url("s")
        assert "redirect_uri" in url
        assert "localhost" in url

    def test_exchange_code_returns_access_token(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "tok-abc"}
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.google_adapter.requests.post", return_value=mock_resp) as mock_post:
            token = adapter.exchange_code("auth-code")
        assert token == "tok-abc"
        mock_post.assert_called_once()

    def test_exchange_code_returns_none_on_http_error(self):
        adapter = self._make_adapter()
        with patch("adapters.auth.google_adapter.requests.post", side_effect=Exception("timeout")):
            result = adapter.exchange_code("bad-code")
        assert result is None

    def test_get_user_info_returns_normalized_dict(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "sub": "123",
            "name": "Alice",
            "email": "alice@gmail.com",
            "picture": "https://pic.url/a.jpg",
        }
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.google_adapter.requests.get", return_value=mock_resp):
            user = adapter.get_user_info("tok-abc")
        assert user["id"] == "123"
        assert user["name"] == "Alice"
        assert user["email"] == "alice@gmail.com"
        assert user["avatar_url"] == "https://pic.url/a.jpg"
        assert user["provider"] == "google"

    def test_get_user_info_falls_back_to_email_for_name(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "sub": "42",
            "email": "bob@gmail.com",
            # no "name" key
        }
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.google_adapter.requests.get", return_value=mock_resp):
            user = adapter.get_user_info("tok")
        assert user["name"] == "bob@gmail.com"

    def test_get_user_info_returns_none_on_http_error(self):
        adapter = self._make_adapter()
        with patch("adapters.auth.google_adapter.requests.get", side_effect=Exception("503")):
            result = adapter.get_user_info("tok")
        assert result is None


# ── GitHubOAuthAdapter ─────────────────────────────────────────────────────────

class TestGitHubOAuthAdapter:
    def _make_adapter(self, client_id="ghid", client_secret="ghsec", base_url="http://localhost:8501"):
        env = {
            "GITHUB_CLIENT_ID": client_id,
            "GITHUB_CLIENT_SECRET": client_secret,
            "APP_BASE_URL": base_url,
        }
        with patch.dict(os.environ, env, clear=False):
            from adapters.auth.github_adapter import GitHubOAuthAdapter
            return GitHubOAuthAdapter()

    def test_get_auth_url_contains_github_domain(self):
        adapter = self._make_adapter()
        url = adapter.get_auth_url("gh-state")
        assert "github.com/login/oauth/authorize" in url
        assert "gh-state" in url
        assert "ghid" in url

    def test_exchange_code_returns_access_token(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "ghtok"}
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.github_adapter.requests.post", return_value=mock_resp):
            token = adapter.exchange_code("code")
        assert token == "ghtok"

    def test_exchange_code_returns_none_on_error(self):
        adapter = self._make_adapter()
        with patch("adapters.auth.github_adapter.requests.post", side_effect=Exception("err")):
            result = adapter.exchange_code("code")
        assert result is None

    def test_get_user_info_returns_normalized_dict_with_public_email(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 99,
            "name": "Bob",
            "login": "bob99",
            "email": "bob@example.com",
            "avatar_url": "https://avatars.example.com/99",
        }
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.github_adapter.requests.get", return_value=mock_resp):
            user = adapter.get_user_info("ghtok")
        assert user["id"] == "99"
        assert user["name"] == "Bob"
        assert user["email"] == "bob@example.com"
        assert user["provider"] == "github"

    def test_get_user_info_fetches_primary_email_when_not_public(self):
        adapter = self._make_adapter()
        user_resp = MagicMock()
        user_resp.json.return_value = {
            "id": 7,
            "name": "Carol",
            "login": "carol",
            "email": None,
            "avatar_url": "",
        }
        user_resp.raise_for_status.return_value = None

        emails_resp = MagicMock()
        emails_resp.json.return_value = [
            {"email": "carol@private.com", "primary": True, "verified": True},
            {"email": "carol@other.com", "primary": False, "verified": True},
        ]
        emails_resp.raise_for_status.return_value = None

        responses = [user_resp, emails_resp]
        with patch("adapters.auth.github_adapter.requests.get", side_effect=responses):
            user = adapter.get_user_info("ghtok")
        assert user["email"] == "carol@private.com"

    def test_get_user_info_falls_back_to_login_for_name(self):
        adapter = self._make_adapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "id": 5,
            "name": None,
            "login": "dave42",
            "email": "dave@example.com",
            "avatar_url": "",
        }
        mock_resp.raise_for_status.return_value = None
        with patch("adapters.auth.github_adapter.requests.get", return_value=mock_resp):
            user = adapter.get_user_info("tok")
        assert user["name"] == "dave42"

    def test_get_user_info_returns_none_on_http_error(self):
        adapter = self._make_adapter()
        with patch("adapters.auth.github_adapter.requests.get", side_effect=Exception("net")):
            result = adapter.get_user_info("tok")
        assert result is None

    def test_fetch_primary_email_returns_none_on_error(self):
        adapter = self._make_adapter()
        user_resp = MagicMock()
        user_resp.json.return_value = {
            "id": 8, "name": "Eve", "login": "eve", "email": None, "avatar_url": "",
        }
        user_resp.raise_for_status.return_value = None

        emails_resp = MagicMock()
        emails_resp.raise_for_status.side_effect = Exception("403 Forbidden")

        with patch("adapters.auth.github_adapter.requests.get", side_effect=[user_resp, emails_resp]):
            user = adapter.get_user_info("tok")
        assert user is not None
        assert user["email"] == ""


# ── providers/auth.py factory ──────────────────────────────────────────────────

class TestGetOAuthProviders:
    def test_empty_when_no_env_vars(self):
        with patch.dict(os.environ, {"GOOGLE_CLIENT_ID": "", "GITHUB_CLIENT_ID": ""}, clear=False):
            from providers.auth import get_oauth_providers
            providers = get_oauth_providers()
        assert providers == {}

    def test_google_enabled_when_client_id_set(self):
        env = {"GOOGLE_CLIENT_ID": "gid", "GOOGLE_CLIENT_SECRET": "gsec"}
        with patch.dict(os.environ, env, clear=False):
            # Ensure GitHub is not accidentally enabled
            os.environ.pop("GITHUB_CLIENT_ID", None)
            from providers.auth import get_oauth_providers
            providers = get_oauth_providers()
        assert "google" in providers
        assert "github" not in providers

    def test_github_enabled_when_client_id_set(self):
        env = {"GITHUB_CLIENT_ID": "ghid", "GITHUB_CLIENT_SECRET": "ghsec"}
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("GOOGLE_CLIENT_ID", None)
            from providers.auth import get_oauth_providers
            providers = get_oauth_providers()
        assert "github" in providers
        assert "google" not in providers

    def test_both_enabled_when_both_client_ids_set(self):
        env = {
            "GOOGLE_CLIENT_ID": "gid", "GOOGLE_CLIENT_SECRET": "gsec",
            "GITHUB_CLIENT_ID": "ghid", "GITHUB_CLIENT_SECRET": "ghsec",
        }
        with patch.dict(os.environ, env, clear=False):
            from providers.auth import get_oauth_providers
            providers = get_oauth_providers()
        assert "google" in providers
        assert "github" in providers

    def test_adapters_satisfy_protocol(self):
        from providers.auth import OAuthProvider
        env = {
            "GOOGLE_CLIENT_ID": "gid", "GOOGLE_CLIENT_SECRET": "gsec",
            "GITHUB_CLIENT_ID": "ghid", "GITHUB_CLIENT_SECRET": "ghsec",
        }
        with patch.dict(os.environ, env, clear=False):
            from providers.auth import get_oauth_providers
            providers = get_oauth_providers()
        for name, adapter in providers.items():
            assert isinstance(adapter, OAuthProvider), f"{name} adapter does not satisfy OAuthProvider"
