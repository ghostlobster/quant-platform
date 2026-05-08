"""
adapters/auth/github_adapter.py — GitHub OAuth2 adapter.

ENV vars (all required when GITHUB_CLIENT_ID is set)
----------------------------------------------------
    GITHUB_CLIENT_ID
    GITHUB_CLIENT_SECRET
    APP_BASE_URL   (default: http://localhost:8501)
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlencode

import requests
import structlog

log = structlog.get_logger(__name__)

_AUTH_URL = "https://github.com/login/oauth/authorize"
_TOKEN_URL = "https://github.com/login/oauth/access_token"
_USERINFO_URL = "https://api.github.com/user"
_EMAILS_URL = "https://api.github.com/user/emails"
_SCOPES = "read:user user:email"


class GitHubOAuthAdapter:
    """GitHub OAuth2 provider using the Authorization Code flow."""

    def __init__(self) -> None:
        self._client_id = os.environ.get("GITHUB_CLIENT_ID", "")
        self._client_secret = os.environ.get("GITHUB_CLIENT_SECRET", "")
        base_url = os.environ.get("APP_BASE_URL", "http://localhost:8501").rstrip("/")
        self._redirect_uri = f"{base_url}/"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "scope": _SCOPES,
            "state": state,
        }
        return f"{_AUTH_URL}?{urlencode(params)}"

    def exchange_code(self, code: str) -> Optional[str]:
        try:
            resp = requests.post(
                _TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret,
                    "redirect_uri": self._redirect_uri,
                },
                headers={"Accept": "application/json"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
        except Exception as exc:
            log.warning("github_oauth: token exchange failed", error=str(exc))
            return None

    def get_user_info(self, token: str) -> Optional[dict]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        try:
            resp = requests.get(_USERINFO_URL, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            # GitHub may not expose email publicly; fetch from the emails endpoint.
            email = data.get("email") or self._fetch_primary_email(headers)

            return {
                "id": str(data.get("id", "")),
                "name": data.get("name") or data.get("login", ""),
                "email": email or "",
                "avatar_url": data.get("avatar_url", ""),
                "provider": "github",
            }
        except Exception as exc:
            log.warning("github_oauth: userinfo fetch failed", error=str(exc))
            return None

    def _fetch_primary_email(self, headers: dict) -> Optional[str]:
        try:
            resp = requests.get(_EMAILS_URL, headers=headers, timeout=10)
            resp.raise_for_status()
            for entry in resp.json():
                if entry.get("primary") and entry.get("verified"):
                    return entry["email"]
        except Exception:
            pass
        return None
