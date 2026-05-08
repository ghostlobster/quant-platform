"""
adapters/auth/google_adapter.py — Google OAuth2 adapter.

ENV vars (all required when GOOGLE_CLIENT_ID is set)
----------------------------------------------------
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    APP_BASE_URL   (default: http://localhost:8501)
"""
from __future__ import annotations

import os
from typing import Optional
from urllib.parse import urlencode

import requests
import structlog

log = structlog.get_logger(__name__)

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_SCOPES = "openid email profile"


class GoogleOAuthAdapter:
    """Google OAuth2 provider using the Authorization Code flow."""

    def __init__(self) -> None:
        self._client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self._client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        base_url = os.environ.get("APP_BASE_URL", "http://localhost:8501").rstrip("/")
        self._redirect_uri = f"{base_url}/"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": self._client_id,
            "redirect_uri": self._redirect_uri,
            "response_type": "code",
            "scope": _SCOPES,
            "state": state,
            "access_type": "online",
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
                    "grant_type": "authorization_code",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("access_token")
        except Exception as exc:
            log.warning("google_oauth: token exchange failed", error=str(exc))
            return None

    def get_user_info(self, token: str) -> Optional[dict]:
        try:
            resp = requests.get(
                _USERINFO_URL,
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "id": data.get("sub", ""),
                "name": data.get("name", data.get("email", "")),
                "email": data.get("email", ""),
                "avatar_url": data.get("picture", ""),
                "provider": "google",
            }
        except Exception as exc:
            log.warning("google_oauth: userinfo fetch failed", error=str(exc))
            return None
