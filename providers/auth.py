"""
providers/auth.py — OAuthProvider protocol and factory.

ENV vars
--------
    GOOGLE_CLIENT_ID       — enables Google OAuth when set
    GOOGLE_CLIENT_SECRET   — required alongside GOOGLE_CLIENT_ID
    GITHUB_CLIENT_ID       — enables GitHub OAuth when set
    GITHUB_CLIENT_SECRET   — required alongside GITHUB_CLIENT_ID
    APP_BASE_URL           — redirect URI base (default: http://localhost:8501)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class OAuthProvider(Protocol):
    """Duck-typed interface for a single OAuth2 provider."""

    def get_auth_url(self, state: str) -> str:
        """Return the provider's authorization URL with state embedded."""
        ...

    def exchange_code(self, code: str) -> Optional[str]:
        """Exchange an authorization code for an access token. Returns None on failure."""
        ...

    def get_user_info(self, token: str) -> Optional[dict]:
        """Fetch normalized user info using an access token. Returns None on failure."""
        ...


def get_oauth_providers() -> dict[str, OAuthProvider]:
    """
    Return a dict of enabled OAuthProvider adapters keyed by provider name.

    A provider is enabled when its CLIENT_ID env var is non-empty.
    Both Google and GitHub may be active simultaneously.

    Returns
    -------
    dict[str, OAuthProvider]
        Keys are ``"google"`` and/or ``"github"``.  Empty dict when no
        credentials are configured (common in unit-test environments).
    """
    providers: dict[str, OAuthProvider] = {}

    if os.environ.get("GOOGLE_CLIENT_ID", "").strip():
        from adapters.auth.google_adapter import GoogleOAuthAdapter
        providers["google"] = GoogleOAuthAdapter()

    if os.environ.get("GITHUB_CLIENT_ID", "").strip():
        from adapters.auth.github_adapter import GitHubOAuthAdapter
        providers["github"] = GitHubOAuthAdapter()

    return providers
