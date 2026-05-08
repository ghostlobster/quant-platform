"""
pages/login.py — OAuth login page.

Handles two responsibilities in a single render() call:
1. If the URL carries ``?code=...&state=...`` query params (OAuth callback),
   complete the token exchange and store the user in session state.
2. Otherwise show login buttons for every enabled OAuth provider.
"""
from __future__ import annotations

import streamlit as st

from auth.oauth import get_auth_url, get_enabled_provider_names, handle_callback

_PROVIDER_LABELS = {
    "google": "Login with Google",
    "github": "Login with GitHub",
}


def render() -> None:
    params = st.query_params

    # ── Handle OAuth callback ──────────────────────────────────────────────────
    if "code" in params and "state" in params:
        with st.spinner("Completing sign-in…"):
            user = handle_callback(params["code"], params["state"])
        if user:
            st.session_state["user"] = user
            st.query_params.clear()
            st.rerun()
        else:
            st.error("Sign-in failed. The link may have expired — please try again.")
            st.query_params.clear()

    # ── Login UI ───────────────────────────────────────────────────────────────
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("## Quant Platform")
        st.caption("Sign in to continue")
        st.divider()

        providers = get_enabled_provider_names()
        if not providers:
            st.warning(
                "No OAuth providers are configured. "
                "Set GOOGLE_CLIENT_ID or GITHUB_CLIENT_ID in your environment."
            )
            return

        for provider in providers:
            label = _PROVIDER_LABELS.get(provider, f"Login with {provider.title()}")
            url = get_auth_url(provider)
            st.link_button(label, url, use_container_width=True)
