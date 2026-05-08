"""
auth/session.py — Streamlit session-state helpers for OAuth authentication.

All state is stored in st.session_state under the key ``"user"``.
The value is a dict ``{id, name, email, avatar_url, provider}`` when
authenticated, or absent / None when not.

Public API
----------
    is_authenticated() → bool
    get_user()         → dict | None
    logout()           → None
"""
from __future__ import annotations

from typing import Optional

import streamlit as st


def is_authenticated() -> bool:
    """Return True when a user dict is present in session state."""
    return bool(st.session_state.get("user"))


def get_user() -> Optional[dict]:
    """Return the authenticated user dict, or None if not logged in."""
    return st.session_state.get("user")


def logout() -> None:
    """Clear authentication state from the current session."""
    st.session_state.pop("user", None)
