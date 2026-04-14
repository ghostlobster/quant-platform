"""
providers/llm.py — LLMProvider protocol and factory.

ENV vars
--------
    LLM_PROVIDER   mock | anthropic | openai | ollama  (default: mock)
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    OLLAMA_BASE_URL  (default: http://localhost:11434)
"""
from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Duck-typed interface for large language model completions and embeddings."""

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Return a text completion for *prompt*."""
        ...

    def embed(self, text: str) -> list[float]:
        """Return a vector embedding for *text*."""
        ...

    @property
    def model_name(self) -> str:
        """Human-readable model identifier."""
        ...


def get_llm(provider: Optional[str] = None) -> LLMProvider:
    """
    Return a configured LLMProvider adapter.

    Parameters
    ----------
    provider : str, optional
        Override the LLM_PROVIDER env var.  One of:
        ``anthropic``, ``openai``, ``ollama``, ``mock``.

    Raises
    ------
    ValueError
        If the provider name is not recognised.
    ImportError
        If the required SDK is not installed (raised at adapter instantiation
        time, not at import time).
    """
    name = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower().strip()
    if name == "anthropic":
        from adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter()
    if name == "openai":
        from adapters.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter()
    if name == "ollama":
        from adapters.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter()
    if name == "mock":
        from adapters.llm.mock_adapter import MockLLMAdapter
        return MockLLMAdapter()
    raise ValueError(
        f"Unknown LLM provider: {name!r}. "
        "Valid options: anthropic, openai, ollama, mock"
    )
