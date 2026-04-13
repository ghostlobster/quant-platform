from __future__ import annotations

import os
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...
    def complete(self, prompt: str, *, system: str = "", max_tokens: int = 1024, temperature: float = 0.0) -> str: ...
    def embed(self, text: str) -> list[float]: ...


def get_llm(provider: Optional[str] = None) -> LLMProvider:
    name = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if name == "anthropic":
        from adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter()
    elif name == "openai":
        from adapters.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter()
    elif name == "ollama":
        from adapters.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter()
    elif name == "mock":
        from adapters.llm.mock_adapter import MockLLMAdapter
        return MockLLMAdapter()
    raise ValueError(f"Unknown LLM provider: {name!r}. Valid: anthropic, openai, ollama, mock")
