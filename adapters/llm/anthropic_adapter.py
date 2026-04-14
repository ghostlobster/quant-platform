"""
adapters/llm/anthropic_adapter.py — Wraps the Anthropic SDK.

Requires:  pip install anthropic
ENV vars:  ANTHROPIC_API_KEY, ANTHROPIC_MODEL (default: claude-opus-4-6)
"""
from __future__ import annotations

import os

try:
    import anthropic as _anthropic
except ImportError:
    _anthropic = None  # type: ignore[assignment]


class AnthropicAdapter:
    """LLMProvider backed by Anthropic Messages API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if _anthropic is None:
            raise ImportError(
                "anthropic package is required for AnthropicAdapter. "
                "Install it with: pip install anthropic"
            )
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        return msg.content[0].text

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError(
            "Anthropic does not currently expose an embeddings endpoint. "
            "Use openai or ollama for embeddings."
        )
