"""
adapters/llm/openai_adapter.py — Wraps the OpenAI SDK.

Requires:  pip install openai
ENV vars:  OPENAI_API_KEY
           OPENAI_MODEL        (default: gpt-4o-mini)
           OPENAI_EMBED_MODEL  (default: text-embedding-3-small)
"""
from __future__ import annotations

import os

try:
    import openai as _openai
except ImportError:
    _openai = None  # type: ignore[assignment]


class OpenAIAdapter:
    """LLMProvider backed by OpenAI Chat Completions and Embeddings APIs."""

    def __init__(
        self,
        model: str | None = None,
        embed_model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if _openai is None:
            raise ImportError(
                "openai package is required for OpenAIAdapter. "
                "Install it with: pip install openai"
            )
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._embed_model = embed_model or os.environ.get(
            "OPENAI_EMBED_MODEL", "text-embedding-3-small"
        )
        self._client = _openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
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
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=self._embed_model,
            input=text,
        )
        return resp.data[0].embedding
