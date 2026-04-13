from __future__ import annotations

import os


class AnthropicAdapter:
    def __init__(self) -> None:
        try:
            import anthropic

            self._client = anthropic.Anthropic()
            self._model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        except ImportError as e:
            raise ImportError("anthropic SDK not installed. Run: pip install anthropic") from e

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
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system or "You are a quantitative trading assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    def embed(self, text: str) -> list[float]:
        # Anthropic doesn't have embeddings yet; placeholder
        return [0.0] * 1536
