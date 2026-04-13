from __future__ import annotations

import os


class OpenAIAdapter:
    def __init__(self) -> None:
        try:
            import openai

            self._client = openai.OpenAI()
            self._model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            self._embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        except ImportError as e:
            raise ImportError("openai SDK not installed. Run: pip install openai") from e

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
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system or "You are a quantitative trading assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._embed_model, input=text)
        return resp.data[0].embedding
