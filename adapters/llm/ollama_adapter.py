from __future__ import annotations

import os


class OllamaAdapter:
    def __init__(self) -> None:
        import requests  # stdlib-compatible, always available

        self._requests = requests
        self._base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model = os.environ.get("OLLAMA_MODEL", "llama3")

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
        payload: dict = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        r = self._requests.post(f"{self._base_url}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["response"]

    def embed(self, text: str) -> list[float]:
        r = self._requests.post(
            f"{self._base_url}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]
