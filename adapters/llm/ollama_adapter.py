"""
adapters/llm/ollama_adapter.py — Wraps the local Ollama REST API.

No extra SDK needed — uses the stdlib ``urllib`` (or ``requests`` if
installed) to call the Ollama HTTP server.

ENV vars
--------
    OLLAMA_BASE_URL   (default: http://localhost:11434)
    OLLAMA_MODEL      (default: llama3)
    OLLAMA_EMBED_MODEL  (default: nomic-embed-text)
"""
from __future__ import annotations

import json
import os

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


class OllamaAdapter:
    """LLMProvider backed by a local Ollama server."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL", "llama3")
        self._embed_model = embed_model or os.environ.get(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text"
        )

    @property
    def model_name(self) -> str:
        return self._model

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode()
        if _requests is not None:
            resp = _requests.post(url, data=data, timeout=120)
            resp.raise_for_status()
            return resp.json()
        # Fallback to urllib
        import urllib.request

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

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
        result = self._post("/api/generate", payload)
        return result.get("response", "")

    def embed(self, text: str) -> list[float]:
        result = self._post(
            "/api/embeddings",
            {"model": self._embed_model, "prompt": text},
        )
        return result.get("embedding", [])
