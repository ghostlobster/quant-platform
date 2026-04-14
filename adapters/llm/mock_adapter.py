"""
adapters/llm/mock_adapter.py — Deterministic mock LLM for tests.

No external SDK required.  Responses are predictable so tests can
assert on exact outputs without network calls.
"""
from __future__ import annotations


class MockLLMAdapter:
    """
    Deterministic mock that echoes prompts back.

    ``complete()`` returns ``"mock::<first 40 chars of prompt>"``
    ``embed()``    returns a fixed 8-element unit vector.
    """

    _MODEL = "mock-llm-v0"

    @property
    def model_name(self) -> str:
        return self._MODEL

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        preview = (prompt[:40] + "...") if len(prompt) > 40 else prompt
        return f"mock::{preview}"

    def embed(self, text: str) -> list[float]:
        # Reproducible 8-d vector; magnitude ≈ 1.0
        return [0.125] * 8
