class MockLLMAdapter:
    @property
    def model_name(self) -> str:
        return "mock-llm"

    def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        return f"[mock response to: {prompt[:50]}]"

    def embed(self, text: str) -> list[float]:
        return [0.0] * 384
