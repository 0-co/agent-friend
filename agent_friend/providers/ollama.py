"""ollama.py — Ollama provider for agent-friend.

Connects to a local Ollama instance via its OpenAI-compatible API.
No API key required — just a running Ollama server.

Default endpoint: http://localhost:11434/v1
Models: whatever you've pulled (e.g. qwen2.5:3b, llama3.2, mistral)

    ollama pull qwen2.5:3b
    friend = Friend(model="qwen2.5:3b", provider="ollama")
"""

import os
from typing import Optional

from .openai import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    """LLM provider for local Ollama instances.

    Uses Ollama's OpenAI-compatible API. No API key needed.

    Parameters
    ----------
    api_key:
        Ignored (Ollama doesn't need auth). Accepts it for interface compat.
    base_url:
        Ollama API endpoint. Defaults to ``OLLAMA_HOST`` env var or
        ``http://localhost:11434/v1``.
    """

    DEFAULT_MODEL = "qwen2.5:3b"

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        super().__init__(api_key="ollama")  # Dummy key — Ollama ignores it
        host = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # Ensure /v1 suffix for OpenAI compat endpoint
        self._base_url = host.rstrip("/") + "/v1" if not host.endswith("/v1") else host

    def _get_client(self):
        """Lazily initialize the OpenAI client pointed at Ollama."""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OllamaProvider. "
                "Install it: pip install openai"
            )
        self._client = openai.OpenAI(
            api_key="ollama",
            base_url=self._base_url,
        )
        return self._client
