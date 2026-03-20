"""openrouter.py — OpenRouter provider for agent-friend.

OpenRouter exposes an OpenAI-compatible API at https://openrouter.ai/api/v1
with access to hundreds of models, including several free-tier options:
  - google/gemini-2.0-flash-exp:free
  - meta-llama/llama-3.3-70b-instruct:free
  - mistralai/mistral-7b-instruct:free

API key: https://openrouter.ai/ (free account, no credit card required)
Set via: export OPENROUTER_API_KEY=sk-or-...
"""

import os
from typing import Optional

from .openai import OpenAIProvider


class OpenRouterProvider(OpenAIProvider):
    """LLM provider for OpenRouter (OpenAI-compatible API).

    Supports 200+ models including free-tier options. Requires the ``openai``
    package to be installed (used as the HTTP client).

    Parameters
    ----------
    api_key:
        OpenRouter API key. Falls back to ``OPENROUTER_API_KEY`` env var.
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "google/gemini-2.0-flash-exp:free"

    # Popular free models on OpenRouter
    FREE_MODELS = [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        # Fall back to OPENROUTER_API_KEY if not provided
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(api_key=resolved_key)
        self._base_url = self.BASE_URL

    def _get_client(self):
        """Lazily initialize the OpenAI client pointed at OpenRouter."""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenRouterProvider. "
                "Install it: pip install openai"
            )
        self._client = openai.OpenAI(
            api_key=self.api_key or "no-key",  # OpenRouter rejects empty strings
            base_url=self.BASE_URL,
        )
        return self._client
