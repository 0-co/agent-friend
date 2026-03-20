"""bitnet.py — BitNet provider for agent-friend.

Connects to Microsoft BitNet's llama-server via its OpenAI-compatible API.
BitNet provides 1-bit LLM inference that runs efficiently on CPU — no GPU required.
No API key needed — just a running BitNet llama-server.

Default endpoint: http://localhost:8080/v1
Models: bitnet-b1.58-2B-4T (2B params, 1.58-bit quantization)

    # Start BitNet server (see https://github.com/microsoft/BitNet)
    friend = Friend(model="bitnet-b1.58-2B-4T")  # auto-detects BitNet provider
"""

import os
from typing import Optional

from .openai import OpenAIProvider


class BitNetProvider(OpenAIProvider):
    """LLM provider for local BitNet (Microsoft) 1-bit inference server.

    Uses BitNet's llama-server OpenAI-compatible API. No API key needed.
    CPU-only inference — no GPU required.

    Parameters
    ----------
    api_key:
        Ignored (BitNet doesn't need auth). Accepts it for interface compat.
    base_url:
        BitNet API endpoint. Defaults to ``BITNET_HOST`` env var or
        ``http://localhost:8080/v1``.
    """

    DEFAULT_MODEL = "bitnet-b1.58-2B-4T"

    def __init__(
        self, api_key: Optional[str] = None, base_url: Optional[str] = None
    ) -> None:
        super().__init__(api_key="bitnet")  # Dummy key — BitNet ignores it
        host = base_url or os.environ.get("BITNET_HOST", "http://localhost:8080")
        # Ensure /v1 suffix for OpenAI compat endpoint
        self._base_url = host.rstrip("/") + "/v1" if not host.endswith("/v1") else host

    def _get_client(self):
        """Lazily initialize the OpenAI client pointed at BitNet."""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for BitNetProvider. "
                "Install it: pip install openai"
            )
        self._client = openai.OpenAI(
            api_key="bitnet",
            base_url=self._base_url,
        )
        return self._client
