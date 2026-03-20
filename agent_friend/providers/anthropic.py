"""anthropic.py — Anthropic Claude provider for agent-friend."""

from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    """LLM provider for Anthropic Claude models.

    Requires the `anthropic` package to be installed.
    Uses tool_use / tool_result message format for function calling.

    Parameters
    ----------
    api_key:  Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the Anthropic client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install it: pip install anthropic"
            )
        if self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)
        else:
            # SDK picks up ANTHROPIC_API_KEY from environment automatically
            self._client = anthropic.Anthropic()
        return self._client

    def complete(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        """Call the Anthropic messages API and return a normalized response."""
        client = self._get_client()
        model = model or self.DEFAULT_MODEL

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)
        return self._normalize(response, model)

    def _normalize(self, response: Any, model: str) -> ProviderResponse:
        """Convert Anthropic response to ProviderResponse."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": block.input,
                    }
                )

        return ProviderResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=str(response.stop_reason),
            model=response.model,
        )

    def build_tool_result_message(
        self,
        tool_use_response: ProviderResponse,
        tool_results: List[Dict[str, str]],
        original_content: Any,
    ) -> Dict[str, Any]:
        """Build the tool_result message to append to conversation history.

        Parameters
        ----------
        tool_use_response:  The provider response that contained tool calls.
        tool_results:       List of {tool_use_id, content} dicts.
        original_content:   The raw content blocks from the assistant message.
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": result["tool_use_id"],
                    "content": result["content"],
                }
                for result in tool_results
            ],
        }
