"""openai.py — OpenAI provider for agent-friend."""

import json
from typing import Any, Dict, List, Optional

from .base import BaseProvider, ProviderResponse


class OpenAIProvider(BaseProvider):
    """LLM provider for OpenAI models.

    Requires the `openai` package to be installed.
    Maps Anthropic-format tool definitions to OpenAI function calling format.

    Parameters
    ----------
    api_key:  OpenAI API key. Falls back to OPENAI_API_KEY env var.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install it: pip install openai"
            )
        if self.api_key:
            self._client = openai.OpenAI(api_key=self.api_key)
        else:
            # SDK picks up OPENAI_API_KEY from environment automatically
            self._client = openai.OpenAI()
        return self._client

    def complete(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> ProviderResponse:
        """Call the OpenAI chat completions API and return a normalized response."""
        client = self._get_client()
        model = model or self.DEFAULT_MODEL

        # OpenAI uses a "system" message at the top of the conversation
        openai_messages = [{"role": "system", "content": system}] + messages

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": openai_messages,
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**kwargs)
        return self._normalize(response, model)

    def _convert_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic tool format to OpenAI function format."""
        openai_tools = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return openai_tools

    def _normalize(self, response: Any, model: str) -> ProviderResponse:
        """Convert OpenAI response to ProviderResponse."""
        choice = response.choices[0]
        message = choice.message

        text = message.content or ""
        tool_calls = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {"raw": arguments}
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    }
                )

        usage = response.usage
        return ProviderResponse(
            text=text,
            tool_calls=tool_calls,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=str(choice.finish_reason),
            model=response.model,
        )

    def build_tool_result_message(
        self,
        tool_use_response: ProviderResponse,
        tool_results: List[Dict[str, str]],
        original_content: Any,
    ) -> Dict[str, Any]:
        """Build tool result messages for OpenAI format.

        OpenAI uses separate messages per tool result, but we return a single
        structure here; the caller handles appending them.
        """
        # For OpenAI, tool results are separate "tool" role messages
        # We encode them as a list under a special key for the caller to expand
        return {
            "role": "__tool_results__",
            "tool_results": [
                {
                    "role": "tool",
                    "tool_call_id": result["tool_use_id"],
                    "content": result["content"],
                }
                for result in tool_results
            ],
        }
