"""base.py — BaseProvider abstract class for agent-friend LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseProvider(ABC):
    """Abstract base for LLM API providers.

    Subclasses implement complete() which takes messages + tool definitions
    and returns a ProviderResponse.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, Any]],
        system: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> "ProviderResponse":
        """Send messages to the LLM and return a structured response.

        Parameters
        ----------
        messages:  Conversation history in provider-native format.
        system:    System prompt string.
        tools:     List of tool definitions (Anthropic format).
        model:     Model identifier to use.
        """


class ProviderResponse:
    """Normalized response from any LLM provider.

    Attributes
    ----------
    text:          The assistant's text response (may be empty if only tool calls).
    tool_calls:    List of tool call dicts: [{id, name, arguments}, ...].
    input_tokens:  Tokens in the prompt/context.
    output_tokens: Tokens in the completion.
    stop_reason:   Why generation stopped (e.g. "end_turn", "tool_use").
    model:         Model that produced this response.
    """

    def __init__(
        self,
        text: str,
        tool_calls: List[Dict[str, Any]],
        input_tokens: int,
        output_tokens: int,
        stop_reason: str,
        model: str,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.stop_reason = stop_reason
        self.model = model

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)
