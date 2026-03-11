"""friend.py — Friend class, the main entry point for agent-friend."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Union

from .config import FriendConfig, load_from_dict, load_from_yaml
from .providers.base import BaseProvider, ProviderResponse
from .tools.base import BaseTool


# ---------------------------------------------------------------------------
# Pricing (per 1M tokens, input/output)
# ---------------------------------------------------------------------------

_TOKEN_COSTS: Dict[str, tuple] = {
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-6": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # OpenRouter free-tier models ($0 but rate-limited)
    "google/gemini-2.0-flash-exp:free": (0.0, 0.0),
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "mistralai/mistral-7b-instruct:free": (0.0, 0.0),
    "qwen/qwen-2.5-72b-instruct:free": (0.0, 0.0),
}

_TOOL_NAME_MAP = {
    "memory": "agent_friend.tools.memory:MemoryTool",
    "code": "agent_friend.tools.code:CodeTool",
    "search": "agent_friend.tools.search:SearchTool",
    "browser": "agent_friend.tools.browser:BrowserTool",
    "email": "agent_friend.tools.email:EmailTool",
}


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """Result of a single Friend.chat() call.

    Attributes
    ----------
    text:          The final assistant text response.
    tool_calls:    All tool calls made during this exchange.
    input_tokens:  Total input tokens used (across all LLM calls in this turn).
    output_tokens: Total output tokens used.
    cost_usd:      Estimated cost in USD.
    model:         Model that produced the response.
    """

    text: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


# ---------------------------------------------------------------------------
# BudgetExceeded
# ---------------------------------------------------------------------------

class BudgetExceeded(Exception):
    """Raised when estimated cost would exceed the configured budget_usd limit."""

    def __init__(self, spent: float, limit: float) -> None:
        self.spent = spent
        self.limit = limit
        super().__init__(
            f"Budget exceeded: spent ${spent:.4f} of ${limit:.2f} limit."
        )


# ---------------------------------------------------------------------------
# Friend
# ---------------------------------------------------------------------------

class Friend:
    """A composable personal AI agent.

    Minimal usage:

        friend = Friend(seed="You are a helpful assistant.", api_key="sk-...")
        response = friend.chat("What is 2+2?")
        print(response.text)

    With tools:

        friend = Friend(
            seed="You are a helpful assistant.",
            tools=["search", "code", "memory"],
            model="claude-sonnet-4-6",
            budget_usd=1.0,
        )

    Parameters are passed through to FriendConfig.
    """

    def __init__(
        self,
        seed: str = "You are a helpful personal AI assistant.",
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        provider: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        memory_path: str = "~/.agent_friend/memory.db",
        budget_usd: Optional[float] = None,
        max_context_messages: int = 20,
    ) -> None:
        self._config = FriendConfig(
            seed=seed,
            model=model,
            api_key=api_key,
            provider=provider,
            tools=tools or [],
            memory_path=memory_path,
            budget_usd=budget_usd,
            max_context_messages=max_context_messages,
        )
        self._provider: Optional[BaseProvider] = None
        self._tools: List[BaseTool] = []
        self._conversation: List[Dict[str, Any]] = []
        self._total_cost_usd: float = 0.0

        self._initialize_tools(self._config.tools)

    # ------------------------------------------------------------------
    # Class methods for alternative construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, data: Dict[str, Any]) -> "Friend":
        """Create a Friend from a configuration dictionary."""
        config = load_from_dict(data)
        return cls._from_friend_config(config)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Friend":
        """Create a Friend from a YAML configuration file."""
        config = load_from_yaml(path)
        return cls._from_friend_config(config)

    @classmethod
    def _from_friend_config(cls, config: FriendConfig) -> "Friend":
        instance = cls.__new__(cls)
        instance._config = config
        instance._provider = None
        instance._tools = []
        instance._conversation = []
        instance._total_cost_usd = 0.0
        instance._initialize_tools(config.tools)
        return instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str) -> ChatResponse:
        """Send a message and return a ChatResponse.

        Runs the full tool call loop: if the LLM requests tool calls,
        executes them and continues until the LLM returns a final response.
        """
        self._conversation.append({"role": "user", "content": message})
        return self._run_loop()

    def stream(self, message: str) -> Iterator[str]:
        """Stream response text chunks.

        Yields text chunks as they arrive. Tool calls are executed silently
        and their results fed back to the LLM; only final text is yielded.

        Note: This is a simple implementation — it buffers tool call turns
        and only streams the final text response.
        """
        response = self.chat(message)
        # Yield the response text word-by-word to simulate streaming
        words = response.text.split(" ")
        for index, word in enumerate(words):
            if index < len(words) - 1:
                yield word + " "
            else:
                yield word

    def reset(self) -> None:
        """Clear conversation history while keeping memory and tool state."""
        self._conversation = []

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost spent across all chats in this session."""
        return self._total_cost_usd

    # ------------------------------------------------------------------
    # Tool call loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> ChatResponse:
        """Execute the LLM + tool call loop until a final response."""
        provider = self._get_provider()
        all_tool_calls: List[Dict[str, Any]] = []
        total_input = 0
        total_output = 0
        final_text = ""
        final_model = self._config.model

        messages = list(self._conversation)

        for _ in range(20):  # Safety cap on tool call iterations
            tool_defs = self._build_tool_definitions()
            provider_name = self._config.resolve_provider()

            response = provider.complete(
                messages=messages,
                system=self._config.seed,
                tools=tool_defs or None,
                model=self._config.model,
            )

            total_input += response.input_tokens
            total_output += response.output_tokens
            final_model = response.model

            turn_cost = _calculate_cost(
                response.input_tokens, response.output_tokens, response.model
            )
            self._total_cost_usd += turn_cost

            if self._config.budget_usd is not None:
                if self._total_cost_usd > self._config.budget_usd:
                    raise BudgetExceeded(self._total_cost_usd, self._config.budget_usd)

            if not response.has_tool_calls:
                final_text = response.text
                # Append assistant message to persistent conversation
                self._conversation.append(
                    {"role": "assistant", "content": response.text}
                )
                break

            # Append assistant tool-use message
            all_tool_calls.extend(response.tool_calls)
            assistant_msg = self._build_assistant_tool_message(response, provider_name)
            messages.append(assistant_msg)

            # Execute tools and build result message
            tool_results = self._execute_tool_calls(response.tool_calls)
            result_msg = self._build_tool_result_message(
                provider_name, response, tool_results
            )

            if result_msg.get("role") == "__tool_results__":
                # OpenAI: multiple tool messages
                for msg in result_msg["tool_results"]:
                    messages.append(msg)
            else:
                messages.append(result_msg)
        else:
            final_text = "(Tool call limit reached without final response.)"

        total_cost = _calculate_cost(0, 0, final_model)  # already accumulated above
        return ChatResponse(
            text=final_text,
            tool_calls=all_tool_calls,
            input_tokens=total_input,
            output_tokens=total_output,
            cost_usd=_calculate_cost(total_input, total_output, final_model),
            model=final_model,
        )

    def _build_assistant_tool_message(
        self, response: ProviderResponse, provider_name: str
    ) -> Dict[str, Any]:
        """Build the assistant message containing tool_use blocks."""
        if provider_name in ("openai", "openrouter"):
            # OpenAI/OpenRouter: assistant message with tool_calls array
            return {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": _json_dumps(tc["arguments"]),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
        # Anthropic: content array with text + tool_use blocks
        content = []
        if response.text:
            content.append({"type": "text", "text": response.text})
        for tool_call in response.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "input": tool_call["arguments"],
                }
            )
        return {"role": "assistant", "content": content}

    def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Run each tool call and return results."""
        results = []
        tool_map = {
            tool_def_name: tool_instance
            for tool_instance in self._tools
            for tool_def_name in [td["name"] for td in tool_instance.definitions()]
        }

        for tool_call in tool_calls:
            name = tool_call["name"]
            arguments = tool_call["arguments"]
            tool_instance = tool_map.get(name)

            if tool_instance is None:
                content = f"Tool not found: {name}"
            else:
                try:
                    content = tool_instance.execute(name, arguments)
                except Exception as error:
                    content = f"Tool error ({name}): {error}"

            results.append({"tool_use_id": tool_call["id"], "content": content})

        return results

    def _build_tool_result_message(
        self,
        provider_name: str,
        response: ProviderResponse,
        tool_results: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Build the tool result message in provider-native format."""
        if provider_name in ("openai", "openrouter"):
            from .providers.openai import OpenAIProvider
            provider = OpenAIProvider()
            return provider.build_tool_result_message(response, tool_results, None)

        from .providers.anthropic import AnthropicProvider
        provider_obj = AnthropicProvider()
        return provider_obj.build_tool_result_message(response, tool_results, None)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _get_provider(self) -> BaseProvider:
        """Lazily create the provider based on config."""
        if self._provider is not None:
            return self._provider

        provider_name = self._config.resolve_provider()
        api_key = self._config.resolve_api_key()

        if provider_name == "openrouter":
            from .providers.openrouter import OpenRouterProvider
            self._provider = OpenRouterProvider(api_key=api_key)
        elif provider_name == "openai":
            from .providers.openai import OpenAIProvider
            self._provider = OpenAIProvider(api_key=api_key)
        else:
            from .providers.anthropic import AnthropicProvider
            self._provider = AnthropicProvider(api_key=api_key)

        return self._provider

    def _initialize_tools(self, tool_specs: List[Any]) -> None:
        """Resolve string tool names to BaseTool instances."""
        from .tools.memory import MemoryTool
        from .tools.code import CodeTool
        from .tools.search import SearchTool
        from .tools.browser import BrowserTool
        from .tools.email import EmailTool

        name_to_class = {
            "memory": MemoryTool,
            "code": CodeTool,
            "search": SearchTool,
            "browser": BrowserTool,
            "email": EmailTool,
        }

        for spec in tool_specs:
            if isinstance(spec, str):
                tool_class = name_to_class.get(spec.lower())
                if tool_class is None:
                    raise ValueError(
                        f"Unknown tool name: '{spec}'. "
                        f"Valid names: {list(name_to_class.keys())}"
                    )
                # MemoryTool needs the configured memory_path
                if tool_class is MemoryTool:
                    self._tools.append(MemoryTool(db_path=self._config.memory_path))
                # EmailTool needs an inbox address (from env or default)
                elif tool_class is EmailTool:
                    inbox = os.environ.get("AGENTMAIL_INBOX", "")
                    self._tools.append(EmailTool(inbox=inbox))
                else:
                    self._tools.append(tool_class())
            elif isinstance(spec, BaseTool):
                self._tools.append(spec)
            else:
                raise TypeError(
                    f"Tool must be a string name or BaseTool instance, got: {type(spec)}"
                )

    def _build_tool_definitions(self) -> List[Dict[str, Any]]:
        """Collect all tool definitions from registered tools."""
        definitions = []
        for tool in self._tools:
            definitions.extend(tool.definitions())
        return definitions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Calculate estimated cost in USD for a given token usage and model."""
    # Normalize model name for lookup (strip dates/versions for fuzzy match)
    costs = _TOKEN_COSTS.get(model)
    if costs is None:
        # Fuzzy match: find the first key that appears in the model string
        for key, value in _TOKEN_COSTS.items():
            if key in model or model in key:
                costs = value
                break
    if costs is None:
        return 0.0  # Unknown model — don't fail, just return zero cost

    input_cost_per_million, output_cost_per_million = costs
    return (
        input_tokens * input_cost_per_million / 1_000_000
        + output_tokens * output_cost_per_million / 1_000_000
    )


def _json_dumps(obj: Any) -> str:
    """Serialize to JSON string, handling non-serializable types gracefully."""
    import json
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return json.dumps(str(obj))
