"""toolkit.py — Batch tool collection with multi-framework export."""

from typing import Any, Callable, Dict, List, Union

from .tools.base import BaseTool
from .tools.function_tool import FunctionTool


class Toolkit:
    """Collection of tools with batch export to any AI framework.

    Usage::

        from agent_friend import Toolkit, tool

        @tool
        def func_a(x: str) -> str:
            \"\"\"Do something with x.\"\"\"
            return x

        @tool
        def func_b(y: int) -> int:
            \"\"\"Do something with y.\"\"\"
            return y

        kit = Toolkit([func_a, func_b])
        kit.to_openai()     # List of all tools in OpenAI format
        kit.to_anthropic()  # List of all tools in Anthropic format
        kit.to_mcp()        # List of all tools in MCP format
    """

    def __init__(self, tools: List[Union[BaseTool, Callable]]) -> None:
        self._tools: List[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                self._tools.append(t)
            elif hasattr(t, "_agent_tool"):
                self._tools.append(t._agent_tool)
            elif callable(t):
                # Wrap plain callable as FunctionTool
                ft = FunctionTool(
                    t,
                    t.__name__,
                    (t.__doc__ or "").strip() or t.__name__,
                )
                self._tools.append(ft)
            else:
                raise TypeError(
                    f"Expected BaseTool, @tool-decorated function, or callable, got {type(t)}"
                )

    def to_anthropic(self) -> List[Dict[str, Any]]:
        """Export all tools in Anthropic Claude format."""
        result: List[Dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.to_anthropic())
        return result

    def to_openai(self) -> List[Dict[str, Any]]:
        """Export all tools in OpenAI function-calling format."""
        result: List[Dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.to_openai())
        return result

    def to_google(self) -> List[Dict[str, Any]]:
        """Export all tools in Google Gemini format."""
        result: List[Dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.to_google())
        return result

    def to_mcp(self) -> List[Dict[str, Any]]:
        """Export all tools in MCP (Model Context Protocol) format."""
        result: List[Dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.to_mcp())
        return result

    def to_json_schema(self) -> List[Dict[str, Any]]:
        """Export raw JSON Schema for all tools."""
        result: List[Dict[str, Any]] = []
        for t in self._tools:
            result.extend(t.to_json_schema())
        return result

    def __len__(self) -> int:
        return sum(len(t.definitions()) for t in self._tools)

    def __repr__(self) -> str:
        return f"Toolkit({len(self)} tools)"
