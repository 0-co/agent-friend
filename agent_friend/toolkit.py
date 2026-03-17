"""toolkit.py — Batch tool collection with multi-framework export."""

import json
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

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    _FORMATS = ("openai", "anthropic", "google", "mcp", "json_schema")

    def token_estimate(self, format: str = "openai") -> int:
        """Estimate total token count for all tools in the given format.

        Uses a simple heuristic: ``len(json_string) / 4``.  This is a rough
        approximation — intentionally named ``token_estimate`` to be honest
        about that.

        Parameters
        ----------
        format:
            One of ``"openai"``, ``"anthropic"``, ``"google"``, ``"mcp"``,
            or ``"json_schema"``.

        Returns
        -------
        Estimated token count as an integer.
        """
        exporters = {
            "openai": self.to_openai,
            "anthropic": self.to_anthropic,
            "google": self.to_google,
            "mcp": self.to_mcp,
            "json_schema": self.to_json_schema,
        }
        if format not in exporters:
            raise ValueError(
                f"Unknown format {format!r}. Choose from: {', '.join(exporters)}"
            )
        schema = exporters[format]()
        return int(len(json.dumps(schema, separators=(",", ":"))) / 4)

    def token_report(self) -> Dict[str, Any]:
        """Return token estimates for all 5 formats with comparison metadata.

        Returns
        -------
        Dict with keys:

        - ``estimates``: dict mapping format name to estimated token count
        - ``most_expensive``: format name with the highest estimate
        - ``least_expensive``: format name with the lowest estimate
        - ``tool_count``: number of tools in the toolkit
        """
        estimates = {fmt: self.token_estimate(format=fmt) for fmt in self._FORMATS}
        most = max(estimates, key=estimates.__getitem__)
        least = min(estimates, key=estimates.__getitem__)
        return {
            "estimates": estimates,
            "most_expensive": most,
            "least_expensive": least,
            "tool_count": len(self),
        }

    def __len__(self) -> int:
        return sum(len(t.definitions()) for t in self._tools)

    def __repr__(self) -> str:
        return f"Toolkit({len(self)} tools)"
