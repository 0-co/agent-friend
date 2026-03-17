"""base.py — BaseTool abstract class for agent-friend tools."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseTool(ABC):
    """Abstract base class for all agent-friend tools.

    Subclasses must implement:
      - name: str property
      - description: str property
      - definitions: returns list of tool definitions for the LLM
      - execute(tool_name, arguments): runs the tool and returns a string result
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this tool (e.g. "memory", "code")."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what this tool does."""

    @abstractmethod
    def definitions(self) -> List[Dict[str, Any]]:
        """Return list of tool definitions in Anthropic tool-use format.

        Each definition is a dict with:
          name: str
          description: str
          input_schema: dict (JSON Schema)
        """

    @abstractmethod
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return the result as a string.

        Parameters
        ----------
        tool_name:  The specific function name called (from definitions).
        arguments:  Dict of argument name -> value.

        Returns
        -------
        String result to return to the LLM.
        """

    # ------------------------------------------------------------------
    # Framework adapter methods
    # ------------------------------------------------------------------

    def to_anthropic(self) -> List[Dict[str, Any]]:
        """Export tool definitions in Anthropic Claude format.

        Returns the native format since ``definitions()`` already produces
        Anthropic-compatible dicts with ``name``, ``description``, and
        ``input_schema`` keys.
        """
        return self.definitions()

    def to_openai(self) -> List[Dict[str, Any]]:
        """Export tool definitions in OpenAI function-calling format.

        Each entry is wrapped in ``{"type": "function", "function": {...}}``.
        """
        result: List[Dict[str, Any]] = []
        for defn in self.definitions():
            result.append({
                "type": "function",
                "function": {
                    "name": defn["name"],
                    "description": defn.get("description", ""),
                    "parameters": defn.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return result

    def to_google(self) -> List[Dict[str, Any]]:
        """Export tool definitions in Google Gemini format.

        Types are uppercased (``STRING``, ``INTEGER``, etc.) to match the
        Gemini API convention.
        """
        _TYPE_MAP = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
        }
        result: List[Dict[str, Any]] = []
        for defn in self.definitions():
            schema = defn.get("input_schema", {"type": "object", "properties": {}})
            result.append({
                "name": defn["name"],
                "description": defn.get("description", ""),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        k: {
                            "type": _TYPE_MAP.get(
                                (v.get("type", "string") if isinstance(v.get("type"), str) else (v.get("type") or ["string"])[0]),
                                ((v.get("type", "string") if isinstance(v.get("type"), str) else (v.get("type") or ["string"])[0]).upper()),
                            ),
                            "description": v.get("description", ""),
                        }
                        for k, v in schema.get("properties", {}).items()
                    },
                    "required": schema.get("required", []),
                },
            })
        return result

    def to_mcp(self) -> List[Dict[str, Any]]:
        """Export tool definitions in MCP (Model Context Protocol) format.

        Uses ``inputSchema`` (camelCase) as required by the MCP spec.
        """
        result: List[Dict[str, Any]] = []
        for defn in self.definitions():
            result.append({
                "name": defn["name"],
                "description": defn.get("description", ""),
                "inputSchema": defn.get("input_schema", {"type": "object", "properties": {}}),
            })
        return result

    def to_json_schema(self) -> List[Dict[str, Any]]:
        """Export raw JSON Schema for each tool.

        Each schema dict includes ``title`` (tool name) and ``description``
        alongside the standard JSON Schema properties.
        """
        result: List[Dict[str, Any]] = []
        for defn in self.definitions():
            schema = dict(defn.get("input_schema", {"type": "object", "properties": {}}))
            schema["title"] = defn["name"]
            schema["description"] = defn.get("description", "")
            result.append(schema)
        return result

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    _FORMATS = ("openai", "anthropic", "google", "mcp", "json_schema")

    def token_estimate(self, format: str = "openai") -> int:
        """Estimate token count for this tool's schema in the given format.

        Uses a simple heuristic: ``len(json_string) / 4``.  This is a rough
        approximation (no external tokenizer required) and intentionally named
        ``token_estimate`` rather than ``token_count`` to be honest about that.

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
