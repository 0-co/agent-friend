"""json_tool.py — JSONTool for agent-friend (stdlib only).

Agents can parse, query, transform, and validate JSON data without
resorting to CodeTool for every JSON manipulation.

Usage::

    tool = JSONTool()
    data = '{"user": {"name": "Alice", "age": 30}, "tags": ["ai", "python"]}'
    tool.json_get(data, "user.name")      # "Alice"
    tool.json_get(data, "tags[0]")        # "ai"
    tool.json_keys(data)                   # ["user", "tags"]
    tool.json_set(data, "user.email", "a@b.com")
    tool.json_filter('[{"role":"admin"},{"role":"user"}]', "role", "admin")
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


def _parse(data: str) -> Any:
    """Parse a JSON string, returning the Python object."""
    if isinstance(data, (dict, list)):
        return data
    return json.loads(data)


def _get_by_path(obj: Any, path: str) -> Any:
    """Traverse *obj* using dot-notation path with optional array indexing.

    Examples
    --------
    ``"user.name"`` → ``obj["user"]["name"]``
    ``"users[0].email"`` → ``obj["users"][0]["email"]``
    ``"data[*].id"`` → list of all ``id`` values in ``data``
    """
    parts = _split_path(path)
    return _traverse(obj, parts)


def _split_path(path: str) -> List[Union[str, int, str]]:
    """Split dot-notation path into a list of keys / indexes."""
    parts: List[Any] = []
    for segment in path.split("."):
        # Handle pure index segment: [0] or [*]
        pure_index = re.match(r"^\[(\d+|\*)\]$", segment)
        if pure_index:
            idx = pure_index.group(1)
            parts.append("*" if idx == "*" else int(idx))
            continue
        # Handle key with optional index suffix: name[0] or name[*]
        match = re.match(r"^([^\[]+)(?:\[(\d+|\*)\])?$", segment)
        if match:
            key, idx = match.group(1), match.group(2)
            if key:
                parts.append(key)
            if idx is not None:
                parts.append("*" if idx == "*" else int(idx))
        else:
            parts.append(segment)
    return parts


def _traverse(obj: Any, parts: List[Any]) -> Any:
    if not parts:
        return obj
    part = parts[0]
    rest = parts[1:]

    if part == "*":
        if not isinstance(obj, list):
            raise ValueError(f"Expected list for [*] wildcard, got {type(obj).__name__}")
        return [_traverse(item, rest) for item in obj]

    if isinstance(part, int):
        return _traverse(obj[part], rest)

    if isinstance(obj, dict):
        return _traverse(obj[part], rest)

    raise TypeError(f"Cannot traverse {type(obj).__name__} with key {part!r}")


def _set_by_path(obj: Any, path: str, value: Any) -> Any:
    """Return a deep copy of *obj* with *value* set at *path*."""
    import copy
    result = copy.deepcopy(obj)
    parts = _split_path(path)
    _set_nested(result, parts, value)
    return result


def _set_nested(obj: Any, parts: List[Any], value: Any) -> None:
    if len(parts) == 1:
        key = parts[0]
        if isinstance(obj, dict):
            obj[key] = value
        elif isinstance(obj, list) and isinstance(key, int):
            obj[key] = value
        return
    part = parts[0]
    _set_nested(obj[part], parts[1:], value)


class JSONTool(BaseTool):
    """Parse, query, transform, and validate JSON data.

    Useful when working with JSON responses from ``HTTPTool`` or
    any other source — extract fields, filter arrays, set values,
    and pretty-print output.

    All methods accept either a JSON string or a Python dict/list.
    """

    # ── public Python API ─────────────────────────────────────────────────

    def json_get(self, data: str, path: str) -> str:
        """Return the value at *path* in *data* as a JSON string.

        Parameters
        ----------
        data: JSON string or Python object.
        path: Dot-notation path, e.g. ``"user.name"``, ``"items[0].price"``,
              ``"users[*].email"`` (wildcard returns list).
        """
        obj = _parse(data)
        result = _get_by_path(obj, path)
        return json.dumps(result)

    def json_set(self, data: str, path: str, value: str) -> str:
        """Return a new JSON string with *value* set at *path*.

        Parameters
        ----------
        data:  Original JSON string.
        path:  Dot-notation path.
        value: JSON-encoded value to set (string, number, bool, object, or null).
        """
        obj = _parse(data)
        val = json.loads(value) if isinstance(value, str) else value
        new_obj = _set_by_path(obj, path, val)
        return json.dumps(new_obj)

    def json_keys(self, data: str) -> str:
        """Return the top-level keys of a JSON object as a JSON array."""
        obj = _parse(data)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
        return json.dumps(list(obj.keys()))

    def json_filter(self, data: str, key: str, value: str) -> str:
        """Filter a JSON array, keeping items where *key* equals *value*.

        Parameters
        ----------
        data:  JSON array string.
        key:   Key to filter on (string).
        value: JSON-encoded value to match.
        """
        obj = _parse(data)
        if not isinstance(obj, list):
            raise ValueError(f"Expected JSON array, got {type(obj).__name__}")
        target = json.loads(value) if isinstance(value, str) else value
        filtered = [item for item in obj if isinstance(item, dict) and item.get(key) == target]
        return json.dumps(filtered)

    def json_format(self, data: str, indent: int = 2) -> str:
        """Return a pretty-printed version of the JSON string."""
        obj = _parse(data)
        return json.dumps(obj, indent=indent)

    def json_merge(self, base: str, patch: str) -> str:
        """Merge two JSON objects. Keys in *patch* override *base*."""
        obj_base = _parse(base)
        obj_patch = _parse(patch)
        if not isinstance(obj_base, dict) or not isinstance(obj_patch, dict):
            raise ValueError("Both arguments must be JSON objects")
        result = {**obj_base, **obj_patch}
        return json.dumps(result)

    # ── BaseTool protocol ─────────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call from the LLM."""
        try:
            if tool_name == "json_get":
                return self.json_get(arguments["data"], arguments["path"])
            if tool_name == "json_set":
                return self.json_set(arguments["data"], arguments["path"], arguments["value"])
            if tool_name == "json_keys":
                return self.json_keys(arguments["data"])
            if tool_name == "json_filter":
                return self.json_filter(arguments["data"], arguments["key"], arguments["value"])
            if tool_name == "json_format":
                return self.json_format(arguments["data"], arguments.get("indent", 2))
            if tool_name == "json_merge":
                return self.json_merge(arguments["base"], arguments["patch"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as e:
            return f"Error: {e}"
        return f"Unknown json tool: {tool_name}"

    @property
    def name(self) -> str:
        return "json"

    @property
    def description(self) -> str:
        return (
            "Parse, query, transform, and validate JSON data. "
            "Extract fields by path, filter arrays, set values, merge objects, pretty-print."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "json_get",
                "description": (
                    "Extract a value from JSON using dot-notation path. "
                    "Examples: 'user.name', 'items[0].price', 'users[*].email' (wildcard → list). "
                    "Returns the value as a JSON string."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "JSON string to query.",
                        },
                        "path": {
                            "type": "string",
                            "description": "Dot-notation path, e.g. 'user.name' or 'items[0].id'.",
                        },
                    },
                    "required": ["data", "path"],
                },
            },
            {
                "name": "json_set",
                "description": (
                    "Set a value at a path in a JSON object, returning the modified JSON string. "
                    "Path uses dot notation. Value must be a JSON-encoded string."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Original JSON string."},
                        "path": {"type": "string", "description": "Dot-notation path."},
                        "value": {
                            "type": "string",
                            "description": "JSON-encoded value to set (e.g. '\"Alice\"', '42', 'true').",
                        },
                    },
                    "required": ["data", "path", "value"],
                },
            },
            {
                "name": "json_keys",
                "description": "Return the top-level keys of a JSON object as a JSON array.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON object string."},
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "json_filter",
                "description": (
                    "Filter a JSON array, keeping items where the given key equals the given value. "
                    "Returns a JSON array of matching items."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON array string."},
                        "key": {"type": "string", "description": "Key to filter on."},
                        "value": {
                            "type": "string",
                            "description": "JSON-encoded value to match (e.g. '\"admin\"', '1').",
                        },
                    },
                    "required": ["data", "key", "value"],
                },
            },
            {
                "name": "json_format",
                "description": "Pretty-print a JSON string with configurable indentation.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON string to format."},
                        "indent": {
                            "type": "integer",
                            "description": "Number of spaces per indent level. Default 2.",
                            "default": 2,
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "json_merge",
                "description": "Merge two JSON objects. Keys in the patch object override the base.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "base": {"type": "string", "description": "Base JSON object."},
                        "patch": {"type": "string", "description": "Patch JSON object (overrides base)."},
                    },
                    "required": ["base", "patch"],
                },
            },
        ]
