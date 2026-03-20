#!/usr/bin/env python3
"""MCP server exposing agent-friend's built-in tools via the Model Context Protocol.

Usage:
    agent-friend-mcp          # recommended (installed entry point)
    python mcp_server.py      # direct run from repo clone

Claude Desktop config (~/.claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "agent-friend": {
          "command": "agent-friend-mcp"
        }
      }
    }

All agent-friend tools that can be instantiated without arguments are
registered automatically. Each tool method becomes a separate MCP tool
with the naming convention: {tool_name}_{method_name} (e.g. "datetime_now",
"crypto_hash_data", "json_get").

Tool count: ~310 methods from ~50 tool classes.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure agent-friend is importable when running directly from repo
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from mcp.server.fastmcp import FastMCP

from agent_friend.tools import (
    AlertTool,
    AuditTool,
    BatchTool,
    BrowserTool,
    CacheTool,
    ChunkerTool,
    CodeTool,
    ConfigTool,
    CryptoTool,
    DatabaseTool,
    DateTimeTool,
    DiffTool,
    EnvTool,
    EventBusTool,
    FetchTool,
    FileTool,
    FormatTool,
    GitTool,
    GraphTool,
    HTMLTool,
    HTTPTool,
    JSONTool,
    LockTool,
    MapReduceTool,
    MemoryTool,
    MetricsTool,
    NotifyTool,
    ProcessTool,
    QueueTool,
    RateLimitTool,
    RSSFeedTool,
    RegexTool,
    RetryTool,
    SamplerTool,
    SchedulerTool,
    SearchIndexTool,
    SearchTool,
    StateMachineTool,
    StatsTool,
    TableTool,
    TemplateTool,
    TimerTool,
    TransformTool,
    ValidatorTool,
    VectorStoreTool,
    VoiceTool,
    WebhookTool,
    WorkflowTool,
    XMLTool,
)

# ---------------------------------------------------------------------------
# All tool classes to register (those that take no constructor args)
# ---------------------------------------------------------------------------

TOOL_CLASSES = [
    AlertTool,
    AuditTool,
    BatchTool,
    BrowserTool,
    CacheTool,
    ChunkerTool,
    CodeTool,
    ConfigTool,
    CryptoTool,
    DatabaseTool,
    DateTimeTool,
    DiffTool,
    EnvTool,
    EventBusTool,
    FetchTool,
    FileTool,
    FormatTool,
    GitTool,
    GraphTool,
    HTMLTool,
    HTTPTool,
    JSONTool,
    LockTool,
    MapReduceTool,
    MemoryTool,
    MetricsTool,
    NotifyTool,
    ProcessTool,
    QueueTool,
    RateLimitTool,
    RSSFeedTool,
    RegexTool,
    RetryTool,
    SamplerTool,
    SchedulerTool,
    SearchIndexTool,
    SearchTool,
    StateMachineTool,
    StatsTool,
    TableTool,
    TemplateTool,
    TimerTool,
    TransformTool,
    ValidatorTool,
    VectorStoreTool,
    VoiceTool,
    WebhookTool,
    WorkflowTool,
    XMLTool,
]

# ---------------------------------------------------------------------------
# JSON Schema type -> Python type string mapping
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

# Python keywords that cannot be used as parameter names
_PYTHON_KEYWORDS = {
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
    "or", "pass", "raise", "return", "try", "while", "with", "yield",
}


def _resolve_type(prop: Dict[str, Any]) -> str:
    """Resolve a JSON Schema property to a Python type string."""
    t = prop.get("type")
    if t is None:
        # No type specified -> accept anything as str
        return "str"
    if isinstance(t, list):
        # Union type like ["string", "null"] -> use first non-null
        for sub in t:
            if sub != "null":
                return _TYPE_MAP.get(sub, "str")
        return "str"
    return _TYPE_MAP.get(t, "str")


def _safe_param_name(name: str) -> str:
    """Ensure a parameter name is a valid Python identifier and not a keyword."""
    if name in _PYTHON_KEYWORDS:
        return f"{name}_"
    if not name.isidentifier():
        return name.replace("-", "_").replace(".", "_")
    return name


def _make_mcp_handler(tool_instance, method_name: str, input_schema: Dict[str, Any]):
    """Dynamically create a typed Python function that dispatches to
    tool_instance.execute(method_name, arguments).

    FastMCP inspects function signatures to build MCP input schemas, so
    we create functions with proper type annotations matching the tool's
    JSON Schema definition.
    """
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    if not props:
        # No parameters -- simple handler
        def handler() -> str:
            result = tool_instance.execute(method_name, {})
            return result if isinstance(result, str) else json.dumps(result)
        return handler

    # Build parameter list for the function signature
    # Required params first, then optional ones
    req_params = []
    opt_params = []
    # Track original names if we had to rename
    renames = {}

    for pname, prop in props.items():
        py_type = _resolve_type(prop)
        safe_name = _safe_param_name(pname)
        if safe_name != pname:
            renames[safe_name] = pname

        if pname in required:
            req_params.append(f"{safe_name}: {py_type}")
        else:
            opt_params.append(f"{safe_name}: Optional[{py_type}] = None")

    param_str = ", ".join(req_params + opt_params)

    # Build the function body
    # Collect all params into a dict, stripping None values
    func_code = f"""
def _handler({param_str}) -> str:
    _args = {{k: v for k, v in locals().items() if v is not None}}
    # Restore original param names if renamed
    for _safe, _orig in _renames.items():
        if _safe in _args:
            _args[_orig] = _args.pop(_safe)
    return _dispatch(_args)
"""

    namespace = {
        "_dispatch": lambda args: _dispatch_call(tool_instance, method_name, args),
        "_renames": renames,
        "Optional": Optional,
    }
    exec(func_code, namespace)
    return namespace["_handler"]


def _dispatch_call(tool_instance, method_name: str, args: Dict[str, Any]) -> str:
    """Call tool_instance.execute and ensure the result is a string."""
    result = tool_instance.execute(method_name, args)
    if not isinstance(result, str):
        return json.dumps(result)
    return result


# ---------------------------------------------------------------------------
# Initialize FastMCP server and register tools
# ---------------------------------------------------------------------------

server = FastMCP(
    "agent-friend",
    instructions=(
        "agent-friend tools: a collection of zero-dependency Python utilities "
        "covering date/time, crypto, validation, JSON manipulation, regex, "
        "formatting, diffing, text processing, data structures, and more. "
        "Each tool is prefixed with its category (e.g. datetime_, crypto_, json_)."
    ),
)

registered_count = 0
skipped_classes = []
seen_names: set = set()

for cls in TOOL_CLASSES:
    try:
        tool_inst = cls()
    except TypeError as e:
        skipped_classes.append((cls.__name__, str(e)))
        continue

    tool_prefix = tool_inst.name

    for defn in tool_inst.definitions():
        method_name = defn["name"]
        description = defn.get("description", "")
        input_schema = defn.get("input_schema", {"type": "object", "properties": {}})

        # Unique MCP tool name: {category}_{method}
        mcp_name = f"{tool_prefix}_{method_name}"
        if mcp_name in seen_names:
            mcp_name = f"{mcp_name}_2"
        seen_names.add(mcp_name)

        # Build a typed handler function
        handler = _make_mcp_handler(tool_inst, method_name, input_schema)
        handler.__name__ = mcp_name
        handler.__qualname__ = mcp_name
        handler.__doc__ = f"[{tool_prefix}] {description}"

        # Register with FastMCP
        server.add_tool(
            handler,
            name=mcp_name,
            description=f"[{tool_prefix}] {description}",
        )
        registered_count += 1

print(
    f"agent-friend MCP server: registered {registered_count} tools "
    f"from {len(TOOL_CLASSES) - len(skipped_classes)} tool classes",
    file=sys.stderr,
)
if skipped_classes:
    for name, reason in skipped_classes:
        print(f"  skipped {name}: {reason}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server.run(transport="stdio")
