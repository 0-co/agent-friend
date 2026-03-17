"""audit.py — Analyze tool definitions and report token cost across formats.

Reads tool definitions from JSON (any of 5 supported formats), converts them
to FunctionTool instances, and uses the existing token_estimate / token_report
machinery to produce a colored terminal report.
"""

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

from .tools.function_tool import FunctionTool
from .toolkit import Toolkit


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_FORMATS = ("openai", "anthropic", "mcp", "json_schema", "simple")


def detect_format(obj: Dict[str, Any]) -> str:
    """Detect which tool definition format a single JSON object uses.

    Returns one of: "openai", "anthropic", "mcp", "json_schema", "simple".
    Raises ValueError if the format cannot be determined.
    """
    # OpenAI: {"type": "function", "function": {...}}
    if obj.get("type") == "function" and "function" in obj:
        return "openai"

    # Anthropic: {"name": ..., "input_schema": {...}}
    if "name" in obj and "input_schema" in obj:
        return "anthropic"

    # MCP: {"name": ..., "inputSchema": {...}}
    if "name" in obj and "inputSchema" in obj:
        return "mcp"

    # JSON Schema: {"type": "object", "properties": {...}}
    if obj.get("type") == "object" and "properties" in obj:
        return "json_schema"

    # Simple/generic: {"name": ..., "description": ..., "parameters": {...}}
    if "name" in obj and "parameters" in obj:
        return "simple"

    # Fallback: if it at least has a name and description, treat as simple
    if "name" in obj and "description" in obj:
        return "simple"

    raise ValueError(
        "Cannot detect tool format. Expected one of: OpenAI, Anthropic, MCP, "
        "JSON Schema, or Simple (name + description + parameters)."
    )


# ---------------------------------------------------------------------------
# Parsing: normalize any format into (name, description, input_schema)
# ---------------------------------------------------------------------------


def _normalize_tool(obj: Dict[str, Any], fmt: str) -> Tuple[str, str, Dict[str, Any]]:
    """Extract (name, description, input_schema) from a tool definition."""
    if fmt == "openai":
        fn = obj.get("function", {})
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        schema = fn.get("parameters", {"type": "object", "properties": {}})
        return name, desc, schema

    if fmt == "anthropic":
        name = obj.get("name", "unknown")
        desc = obj.get("description", "")
        schema = obj.get("input_schema", {"type": "object", "properties": {}})
        return name, desc, schema

    if fmt == "mcp":
        name = obj.get("name", "unknown")
        desc = obj.get("description", "")
        schema = obj.get("inputSchema", {"type": "object", "properties": {}})
        return name, desc, schema

    if fmt == "json_schema":
        name = obj.get("title", "unknown")
        desc = obj.get("description", "")
        # The schema IS the object (minus title/description metadata)
        schema = {
            "type": "object",
            "properties": obj.get("properties", {}),
        }
        if "required" in obj:
            schema["required"] = obj["required"]
        return name, desc, schema

    # simple
    name = obj.get("name", "unknown")
    desc = obj.get("description", "")
    schema = obj.get("parameters", {"type": "object", "properties": {}})
    return name, desc, schema


def parse_tools(data: Any) -> List[FunctionTool]:
    """Parse JSON data into a list of FunctionTool instances.

    Accepts a single tool object or an array of tool objects in any of the
    5 supported formats. All tools in one input must share the same format.
    """
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Expected a JSON object or array, got {type(data).__name__}")

    if not items:
        return []

    tools: List[FunctionTool] = []
    for item in items:
        fmt = detect_format(item)
        name, desc, schema = _normalize_tool(item, fmt)
        # Create a stub callable so FunctionTool is happy
        def _stub(**kwargs: Any) -> str:
            return ""
        ft = FunctionTool(_stub, name, desc, input_schema=schema)
        tools.append(ft)

    return tools


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

_REPORT_FORMATS = ("openai", "anthropic", "google", "mcp", "json_schema")


def generate_report(
    tools: List[FunctionTool],
    *,
    use_color: bool = True,
) -> str:
    """Generate a formatted token cost report for the given tools.

    Returns the report as a string (with ANSI escapes if use_color is True).
    """
    # Color codes
    if use_color and sys.stderr.isatty():
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
    else:
        BOLD = CYAN = GREEN = YELLOW = GRAY = RESET = ""

    lines: List[str] = []
    lines.append(f"\n{BOLD}agent-friend audit{RESET} — tool token cost report\n")

    if not tools:
        lines.append(f"  {GRAY}No tools found in input.{RESET}\n")
        return "\n".join(lines)

    # Per-tool breakdown (use anthropic format for the per-tool view, it's native)
    lines.append(f"  {BOLD}{'Tool':<24s}{'Description':<16s}{'Tokens (est.)':>14s}{RESET}")

    total_tokens = 0
    long_descriptions: List[Tuple[str, int]] = []

    for ft in tools:
        desc_len = len(ft.description)
        # Estimate tokens for this single tool using anthropic format
        est = ft.token_estimate(format="anthropic")
        total_tokens += est
        lines.append(
            f"  {CYAN}{ft.name:<24s}{RESET}"
            f"{GRAY}{desc_len} chars{RESET}{'':>{10 - len(str(desc_len))}s}"
            f"{GREEN}~{est} tokens{RESET}"
        )
        if desc_len > 200:
            long_descriptions.append((ft.name, desc_len))

    # Separator
    lines.append(f"  {GRAY}{'─' * 54}{RESET}")
    lines.append(
        f"  {BOLD}Total ({len(tools)} tool{'s' if len(tools) != 1 else ''}){RESET}"
        f"{'':>24s}{GREEN}~{total_tokens} tokens{RESET}"
    )

    # Format comparison using Toolkit
    kit = Toolkit(tools)
    report = kit.token_report()

    lines.append(f"\n  {BOLD}Format comparison (total):{RESET}")

    # Find cheapest for annotation
    least = report["least_expensive"]

    for fmt in _REPORT_FORMATS:
        est = report["estimates"][fmt]
        marker = f"  {CYAN}<- cheapest{RESET}" if fmt == least else ""
        lines.append(f"    {fmt:<14s}{GREEN}~{est} tokens{RESET}{marker}")

    # Recommendations
    if long_descriptions:
        lines.append("")
        for name, length in long_descriptions:
            lines.append(
                f"  {YELLOW}! {name}: description is {length} chars "
                f"— consider trimming for lower token cost{RESET}"
            )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_audit(file_path: Optional[str] = None, use_color: bool = True) -> int:
    """Run the audit command. Returns exit code (0 = success, 1 = error).

    Parameters
    ----------
    file_path:
        Path to a JSON file, or "-" for stdin, or None to read from stdin.
    use_color:
        Whether to use ANSI color codes in output.
    """
    try:
        if file_path is None or file_path == "-":
            raw = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                raw = f.read()
    except FileNotFoundError:
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error reading input: {e}", file=sys.stderr)
        return 1

    raw = raw.strip()
    if not raw:
        print(generate_report([], use_color=use_color))
        return 0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON: {e}", file=sys.stderr)
        return 1

    try:
        tools = parse_tools(data)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(generate_report(tools, use_color=use_color))
    return 0
