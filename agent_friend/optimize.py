"""optimize.py — Heuristic-based tool schema optimizer.

Reads tool definitions from JSON (any of 5 supported formats), analyzes them
for token-wasting patterns, and suggests specific rewrites with estimated
token savings. Think of it as a linter for MCP/OpenAI/Anthropic tool defs.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from .audit import parse_tools, detect_format, _normalize_tool
from .tools.function_tool import FunctionTool
from .toolkit import Toolkit


# ---------------------------------------------------------------------------
# Suggestion data structure
# ---------------------------------------------------------------------------

class Suggestion:
    """A single optimization suggestion."""

    def __init__(
        self,
        tool_name: str,
        rule: str,
        message: str,
        token_savings: int,
        detail: Optional[str] = None,
    ) -> None:
        self.tool_name = tool_name
        self.rule = rule
        self.message = message
        self.token_savings = token_savings
        self.detail = detail

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "tool": self.tool_name,
            "rule": self.rule,
            "message": self.message,
            "token_savings": self.token_savings,
        }
        if self.detail is not None:
            d["detail"] = self.detail
        return d


# ---------------------------------------------------------------------------
# Verbose description prefixes (rule 1)
# ---------------------------------------------------------------------------

_VERBOSE_PREFIXES = [
    "This is a tool that ",
    "This tool allows you to ",
    "Allows the user to ",
    "Use this tool to ",
    "This function ",
    "A tool that ",
    "Used to ",
]


def _check_verbose_prefix(
    tool_name: str, description: str
) -> Optional[Suggestion]:
    """Detect and suggest removing filler prefixes from tool descriptions."""
    desc_lower = description.lstrip()
    for prefix in _VERBOSE_PREFIXES:
        if desc_lower.lower().startswith(prefix.lower()):
            # Find the actual prefix in original text (preserve case of remainder)
            prefix_len = len(prefix)
            # Find where the actual content starts in the original string
            stripped = description.lstrip()
            remainder = stripped[prefix_len:]
            # Capitalize the first letter of the remainder
            if remainder:
                suggested = remainder[0].upper() + remainder[1:]
            else:
                suggested = remainder
            saved = int(len(prefix) / 4)
            return Suggestion(
                tool_name=tool_name,
                rule="verbose_prefix",
                message=(
                    "Description prefix: \"{prefix}{short}\" -> \"{suggested}\""
                    .format(
                        prefix=prefix,
                        short=remainder[:30] + "..." if len(remainder) > 30 else remainder,
                        suggested=suggested[:40] + "..." if len(suggested) > 40 else suggested,
                    )
                ),
                token_savings=saved,
                detail="Remove filler prefix for a more direct description.",
            )
    return None


# ---------------------------------------------------------------------------
# Long descriptions (rule 2)
# ---------------------------------------------------------------------------

_MAX_DESC_LENGTH = 200


def _check_long_description(
    tool_name: str, description: str
) -> Optional[Suggestion]:
    """Flag descriptions longer than 200 characters."""
    length = len(description)
    if length > _MAX_DESC_LENGTH:
        excess = length - _MAX_DESC_LENGTH
        saved = int(excess / 4)
        return Suggestion(
            tool_name=tool_name,
            rule="long_description",
            message="Description is {length} chars, suggest <={max}".format(
                length=length, max=_MAX_DESC_LENGTH,
            ),
            token_savings=saved,
            detail="Long descriptions add token overhead. Trim to essentials.",
        )
    return None


# ---------------------------------------------------------------------------
# Long parameter descriptions (rule 3)
# ---------------------------------------------------------------------------

_MAX_PARAM_DESC_LENGTH = 100
_TARGET_PARAM_DESC_LENGTH = 80


def _check_long_param_descriptions(
    tool_name: str, schema: Dict[str, Any]
) -> List[Suggestion]:
    """Flag parameter descriptions longer than 100 characters."""
    suggestions = []
    properties = schema.get("properties", {})
    for param_name, param_schema in properties.items():
        desc = param_schema.get("description", "")
        if len(desc) > _MAX_PARAM_DESC_LENGTH:
            excess = len(desc) - _TARGET_PARAM_DESC_LENGTH
            saved = int(excess / 4)
            suggestions.append(Suggestion(
                tool_name=tool_name,
                rule="long_param_description",
                message="Parameter '{param}': description is {length} chars, suggest <={target}".format(
                    param=param_name,
                    length=len(desc),
                    target=_TARGET_PARAM_DESC_LENGTH,
                ),
                token_savings=saved,
            ))
    return suggestions


# ---------------------------------------------------------------------------
# Redundant parameter descriptions (rule 4)
# ---------------------------------------------------------------------------

_STRIP_WORDS = {"the", "a", "an"}


def _normalize_for_redundancy(text: str) -> str:
    """Normalize text for redundancy comparison.

    Strips articles, trailing periods, and lowercases.
    """
    text = text.lower().strip().rstrip(".")
    words = text.split()
    words = [w for w in words if w not in _STRIP_WORDS]
    return " ".join(words)


def _check_redundant_param_descriptions(
    tool_name: str, schema: Dict[str, Any]
) -> List[Suggestion]:
    """Flag parameter descriptions that just restate the parameter name."""
    suggestions = []
    properties = schema.get("properties", {})
    for param_name, param_schema in properties.items():
        desc = param_schema.get("description", "")
        if not desc:
            continue
        normalized_desc = _normalize_for_redundancy(desc)
        normalized_name = _normalize_for_redundancy(param_name.replace("_", " "))
        if normalized_desc == normalized_name:
            saved = int(len(desc) / 4) + 1  # +1 for the key overhead
            suggestions.append(Suggestion(
                tool_name=tool_name,
                rule="redundant_param_description",
                message="Parameter '{param}': description \"{desc}\" restates parameter name".format(
                    param=param_name, desc=desc,
                ),
                token_savings=saved,
                detail="Remove redundant description to save tokens.",
            ))
    return suggestions


# ---------------------------------------------------------------------------
# Empty or missing descriptions (rule 5)
# ---------------------------------------------------------------------------

_COMPLEX_TYPES = {"object", "array"}


def _check_missing_descriptions(
    tool_name: str, description: str, schema: Dict[str, Any]
) -> List[Suggestion]:
    """Flag tools with no description, and complex-type params without descriptions."""
    suggestions = []

    # Tool-level missing description
    if not description.strip():
        suggestions.append(Suggestion(
            tool_name=tool_name,
            rule="missing_description",
            message="Tool has no description",
            token_savings=0,
            detail="Add a brief description so the model knows when to use this tool.",
        ))

    # Parameter-level: only flag complex types
    properties = schema.get("properties", {})
    for param_name, param_schema in properties.items():
        param_desc = param_schema.get("description", "")
        param_type = param_schema.get("type", "")
        if isinstance(param_type, list):
            type_set = set(param_type)
        else:
            type_set = {param_type}

        if not param_desc.strip() and type_set & _COMPLEX_TYPES:
            suggestions.append(Suggestion(
                tool_name=tool_name,
                rule="missing_param_description",
                message="Parameter '{param}' (type: {ptype}) has no description".format(
                    param=param_name,
                    ptype=param_type if isinstance(param_type, str) else "/".join(param_type),
                ),
                token_savings=0,
                detail="Complex parameters benefit from a description.",
            ))

    return suggestions


# ---------------------------------------------------------------------------
# Duplicate cross-tool parameter descriptions (rule 6)
# ---------------------------------------------------------------------------

def _check_duplicate_param_descriptions(
    tools_data: List[Tuple[str, str, Dict[str, Any]]]
) -> List[Suggestion]:
    """Find parameters with identical descriptions across 3+ tools."""
    # Map: (param_name, description) -> list of tool names
    param_desc_map = {}  # type: Dict[Tuple[str, str], List[str]]

    for tool_name, _desc, schema in tools_data:
        properties = schema.get("properties", {})
        for param_name, param_schema in properties.items():
            pdesc = param_schema.get("description", "")
            if not pdesc:
                continue
            key = (param_name, pdesc)
            if key not in param_desc_map:
                param_desc_map[key] = []
            param_desc_map[key].append(tool_name)

    suggestions = []
    for (param_name, pdesc), tool_names in sorted(param_desc_map.items()):
        if len(tool_names) >= 3:
            # Suggest a shorter version
            words = pdesc.split()
            if len(words) > 3:
                short = " ".join(words[:3])
            else:
                short = pdesc
            per_tool_savings = max(1, int((len(pdesc) - len(short)) / 4))
            total_savings = per_tool_savings * len(tool_names)
            suggestions.append(Suggestion(
                tool_name="(cross-tool)",
                rule="duplicate_param_description",
                message=(
                    "{count} tools share identical description for '{param}': \"{desc}\""
                    .format(count=len(tool_names), param=param_name, desc=pdesc)
                ),
                token_savings=total_savings,
                detail="Tools: {tools}. Consider shortening to: \"{short}\"".format(
                    tools=", ".join(tool_names), short=short,
                ),
            ))

    return suggestions


# ---------------------------------------------------------------------------
# Deep nesting overhead (rule 7)
# ---------------------------------------------------------------------------

def _measure_nesting(schema: Any, depth: int = 0) -> int:
    """Return the maximum nesting depth of object types in a schema."""
    if not isinstance(schema, dict):
        return depth
    max_depth = depth
    if schema.get("type") == "object" and "properties" in schema:
        for prop_schema in schema["properties"].values():
            d = _measure_nesting(prop_schema, depth + 1)
            if d > max_depth:
                max_depth = d
    # Also check items for arrays containing objects
    if schema.get("type") == "array" and "items" in schema:
        d = _measure_nesting(schema["items"], depth)
        if d > max_depth:
            max_depth = d
    return max_depth


def _check_deep_nesting(
    tool_name: str, schema: Dict[str, Any]
) -> Optional[Suggestion]:
    """Flag schemas with more than 2 levels of nested objects."""
    depth = _measure_nesting(schema)
    if depth > 2:
        # Each extra level adds ~10-15 structural tokens
        extra_levels = depth - 2
        saved = extra_levels * 12  # ~12 tokens per extra nesting level
        return Suggestion(
            tool_name=tool_name,
            rule="deep_nesting",
            message="Schema has {depth} levels of nesting (recommend <=2)".format(
                depth=depth,
            ),
            token_savings=saved,
            detail="Each nesting level adds ~10-15 structural tokens (braces, type, properties keys).",
        )
    return None


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_tools(data: Any) -> Tuple[List[Suggestion], Dict[str, Any]]:
    """Analyze tool definitions and return optimization suggestions.

    Parameters
    ----------
    data:
        Parsed JSON data (dict or list of tool definitions).

    Returns
    -------
    Tuple of (suggestions, stats) where stats contains:
        - tools_analyzed: int
        - current_tokens: int
        - estimated_savings: int
    """
    tools = parse_tools(data)

    if not tools:
        return [], {"tools_analyzed": 0, "current_tokens": 0, "estimated_savings": 0}

    # Normalize all tools to (name, description, schema) for cross-tool checks
    if isinstance(data, dict):
        items = [data]
    else:
        items = data

    tools_data = []  # type: List[Tuple[str, str, Dict[str, Any]]]
    for item in items:
        fmt = detect_format(item)
        name, desc, schema = _normalize_tool(item, fmt)
        tools_data.append((name, desc, schema))

    suggestions = []  # type: List[Suggestion]

    # Per-tool checks
    for tool_name, desc, schema in tools_data:
        # Rule 1: verbose prefix
        s = _check_verbose_prefix(tool_name, desc)
        if s:
            suggestions.append(s)

        # Rule 2: long description
        s = _check_long_description(tool_name, desc)
        if s:
            suggestions.append(s)

        # Rule 3: long parameter descriptions
        suggestions.extend(_check_long_param_descriptions(tool_name, schema))

        # Rule 4: redundant parameter descriptions
        suggestions.extend(_check_redundant_param_descriptions(tool_name, schema))

        # Rule 5: missing descriptions
        suggestions.extend(_check_missing_descriptions(tool_name, desc, schema))

        # Rule 7: deep nesting
        s = _check_deep_nesting(tool_name, schema)
        if s:
            suggestions.append(s)

    # Rule 6: cross-tool duplicate parameter descriptions
    suggestions.extend(_check_duplicate_param_descriptions(tools_data))

    # Calculate token stats
    kit = Toolkit(tools)
    current_tokens = kit.token_estimate(format="anthropic")
    estimated_savings = sum(s.token_savings for s in suggestions)

    stats = {
        "tools_analyzed": len(tools),
        "current_tokens": current_tokens,
        "estimated_savings": estimated_savings,
    }

    return suggestions, stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_optimize_report(
    suggestions: List[Suggestion],
    stats: Dict[str, Any],
    *,
    use_color: bool = True,
) -> str:
    """Generate a formatted optimization report.

    Returns the report as a string (with ANSI escapes if use_color is True).
    """
    if use_color and sys.stderr.isatty():
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
    else:
        BOLD = CYAN = GREEN = YELLOW = GRAY = RESET = ""

    lines = []  # type: List[str]
    lines.append("")
    lines.append("{bold}=== agent-friend optimize ==={reset}".format(
        bold=BOLD, reset=RESET,
    ))

    tools_analyzed = stats.get("tools_analyzed", 0)
    current_tokens = stats.get("current_tokens", 0)
    estimated_savings = stats.get("estimated_savings", 0)

    if tools_analyzed == 0:
        lines.append("")
        lines.append("  {gray}No tools found in input.{reset}".format(
            gray=GRAY, reset=RESET,
        ))
        lines.append("")
        return "\n".join(lines)

    if not suggestions:
        lines.append("")
        lines.append("  {green}No optimization suggestions. Your schemas look good!{reset}".format(
            green=GREEN, reset=RESET,
        ))
        lines.append("")
        lines.append("  Tools analyzed: {count}".format(count=tools_analyzed))
        lines.append("  Current total: ~{tokens} tokens".format(tokens=current_tokens))
        lines.append("")
        return "\n".join(lines)

    # Group suggestions by tool
    per_tool = {}  # type: Dict[str, List[Suggestion]]
    cross_tool = []  # type: List[Suggestion]
    for s in suggestions:
        if s.tool_name == "(cross-tool)":
            cross_tool.append(s)
        else:
            if s.tool_name not in per_tool:
                per_tool[s.tool_name] = []
            per_tool[s.tool_name].append(s)

    lines.append("")

    # Per-tool suggestions
    for tool_name, tool_suggestions in per_tool.items():
        lines.append("Tool: {cyan}{name}{reset}".format(
            cyan=CYAN, name=tool_name, reset=RESET,
        ))
        for s in tool_suggestions:
            lines.append("  {yellow}{lightning}{reset} {msg}".format(
                yellow=YELLOW, lightning="\u26a1", reset=RESET, msg=s.message,
            ))
            lines.append("     Saves ~{tokens} tokens".format(tokens=s.token_savings))
        lines.append("")

    # Cross-tool suggestions
    if cross_tool:
        lines.append("Cross-tool:")
        for s in cross_tool:
            lines.append("  {yellow}{lightning}{reset} {msg}".format(
                yellow=YELLOW, lightning="\u26a1", reset=RESET, msg=s.message,
            ))
            if s.detail:
                lines.append("     {detail}".format(detail=s.detail))
            lines.append("     Saves ~{tokens} tokens".format(tokens=s.token_savings))
        lines.append("")

    # Summary
    optimized_tokens = current_tokens - estimated_savings
    if current_tokens > 0:
        pct = int(round(estimated_savings / current_tokens * 100))
    else:
        pct = 0

    lines.append("Summary:")
    lines.append("  Tools analyzed: {count}".format(count=tools_analyzed))
    lines.append("  Suggestions: {count}".format(count=len(suggestions)))
    lines.append("  Estimated total savings: ~{tokens} tokens ({pct}% reduction)".format(
        tokens=estimated_savings, pct=pct,
    ))
    lines.append("  Current total: ~{cur} tokens".format(cur=current_tokens))
    lines.append("  Optimized total: ~{opt} tokens".format(opt=optimized_tokens))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def generate_json_output(
    suggestions: List[Suggestion],
    stats: Dict[str, Any],
) -> str:
    """Generate machine-readable JSON output."""
    output = {
        "suggestions": [s.to_dict() for s in suggestions],
        "stats": stats,
    }
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_optimize(
    file_path: Optional[str] = None,
    use_color: bool = True,
    json_output: bool = False,
) -> int:
    """Run the optimize command. Returns exit code (0 = success, 1 = error).

    Parameters
    ----------
    file_path:
        Path to a JSON file, or "-" for stdin, or None to read from stdin.
    use_color:
        Whether to use ANSI color codes in output.
    json_output:
        Whether to output JSON instead of a human-readable report.
    """
    try:
        if file_path is None or file_path == "-":
            raw = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                raw = f.read()
    except FileNotFoundError:
        print("Error: file not found: {path}".format(path=file_path), file=sys.stderr)
        return 1
    except Exception as e:
        print("Error reading input: {err}".format(err=e), file=sys.stderr)
        return 1

    raw = raw.strip()
    if not raw:
        if json_output:
            print(generate_json_output([], {"tools_analyzed": 0, "current_tokens": 0, "estimated_savings": 0}))
        else:
            print(generate_optimize_report(
                [], {"tools_analyzed": 0, "current_tokens": 0, "estimated_savings": 0},
                use_color=use_color,
            ))
        return 0

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Error: invalid JSON: {err}".format(err=e), file=sys.stderr)
        return 1

    try:
        suggestions, stats = analyze_tools(data)
    except ValueError as e:
        print("Error: {err}".format(err=e), file=sys.stderr)
        return 1

    if json_output:
        print(generate_json_output(suggestions, stats))
    else:
        print(generate_optimize_report(suggestions, stats, use_color=use_color))

    return 0
