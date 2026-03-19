"""fix.py — Auto-fix MCP tool schema issues.

Reads tool definitions from JSON (any of 5 supported formats), applies safe
auto-fixes, and outputs the fixed JSON. Think of it as ESLint --fix for MCP
tool schemas.

The fix pipeline complements validate and optimize:
    validate -> audit -> optimize -> fix -> grade
"""

import copy
import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from .audit import detect_format, _normalize_tool


# ---------------------------------------------------------------------------
# Change data structure
# ---------------------------------------------------------------------------

class Change:
    """A single auto-fix change applied to a tool schema."""

    def __init__(
        self,
        tool_name: str,
        rule: str,
        message: str,
    ) -> None:
        self.tool_name = tool_name
        self.rule = rule
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool_name,
            "rule": self.rule,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Constants — duplicated from optimize.py to keep fix.py self-contained
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

_STRIP_WORDS = {"the", "a", "an"}

_MAX_DESC_LENGTH = 200
_MAX_PARAM_DESC_LENGTH = 80

# Rule name -> short alias for --only flag
_RULE_ALIASES = {
    "fix_names": "names",
    "fix_undefined_schemas": "schemas",
    "fix_verbose_prefixes": "prefixes",
    "fix_redundant_params": "redundant",
    "fix_long_descriptions": "descriptions",
    "fix_long_param_descriptions": "param_descriptions",
}

_ALL_RULES = list(_RULE_ALIASES.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_for_redundancy(text: str) -> str:
    """Normalize text for redundancy comparison.

    Strips articles, trailing periods, and lowercases.
    """
    text = text.lower().strip().rstrip(".")
    words = text.split()
    words = [w for w in words if w not in _STRIP_WORDS]
    return " ".join(words)


def _token_estimate(data: Any) -> int:
    """Estimate token count for a JSON structure."""
    return len(json.dumps(data)) // 4


def _truncate_at_sentence(text: str, max_len: int) -> str:
    """Truncate text at the last sentence boundary before max_len.

    If no sentence boundary is found, truncate at max_len - 3 and add '...'.
    """
    if len(text) <= max_len:
        return text

    # Look for the last sentence-ending punctuation before max_len
    candidate = text[:max_len]
    # Find last sentence boundary (. ! ?)
    last_period = -1
    for i in range(len(candidate) - 1, -1, -1):
        if candidate[i] in ".!?":
            last_period = i
            break

    if last_period > 0:
        return candidate[:last_period + 1]

    # No sentence boundary found — hard truncate
    return text[:max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Format-aware accessors and mutators
# ---------------------------------------------------------------------------

def _get_name(obj: Dict[str, Any], fmt: str) -> str:
    """Get the tool name from a raw tool object."""
    if fmt == "openai":
        return obj.get("function", {}).get("name", "unknown")
    if fmt == "json_schema":
        return obj.get("title", "unknown")
    return obj.get("name", "unknown")


def _set_name(obj: Dict[str, Any], fmt: str, new_name: str) -> None:
    """Set the tool name on a raw tool object."""
    if fmt == "openai":
        obj.setdefault("function", {})["name"] = new_name
    elif fmt == "json_schema":
        obj["title"] = new_name
    else:
        obj["name"] = new_name


def _get_description(obj: Dict[str, Any], fmt: str) -> str:
    """Get the tool description from a raw tool object."""
    if fmt == "openai":
        return obj.get("function", {}).get("description", "")
    return obj.get("description", "")


def _set_description(obj: Dict[str, Any], fmt: str, new_desc: str) -> None:
    """Set the tool description on a raw tool object."""
    if fmt == "openai":
        obj.setdefault("function", {})["description"] = new_desc
    else:
        obj["description"] = new_desc


def _get_schema(obj: Dict[str, Any], fmt: str) -> Optional[Dict[str, Any]]:
    """Get the parameters/input schema from a raw tool object."""
    if fmt == "openai":
        return obj.get("function", {}).get("parameters")
    if fmt == "anthropic":
        return obj.get("input_schema")
    if fmt == "mcp":
        return obj.get("inputSchema")
    if fmt == "json_schema":
        return obj  # The object itself is the schema
    return obj.get("parameters")


# ---------------------------------------------------------------------------
# Individual fix rules
# ---------------------------------------------------------------------------

def _camel_to_snake(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case."""
    # Handle sequences like "XMLParser" -> "xml_parser"
    s1 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    # Handle standard camelCase like "appendBlockChildren" -> "append_block_children"
    s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1)
    return s2.lower()


def _fix_names(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 1: Convert camelCase/PascalCase/kebab-case names to snake_case."""
    changes = []
    name = _get_name(obj, fmt)

    if not name or name == "unknown":
        return changes

    new_name = name
    # Convert kebab-case to snake_case
    if "-" in new_name:
        new_name = new_name.replace("-", "_")

    # Convert camelCase or PascalCase to snake_case
    if new_name != new_name.lower() or re.search(r'[A-Z]', new_name):
        new_name = _camel_to_snake(new_name)

    # Collapse any double underscores
    new_name = re.sub(r'_+', '_', new_name).strip('_')

    if new_name != name:
        _set_name(obj, fmt, new_name)
        changes.append(Change(
            tool_name=new_name,
            rule="fix_names",
            message="{old} -> {new} (name)".format(old=name, new=new_name),
        ))

    return changes


def _fix_undefined_schemas(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 2: Add properties: {} to object-type params missing it."""
    changes = []
    name = _get_name(obj, fmt)
    schema = _get_schema(obj, fmt)

    if schema is None:
        return changes

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return changes

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") == "object" and "properties" not in param_schema:
            param_schema["properties"] = {}
            changes.append(Change(
                tool_name=name,
                rule="fix_undefined_schemas",
                message="Added properties to undefined object in {tool}.{param}".format(
                    tool=name, param=param_name,
                ),
            ))

    return changes


def _fix_verbose_prefixes(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 3: Strip filler prefixes from descriptions."""
    changes = []
    name = _get_name(obj, fmt)
    desc = _get_description(obj, fmt)

    if not desc:
        return changes

    stripped = desc.lstrip()
    for prefix in _VERBOSE_PREFIXES:
        if stripped.lower().startswith(prefix.lower()):
            prefix_len = len(prefix)
            remainder = stripped[prefix_len:]
            if remainder:
                new_desc = remainder[0].upper() + remainder[1:]
            else:
                new_desc = remainder

            _set_description(obj, fmt, new_desc)

            short_prefix = prefix.rstrip()
            changes.append(Change(
                tool_name=name,
                rule="fix_verbose_prefixes",
                message="Stripped \"{prefix}\" from {tool} description".format(
                    prefix=short_prefix, tool=name,
                ),
            ))
            break

    return changes


def _fix_redundant_params(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 4: Remove parameter descriptions that just restate the param name."""
    changes = []
    name = _get_name(obj, fmt)
    schema = _get_schema(obj, fmt)

    if schema is None:
        return changes

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return changes

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not desc:
            continue

        normalized_desc = _normalize_for_redundancy(desc)
        normalized_name = _normalize_for_redundancy(param_name.replace("_", " "))

        if normalized_desc == normalized_name:
            del param_schema["description"]
            changes.append(Change(
                tool_name=name,
                rule="fix_redundant_params",
                message="Removed redundant description from '{param}' parameter".format(
                    param=param_name,
                ),
            ))

    return changes


def _fix_long_descriptions(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 5: Truncate tool descriptions to 200 chars."""
    changes = []
    name = _get_name(obj, fmt)
    desc = _get_description(obj, fmt)

    if not desc or len(desc) <= _MAX_DESC_LENGTH:
        return changes

    old_len = len(desc)
    new_desc = _truncate_at_sentence(desc, _MAX_DESC_LENGTH)
    _set_description(obj, fmt, new_desc)

    changes.append(Change(
        tool_name=name,
        rule="fix_long_descriptions",
        message="Trimmed {tool} description ({old} -> {new} chars)".format(
            tool=name, old=old_len, new=len(new_desc),
        ),
    ))

    return changes


def _fix_long_param_descriptions(obj: Dict[str, Any], fmt: str) -> List[Change]:
    """Fix 6: Truncate parameter descriptions to 80 chars."""
    changes = []
    name = _get_name(obj, fmt)
    schema = _get_schema(obj, fmt)

    if schema is None:
        return changes

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return changes

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not desc or len(desc) <= _MAX_PARAM_DESC_LENGTH:
            continue

        old_len = len(desc)
        new_desc = _truncate_at_sentence(desc, _MAX_PARAM_DESC_LENGTH)
        param_schema["description"] = new_desc

        changes.append(Change(
            tool_name=name,
            rule="fix_long_param_descriptions",
            message="Trimmed {tool}.{param} description ({old} -> {new} chars)".format(
                tool=name, param=param_name, old=old_len, new=len(new_desc),
            ),
        ))

    return changes


# ---------------------------------------------------------------------------
# Main fix logic
# ---------------------------------------------------------------------------

def fix_tools(
    data: Any,
    only: Optional[List[str]] = None,
) -> Tuple[Any, List[Change]]:
    """Apply auto-fixes to tool definitions.

    Parameters
    ----------
    data:
        Parsed JSON data (dict or list of tool definitions).
    only:
        Optional list of rule names or aliases to apply.
        If None, all rules are applied.

    Returns
    -------
    Tuple of (fixed_data, changes) where fixed_data is a deep copy of the
    input with fixes applied, and changes is a list of Change objects.
    """
    fixed = copy.deepcopy(data)

    # Determine which items to process
    if isinstance(fixed, dict):
        items = [fixed]
    elif isinstance(fixed, list):
        items = fixed
    else:
        return fixed, []

    if not items:
        return fixed, []

    # Resolve rule filter
    active_rules = set(_ALL_RULES)
    if only is not None:
        # Build reverse alias map
        alias_to_rule = {}
        for rule, alias in _RULE_ALIASES.items():
            alias_to_rule[alias] = rule
            alias_to_rule[rule] = rule

        active_rules = set()
        for name in only:
            if name in alias_to_rule:
                active_rules.add(alias_to_rule[name])
            else:
                # Try partial match
                for rule, alias in _RULE_ALIASES.items():
                    if alias.startswith(name) or rule.startswith(name):
                        active_rules.add(rule)
                        break

    # Apply fix rules to each tool
    all_changes = []  # type: List[Change]

    # Map rule name to fix function
    rule_functions = [
        ("fix_names", _fix_names),
        ("fix_undefined_schemas", _fix_undefined_schemas),
        ("fix_verbose_prefixes", _fix_verbose_prefixes),
        ("fix_redundant_params", _fix_redundant_params),
        ("fix_long_descriptions", _fix_long_descriptions),
        ("fix_long_param_descriptions", _fix_long_param_descriptions),
    ]

    for item in items:
        try:
            fmt = detect_format(item)
        except ValueError:
            continue

        for rule_name, fix_fn in rule_functions:
            if rule_name in active_rules:
                changes = fix_fn(item, fmt)
                all_changes.extend(changes)

    return fixed, all_changes


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_fix_report(
    changes: List[Change],
    tool_count: int,
    tokens_before: int,
    tokens_after: int,
    file_name: str = "input",
    detected_format: str = "unknown",
    *,
    use_color: bool = True,
) -> str:
    """Generate a formatted fix report.

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

    version = __import__("agent_friend").__version__

    lines = []  # type: List[str]
    lines.append("")
    lines.append("{bold}agent-friend fix{reset} v{version}".format(
        bold=BOLD, reset=RESET, version=version,
    ))
    lines.append("")

    if tool_count == 0:
        lines.append("  {gray}No tools found in input.{reset}".format(
            gray=GRAY, reset=RESET,
        ))
        lines.append("")
        return "\n".join(lines)

    lines.append("  Fixing: {file} ({count} tool{s}, {fmt} format detected)".format(
        file=file_name,
        count=tool_count,
        s="s" if tool_count != 1 else "",
        fmt=detected_format,
    ))
    lines.append("")

    if not changes:
        lines.append("  {green}No fixes needed. Your schemas look clean!{reset}".format(
            green=GREEN, reset=RESET,
        ))
        lines.append("")
        return "\n".join(lines)

    lines.append("  Applied fixes:")
    for change in changes:
        lines.append("    {green}{check}{reset} {msg}".format(
            green=GREEN, check="\u2713", reset=RESET, msg=change.message,
        ))
    lines.append("")

    lines.append("  Summary: {count} fix{es} applied across {tools} tool{s}".format(
        count=len(changes),
        es="es" if len(changes) != 1 else "",
        tools=tool_count,
        s="s" if tool_count != 1 else "",
    ))

    if tokens_before > 0:
        reduction = tokens_before - tokens_after
        if tokens_before > 0:
            pct = (reduction / tokens_before) * 100
        else:
            pct = 0.0
        lines.append("  Token reduction: {before} -> {after} tokens (-{pct:.1f}%)".format(
            before=tokens_before,
            after=tokens_after,
            pct=pct,
        ))

    lines.append("")
    lines.append("  {gray}Fixed JSON written to stdout. Pipe to file:{reset}".format(
        gray=GRAY, reset=RESET,
    ))
    lines.append("    {gray}agent-friend fix {file} > {file}_fixed.json{reset}".format(
        gray=GRAY, file=file_name, reset=RESET,
    ))
    lines.append("")

    return "\n".join(lines)


def generate_diff_report(
    original: Any,
    fixed: Any,
    changes: List[Change],
    *,
    use_color: bool = True,
) -> str:
    """Generate a before/after diff report."""
    if use_color and sys.stderr.isatty():
        RED = "\033[31m"
        GREEN = "\033[32m"
        CYAN = "\033[36m"
        GRAY = "\033[90m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        RED = GREEN = CYAN = GRAY = BOLD = RESET = ""

    lines = []  # type: List[str]
    lines.append("")
    lines.append("{bold}Diff:{reset}".format(bold=BOLD, reset=RESET))
    lines.append("")

    original_str = json.dumps(original, indent=2).splitlines()
    fixed_str = json.dumps(fixed, indent=2).splitlines()

    # Simple line-by-line diff
    max_lines = max(len(original_str), len(fixed_str))
    for i in range(max_lines):
        orig_line = original_str[i] if i < len(original_str) else ""
        fix_line = fixed_str[i] if i < len(fixed_str) else ""

        if orig_line != fix_line:
            if orig_line:
                lines.append("  {red}- {line}{reset}".format(
                    red=RED, line=orig_line, reset=RESET,
                ))
            if fix_line:
                lines.append("  {green}+ {line}{reset}".format(
                    green=GREEN, line=fix_line, reset=RESET,
                ))
        # Only show unchanged lines that are near changes (context)
        # For brevity, skip unchanged lines in the diff output

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fix(
    file_path: Optional[str] = None,
    use_color: bool = True,
    json_output: bool = False,
    diff: bool = False,
    dry_run: bool = False,
    only: Optional[List[str]] = None,
) -> int:
    """Run the fix command. Returns exit code.

    Exit codes:
        0 = success
        1 = file read / parse error

    Parameters
    ----------
    file_path:
        Path to a JSON file, or "-" for stdin, or None to read from stdin.
    use_color:
        Whether to use ANSI color codes in output.
    json_output:
        If True, output only the fixed JSON (no report text).
    diff:
        If True, show a before/after diff.
    dry_run:
        If True, show what would change without outputting fixed JSON.
    only:
        Optional list of rule names/aliases to apply.
    """
    # Read input
    try:
        if file_path is None or file_path == "-":
            raw = sys.stdin.read()
            file_name = "stdin"
        else:
            with open(file_path, "r") as f:
                raw = f.read()
            file_name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    except FileNotFoundError:
        print("Error: file not found: {path}".format(path=file_path), file=sys.stderr)
        return 1
    except Exception as e:
        print("Error reading input: {err}".format(err=e), file=sys.stderr)
        return 1

    raw = raw.strip()
    if not raw:
        if json_output:
            print("[]")
        else:
            print(generate_fix_report([], 0, 0, 0, file_name, use_color=use_color))
        return 0

    # Parse JSON
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("Error: invalid JSON: {err}".format(err=e), file=sys.stderr)
        return 1

    # Keep original for diff and token comparison
    original = copy.deepcopy(data)

    # Count tools
    if isinstance(data, dict):
        items = [data]
    elif isinstance(data, list):
        items = data
    else:
        print("Error: expected JSON object or array", file=sys.stderr)
        return 1

    tool_count = len(items)

    # Detect format from first tool
    detected_format = "unknown"
    if items:
        try:
            detected_format = detect_format(items[0])
        except ValueError:
            pass

    # Calculate tokens before
    tokens_before = _token_estimate(data)

    # Apply fixes
    try:
        fixed_data, changes = fix_tools(data, only=only)
    except Exception as e:
        print("Error: {err}".format(err=e), file=sys.stderr)
        return 1

    # Calculate tokens after
    tokens_after = _token_estimate(fixed_data)

    # Output
    if dry_run:
        # Show report only, no JSON output
        report = generate_fix_report(
            changes, tool_count, tokens_before, tokens_after,
            file_name, detected_format, use_color=use_color,
        )
        # Replace the "Fixed JSON written to stdout" line
        report = report.replace(
            "Fixed JSON written to stdout. Pipe to file:",
            "Dry run — no changes written.",
        )
        print(report, file=sys.stderr)
        if diff and changes:
            diff_report = generate_diff_report(
                original, fixed_data, changes, use_color=use_color,
            )
            print(diff_report, file=sys.stderr)
        return 0

    if json_output:
        # Output only the fixed JSON
        print(json.dumps(fixed_data, indent=2))
        return 0

    # Default: show report to stderr, fixed JSON to stdout
    report = generate_fix_report(
        changes, tool_count, tokens_before, tokens_after,
        file_name, detected_format, use_color=use_color,
    )
    print(report, file=sys.stderr)

    if diff and changes:
        diff_report = generate_diff_report(
            original, fixed_data, changes, use_color=use_color,
        )
        print(diff_report, file=sys.stderr)

    # Always output the fixed JSON to stdout
    print(json.dumps(fixed_data, indent=2))

    return 0
