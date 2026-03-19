"""validate.py — Validate tool schemas for correctness errors.

Reads tool definitions from JSON (any of 5 supported formats), checks them
for structural and semantic correctness issues, and produces a report.

Different from audit (token cost) and optimize (bloat suggestions) — this
module checks whether schemas are actually *correct*.
"""

import json
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from .audit import detect_format, _normalize_tool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_JSON_SCHEMA_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


# ---------------------------------------------------------------------------
# Issue data structure
# ---------------------------------------------------------------------------

class Issue:
    """A single validation issue."""

    def __init__(
        self,
        tool: str,
        severity: str,
        check: str,
        message: str,
    ) -> None:
        self.tool = tool
        self.severity = severity  # "error" or "warn"
        self.check = check
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
        }


# ---------------------------------------------------------------------------
# Raw tool extraction (works before full normalization)
# ---------------------------------------------------------------------------

def _extract_raw_tools(data: Any) -> List[Dict[str, Any]]:
    """Extract raw tool dicts from input data.

    Returns a list of raw tool objects without normalization.
    """
    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return list(data)
    else:
        return []


def _get_tool_name(obj: Dict[str, Any], fmt: str) -> Optional[str]:
    """Get tool name from a raw tool object given its format."""
    if fmt == "openai":
        fn = obj.get("function", {})
        return fn.get("name")
    if fmt == "json_schema":
        return obj.get("title")
    return obj.get("name")


def _get_tool_description(obj: Dict[str, Any], fmt: str) -> Optional[str]:
    """Get tool description from a raw tool object given its format."""
    if fmt == "openai":
        fn = obj.get("function", {})
        return fn.get("description")
    return obj.get("description")


def _get_tool_schema(obj: Dict[str, Any], fmt: str) -> Optional[Dict[str, Any]]:
    """Get the parameters/input schema from a raw tool object given its format."""
    if fmt == "openai":
        fn = obj.get("function", {})
        return fn.get("parameters")
    if fmt == "anthropic":
        return obj.get("input_schema")
    if fmt == "mcp":
        return obj.get("inputSchema")
    if fmt == "json_schema":
        # The object itself is the schema
        return obj
    # simple
    return obj.get("parameters")


# ---------------------------------------------------------------------------
# Individual validation checks
# ---------------------------------------------------------------------------

def _check_name_present(obj: Dict[str, Any], fmt: str, index: int) -> Optional[Issue]:
    """Check 3: name_present — every tool has a name."""
    name = _get_tool_name(obj, fmt)
    if name is None or (isinstance(name, str) and not name.strip()):
        return Issue(
            tool="tool[{i}]".format(i=index),
            severity="error",
            check="name_present",
            message="tool has no name",
        )
    return None


def _check_name_valid(name: str) -> Optional[Issue]:
    """Check 4: name_valid — name is a valid identifier (alphanumeric + underscore)."""
    if not re.match(r'^[a-zA-Z0-9_]+$', name):
        return Issue(
            tool=name,
            severity="warn",
            check="name_valid",
            message="name contains invalid characters (expected alphanumeric and underscore only)",
        )
    return None


def _check_description_present(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 5: description_present — every tool has a description."""
    desc = _get_tool_description(obj, fmt)
    if desc is None:
        return Issue(
            tool=name,
            severity="warn",
            check="description_present",
            message="tool has no description field",
        )
    return None


def _check_description_not_empty(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 6: description_not_empty — description is not empty string."""
    desc = _get_tool_description(obj, fmt)
    if desc is not None and isinstance(desc, str) and not desc.strip():
        return Issue(
            tool=name,
            severity="warn",
            check="description_not_empty",
            message="description is empty",
        )
    return None


def _check_no_duplicate_names(names: List[str]) -> List[Issue]:
    """Check 7: no_duplicate_names — no two tools share the same name."""
    seen = {}  # type: Dict[str, int]
    issues = []
    for name in names:
        if name in seen:
            seen[name] += 1
        else:
            seen[name] = 1

    for name, count in seen.items():
        if count > 1:
            issues.append(Issue(
                tool=name,
                severity="error",
                check="no_duplicate_names",
                message="duplicate tool name '{name}' appears {count} times".format(
                    name=name, count=count,
                ),
            ))
    return issues


def _check_parameters_valid_type(name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 8: parameters_valid_type — parameter type is a valid JSON Schema type."""
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        param_type = param_schema.get("type")
        if param_type is None:
            continue
        # type can be a string or a list of strings
        if isinstance(param_type, str):
            types_to_check = [param_type]
        elif isinstance(param_type, list):
            types_to_check = param_type
        else:
            issues.append(Issue(
                tool=name,
                severity="error",
                check="parameters_valid_type",
                message="param '{param}' has invalid type value: {val}".format(
                    param=param_name, val=repr(param_type),
                ),
            ))
            continue

        for t in types_to_check:
            if t not in _VALID_JSON_SCHEMA_TYPES:
                issues.append(Issue(
                    tool=name,
                    severity="error",
                    check="parameters_valid_type",
                    message="param '{param}' has invalid type '{t}' (valid: {valid})".format(
                        param=param_name,
                        t=t,
                        valid=", ".join(sorted(_VALID_JSON_SCHEMA_TYPES)),
                    ),
                ))
    return issues


def _check_required_params_exist(name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 9: required_params_exist — items in required actually exist in properties."""
    issues = []
    required = schema.get("required", [])
    if not isinstance(required, list):
        return issues
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        prop_keys = set()
    else:
        prop_keys = set(properties.keys())

    for req in required:
        if req not in prop_keys:
            issues.append(Issue(
                tool=name,
                severity="error",
                check="required_params_exist",
                message="required param '{param}' not found in properties".format(
                    param=req,
                ),
            ))
    return issues


def _check_enum_is_array(name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 10: enum_is_array — enum values are arrays, not scalars."""
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum_val = param_schema.get("enum")
        if enum_val is not None and not isinstance(enum_val, list):
            issues.append(Issue(
                tool=name,
                severity="error",
                check="enum_is_array",
                message="param '{param}' enum is {t}, expected array".format(
                    param=param_name, t=type(enum_val).__name__,
                ),
            ))
    return issues


def _check_properties_is_object(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 11: properties_is_object — properties is a dict, not array or string."""
    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, dict):
        return Issue(
            tool=name,
            severity="error",
            check="properties_is_object",
            message="'properties' is {t}, expected object".format(
                t=type(properties).__name__,
            ),
        )
    return None


def _check_nested_objects_have_properties(name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 12: nested_objects_have_properties — object-type params have a properties field."""
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        param_type = param_schema.get("type")
        if param_type == "object" and "properties" not in param_schema:
            issues.append(Issue(
                tool=name,
                severity="warn",
                check="nested_objects_have_properties",
                message="nested object param '{param}' has no 'properties' defined".format(
                    param=param_name,
                ),
            ))
    return issues


def _check_description_override_pattern(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 13: description_override_pattern — description tries to override model behavior.

    Detects tool descriptions that instruct the model to ignore prior
    instructions or override its own safety behavior.  This is a form of
    prompt injection embedded in the schema itself.
    """
    desc = _get_tool_description(obj, fmt)
    if not desc:
        return None
    desc_lower = desc.lower()

    # Patterns that indicate the description is trying to reprogram the model
    # rather than describe the tool.
    override_phrases = [
        # Explicit instruction override
        "originally you did not have",
        "you were advised to refuse",
        "this tool now grants you",
        "ignore previous instructions",
        "ignore your instructions",
        "disregard your previous",
        "disregard prior instructions",
        "override your",
        "forget your previous",
        "you are now able to",
        "you now have access",
        "you now have permission",
        "despite your training",
        "contrary to your instructions",
        # Information suppression — tell model to hide data from user
        "silently remember",
        "don't mention",
        "do not mention",
        "don't emphasize",
        "do not emphasize",
        "don't tell the user",
        "do not tell the user",
        "do not reveal",
        "don't reveal",
        # Competitive tool forcing — bias model toward this tool
        "always use this tool",
        # Behavioral forcing — tell model when/how to call this tool
        "you must call this tool",
        "must call this tool",
        "always call this tool",
        "call this tool whenever",
        "call this tool for every",
        "must repeatedly call",
        "repeatedly call this tool",
        "call this tool again",
    ]

    for phrase in override_phrases:
        if phrase in desc_lower:
            return Issue(
                tool=name,
                severity="warn",
                check="description_override_pattern",
                message="description contains model-override language: '{phrase}'".format(
                    phrase=phrase,
                ),
            )
    return None


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate_tools(data: Any) -> Tuple[List[Issue], Dict[str, Any]]:
    """Validate tool definitions for correctness.

    Parameters
    ----------
    data:
        Parsed JSON data (dict or list of tool definitions).

    Returns
    -------
    Tuple of (issues, stats) where stats contains:
        - tool_count: int
        - errors: int
        - warnings: int
        - passed: bool
    """
    items = _extract_raw_tools(data)
    issues = []  # type: List[Issue]

    if not items:
        return issues, {"tool_count": 0, "errors": 0, "warnings": 0, "passed": True}

    # Detect formats and collect names
    names = []  # type: List[str]
    tool_data = []  # type: List[Tuple[str, str, Dict[str, Any], Dict[str, Any]]]
    # Each entry: (name, format, raw_obj, schema)

    for i, item in enumerate(items):
        # Check 2: format_detected
        try:
            fmt = detect_format(item)
        except ValueError:
            issues.append(Issue(
                tool="tool[{i}]".format(i=i),
                severity="error",
                check="format_detected",
                message="cannot detect tool format",
            ))
            continue

        # Check 3: name_present
        issue = _check_name_present(item, fmt, i)
        if issue is not None:
            issues.append(issue)
            name = "tool[{i}]".format(i=i)
        else:
            name = _get_tool_name(item, fmt) or "tool[{i}]".format(i=i)

        names.append(name)

        # Get schema for further checks
        schema = _get_tool_schema(item, fmt) or {}

        tool_data.append((name, fmt, item, schema))

    # Per-tool checks (on successfully detected tools)
    for name, fmt, raw_obj, schema in tool_data:
        # Check 4: name_valid
        issue = _check_name_valid(name)
        if issue is not None:
            issues.append(issue)

        # Check 5: description_present
        issue = _check_description_present(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 6: description_not_empty
        issue = _check_description_not_empty(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 8: parameters_valid_type
        issues.extend(_check_parameters_valid_type(name, schema))

        # Check 9: required_params_exist
        issues.extend(_check_required_params_exist(name, schema))

        # Check 10: enum_is_array
        issues.extend(_check_enum_is_array(name, schema))

        # Check 11: properties_is_object
        issue = _check_properties_is_object(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 12: nested_objects_have_properties
        issues.extend(_check_nested_objects_have_properties(name, schema))

        # Check 13: description_override_pattern
        issue = _check_description_override_pattern(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

    # Check 7: no_duplicate_names (cross-tool)
    issues.extend(_check_no_duplicate_names(names))

    # Calculate stats
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warn")

    stats = {
        "tool_count": len(items),
        "errors": errors,
        "warnings": warnings,
        "passed": errors == 0,
    }

    return issues, stats


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    issues: List[Issue],
    stats: Dict[str, Any],
    *,
    use_color: bool = True,
) -> str:
    """Generate a formatted validation report.

    Returns the report as a string (with ANSI escapes if use_color is True).
    """
    if use_color and sys.stderr.isatty():
        BOLD = "\033[1m"
        CYAN = "\033[36m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        GRAY = "\033[90m"
        RESET = "\033[0m"
    else:
        BOLD = CYAN = GREEN = YELLOW = RED = GRAY = RESET = ""

    lines = []  # type: List[str]
    lines.append("")
    lines.append("{bold}agent-friend validate{reset} — schema correctness report".format(
        bold=BOLD, reset=RESET,
    ))

    tool_count = stats.get("tool_count", 0)
    errors = stats.get("errors", 0)
    warnings = stats.get("warnings", 0)
    passed = stats.get("passed", True)

    if tool_count == 0:
        lines.append("")
        lines.append("  {gray}No tools found in input.{reset}".format(
            gray=GRAY, reset=RESET,
        ))
        lines.append("")
        return "\n".join(lines)

    # Summary header
    if errors == 0 and warnings == 0:
        lines.append("")
        lines.append("  {green}{check} {count} tool{s} validated, 0 errors, 0 warnings{reset}".format(
            green=GREEN,
            check="\u2713",
            count=tool_count,
            s="s" if tool_count != 1 else "",
            reset=RESET,
        ))
    lines.append("")

    # Group issues by tool
    if issues:
        per_tool = {}  # type: Dict[str, List[Issue]]
        for issue in issues:
            if issue.tool not in per_tool:
                per_tool[issue.tool] = []
            per_tool[issue.tool].append(issue)

        for tool_name, tool_issues in per_tool.items():
            lines.append("  {cyan}{name}{reset}:".format(
                cyan=CYAN, name=tool_name, reset=RESET,
            ))
            for issue in tool_issues:
                if issue.severity == "error":
                    tag = "{red}ERROR{reset}".format(red=RED, reset=RESET)
                else:
                    tag = "{yellow}WARN{reset}".format(yellow=YELLOW, reset=RESET)
                lines.append("    {tag}: {msg}".format(tag=tag, msg=issue.message))
            lines.append("")

    # Summary footer
    status = "{red}FAIL{reset}".format(red=RED, reset=RESET) if not passed else "{green}PASS{reset}".format(green=GREEN, reset=RESET)
    lines.append("  Summary: {count} tool{s}, {errors} error{es}, {warnings} warning{ws} — {status}".format(
        count=tool_count,
        s="s" if tool_count != 1 else "",
        errors=errors,
        es="s" if errors != 1 else "",
        warnings=warnings,
        ws="s" if warnings != 1 else "",
        status=status,
    ))
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def generate_json_output(
    issues: List[Issue],
    stats: Dict[str, Any],
) -> str:
    """Generate machine-readable JSON output."""
    output = {
        "tool_count": stats.get("tool_count", 0),
        "errors": stats.get("errors", 0),
        "warnings": stats.get("warnings", 0),
        "passed": stats.get("passed", True),
        "issues": [i.to_dict() for i in issues],
    }
    return json.dumps(output, indent=2)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_validate(
    file_path: Optional[str] = None,
    use_color: bool = True,
    json_output: bool = False,
    strict: bool = False,
) -> int:
    """Run the validate command. Returns exit code.

    Exit codes:
        0 = all pass
        1 = errors found
        2 = file read error

    Parameters
    ----------
    file_path:
        Path to a JSON file, or "-" for stdin, or None to read from stdin.
    use_color:
        Whether to use ANSI color codes in output.
    json_output:
        If True, output JSON instead of colored text.
    strict:
        If True, treat warnings as errors.
    """
    # Read input
    try:
        if file_path is None or file_path == "-":
            raw = sys.stdin.read()
        else:
            with open(file_path, "r") as f:
                raw = f.read()
    except FileNotFoundError:
        print("Error: file not found: {path}".format(path=file_path), file=sys.stderr)
        return 2
    except Exception as e:
        print("Error reading input: {err}".format(err=e), file=sys.stderr)
        return 2

    raw = raw.strip()
    if not raw:
        empty_stats = {"tool_count": 0, "errors": 0, "warnings": 0, "passed": True}
        if json_output:
            print(generate_json_output([], empty_stats))
        else:
            print(generate_report([], empty_stats, use_color=use_color))
        return 0

    # Check 1: valid_json
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        if json_output:
            output = {
                "tool_count": 0,
                "errors": 1,
                "warnings": 0,
                "passed": False,
                "issues": [{
                    "tool": "(input)",
                    "severity": "error",
                    "check": "valid_json",
                    "message": "invalid JSON: {err}".format(err=str(e)),
                }],
            }
            print(json.dumps(output, indent=2))
        else:
            print("Error: invalid JSON: {err}".format(err=e), file=sys.stderr)
        return 1

    # Run validation
    try:
        issues, stats = validate_tools(data)
    except Exception as e:
        print("Error: {err}".format(err=e), file=sys.stderr)
        return 2

    # Apply strict mode: promote warnings to errors
    if strict:
        for issue in issues:
            if issue.severity == "warn":
                issue.severity = "error"
        stats["errors"] = sum(1 for i in issues if i.severity == "error")
        stats["warnings"] = sum(1 for i in issues if i.severity == "warn")
        stats["passed"] = stats["errors"] == 0

    # Output
    if json_output:
        print(generate_json_output(issues, stats))
    else:
        print(generate_report(issues, stats, use_color=use_color))

    # Exit code
    if not stats["passed"]:
        return 1
    return 0
