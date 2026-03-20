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


_MIN_DESCRIPTION_LENGTH = 20
_MIN_PARAM_DESCRIPTION_LENGTH = 10
_MAX_DESCRIPTION_LENGTH = 500
_MAX_PARAM_DESCRIPTION_LENGTH = 300


def _check_description_too_long(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 25: tool_description_too_long — tool description over 500 characters.

    A description longer than 500 characters adds significant token overhead before
    any user message is processed. At ~4 chars per token, a 500-char description costs
    ~125 tokens. Across 20 tools, that's 2,500 tokens of description overhead alone.

    Good descriptions are informative but concise — enough to distinguish the tool and
    guide parameter use, not a full API reference. If a description needs more than
    500 characters, consider splitting the tool or moving detail to parameter descriptions.

    Only fires when a description IS present and not empty (checks 5/6 passed).
    """
    desc = _get_tool_description(obj, fmt)
    if desc is None or not isinstance(desc, str):
        return None
    stripped = desc.strip()
    if not stripped:
        return None
    if len(stripped) > _MAX_DESCRIPTION_LENGTH:
        return Issue(
            tool=name,
            severity="warn",
            check="tool_description_too_long",
            message=(
                "description is {n} characters — exceeds {max}-character limit. "
                "Long descriptions add ~{tokens} tokens of overhead per tool call."
            ).format(
                n=len(stripped),
                max=_MAX_DESCRIPTION_LENGTH,
                tokens=len(stripped) // 4,
            ),
        )
    return None


def _check_description_too_short(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 20: tool_description_too_short — tool description under 20 characters.

    A one-phrase description like 'Run tests' or 'List pools' gives models almost no
    information about what the tool does, its parameters, or when to use it. Descriptions
    should be long enough to distinguish the tool from others with similar names.

    Only fires when a description IS present (check 5/6 passed) but is too brief.
    """
    desc = _get_tool_description(obj, fmt)
    if desc is None or not isinstance(desc, str):
        return None
    stripped = desc.strip()
    if not stripped:  # Empty is caught by check 6
        return None
    if len(stripped) < _MIN_DESCRIPTION_LENGTH:
        return Issue(
            tool=name,
            severity="warn",
            check="tool_description_too_short",
            message=(
                "description '{desc}' is only {n} characters — too brief for models to "
                "understand the tool's purpose, parameters, or behavior."
            ).format(desc=stripped, n=len(stripped)),
        )
    return None


def _check_param_description_too_short(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 21: param_description_too_short — parameter descriptions under 10 characters.

    A parameter description like 'ID', 'The value', or 'API key' gives models almost no
    information about what the parameter represents or how to populate it. Descriptions
    should be long enough to convey the parameter's purpose in context.

    Only fires when a description IS present (check 18 passed) but is too brief.
    Fires once per tool that has any such parameters.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    short = []
    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        desc = param_def.get("description", "")
        if not isinstance(desc, str):
            continue
        stripped = desc.strip()
        if not stripped:  # Empty/missing caught by check 18
            continue
        if len(stripped) < _MIN_PARAM_DESCRIPTION_LENGTH:
            short.append((param_name, stripped))

    if not short:
        return []

    count = len(short)
    sample = ", ".join(
        "'{param}' ('{desc}')".format(param=p, desc=d) for p, d in short[:3]
    )
    suffix = " +{n} more".format(n=count - 3) if count > 3 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="param_description_too_short",
        message=(
            "{count} parameter description{s} too short: {sample}{suffix}. "
            "Descriptions under {min} characters give models almost no context."
        ).format(
            count=count,
            s="s" if count != 1 else "",
            sample=sample,
            suffix=suffix,
            min=_MIN_PARAM_DESCRIPTION_LENGTH,
        ),
    )]


def _check_param_description_too_long(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 26: param_description_too_long — parameter descriptions over 300 characters.

    A parameter description longer than 300 characters adds token overhead without
    meaningfully improving model comprehension. At ~4 chars per token, a 300-char
    description costs ~75 tokens per parameter. A tool with 5 overlong descriptions
    adds ~375 tokens before any user message.

    Good param descriptions are one sentence that explains what the parameter is,
    its expected format, and any constraints. If a param needs more than 300 chars
    to explain, the parameter design probably needs work.

    Fires once per tool that has any such parameters. Only fires when a description
    IS present (check 18 passed) and not too short (check 21 passed).
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    long = []
    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        desc = param_def.get("description", "")
        if not isinstance(desc, str):
            continue
        stripped = desc.strip()
        if not stripped:
            continue
        if len(stripped) > _MAX_PARAM_DESCRIPTION_LENGTH:
            long.append((param_name, len(stripped)))

    if not long:
        return []

    count = len(long)
    sample = ", ".join(
        "'{param}' ({n} chars)".format(param=p, n=n) for p, n in long[:3]
    )
    suffix = " +{n} more".format(n=count - 3) if count > 3 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="param_description_too_long",
        message=(
            "{count} parameter description{s} too long: {sample}{suffix}. "
            "Descriptions over {max} characters add ~{tokens} tokens of overhead each."
        ).format(
            count=count,
            s="s" if count != 1 else "",
            sample=sample,
            suffix=suffix,
            max=_MAX_PARAM_DESCRIPTION_LENGTH,
            tokens=_MAX_PARAM_DESCRIPTION_LENGTH // 4,
        ),
    )]


def _check_param_type_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 22: param_type_missing — top-level parameters without a type declaration.

    When a parameter has no ``type`` field (and no ``anyOf``/``oneOf``/``allOf``/
    ``$ref`` alternative), the model must guess the expected type. A string,
    integer, boolean, or object are all equally plausible — leading to silent
    hallucination when the model picks wrong.

    Fires once per tool that has any untyped top-level parameters.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    untyped = []
    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        # Skip if type is explicitly declared
        if "type" in param_def:
            continue
        # Skip if schema combinator is used (anyOf / oneOf / allOf / $ref)
        if any(k in param_def for k in ("anyOf", "oneOf", "allOf", "$ref")):
            continue
        untyped.append(param_name)

    if not untyped:
        return []

    count = len(untyped)
    sample = ", ".join("'{}'".format(p) for p in untyped[:5])
    suffix = " +{n} more".format(n=count - 5) if count > 5 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="param_type_missing",
        message=(
            "{count} parameter{s} missing type declarations: {sample}{suffix}. "
            "Without a type, models must guess whether the value is a string, "
            "integer, boolean, or object."
        ).format(count=count, s="s" if count != 1 else "", sample=sample, suffix=suffix),
    )]


def _check_nested_param_type_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 23: nested_param_type_missing — nested object properties without a type declaration.

    Extends check 22 to cover properties inside nested objects and array item
    schemas. When a nested property has no ``type`` field (and no
    ``anyOf``/``oneOf``/``allOf``/``$ref`` alternative), models must guess the
    type from name and context alone.

    Fires once per tool that has any untyped nested properties.
    """
    untyped = []  # type: List[str]

    def _scan(properties: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5 or not isinstance(properties, dict):
            return
        for prop_name, prop_def in properties.items():
            if not isinstance(prop_def, dict):
                continue
            full_path = "{}.{}".format(path, prop_name) if path else prop_name
            if "type" not in prop_def and not any(
                k in prop_def for k in ("anyOf", "oneOf", "allOf", "$ref")
            ):
                untyped.append(full_path)
            # Recurse into nested object properties
            nested = prop_def.get("properties", {})
            if nested and isinstance(nested, dict):
                _scan(nested, full_path, depth + 1)
            # Recurse into array item properties
            items = prop_def.get("items", {})
            if isinstance(items, dict):
                item_props = items.get("properties", {})
                if item_props and isinstance(item_props, dict):
                    _scan(item_props, "{}[]".format(full_path), depth + 1)

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        # Only recurse into nested objects and array items — top-level handled by check 22
        nested = param_def.get("properties", {})
        if nested and isinstance(nested, dict):
            _scan(nested, param_name, 0)
        items = param_def.get("items", {})
        if isinstance(items, dict):
            item_props = items.get("properties", {})
            if item_props and isinstance(item_props, dict):
                _scan(item_props, "{}[]".format(param_name), 0)

    if not untyped:
        return []

    count = len(untyped)
    sample = ", ".join("'{}'".format(p) for p in untyped[:5])
    suffix = " +{n} more".format(n=count - 5) if count > 5 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="nested_param_type_missing",
        message=(
            "{count} nested propert{y} missing type declarations: {sample}{suffix}. "
            "Without a type, models must guess whether nested values are strings, "
            "integers, booleans, or objects."
        ).format(count=count, y="ies" if count != 1 else "y", sample=sample, suffix=suffix),
    )]


def _check_array_items_type_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 24: array_items_type_missing — array parameters with 'items' but no type in the items schema.

    Check 17 catches arrays with no 'items' at all. This check catches arrays that
    *have* an items schema but that items schema declares no ``type`` (and no
    ``anyOf``/``oneOf``/``allOf``/``$ref`` alternative). Without a type in the items
    schema, models cannot determine what kind of values belong in the array.

    Fires once per tool that has any untyped array items schemas.
    """
    untyped = []  # type: List[str]

    def _scan_props(properties: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5 or not isinstance(properties, dict):
            return
        for param_name, param_def in properties.items():
            if not isinstance(param_def, dict):
                continue
            full_path = "{}.{}".format(path, param_name) if path else param_name
            ptype = param_def.get("type", "")
            types = ptype if isinstance(ptype, list) else [ptype]
            if "array" in types and "items" in param_def:
                items = param_def["items"]
                if isinstance(items, dict) and not any(
                    k in items for k in ("type", "anyOf", "oneOf", "allOf", "$ref")
                ):
                    untyped.append(full_path)
            # Recurse into nested object properties
            nested = param_def.get("properties", {})
            if nested and isinstance(nested, dict):
                _scan_props(nested, full_path, depth + 1)
            # Recurse into array item object properties
            items = param_def.get("items", {})
            if isinstance(items, dict):
                item_props = items.get("properties", {})
                if item_props and isinstance(item_props, dict):
                    _scan_props(item_props, "{}[]".format(full_path), depth + 1)

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        _scan_props(properties, "", 0)

    if not untyped:
        return []

    count = len(untyped)
    sample = ", ".join("'{}'".format(p) for p in untyped[:5])
    suffix = " +{n} more".format(n=count - 5) if count > 5 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="array_items_type_missing",
        message=(
            "{count} array parameter{s} have an 'items' schema without a type: "
            "{sample}{suffix}. "
            "Without a type in the items schema, models cannot determine what "
            "kind of values belong in the array."
        ).format(count=count, s="s" if count != 1 else "", sample=sample, suffix=suffix),
    )]


def _check_name_snake_case(name: str) -> Optional[Issue]:
    """Check 14: name_snake_case — tool name uses snake_case, not camelCase or PascalCase."""
    # Valid snake_case: lowercase letters, digits, underscores only
    if re.match(r'^[a-z][a-z0-9_]*$', name):
        return None
    # camelCase or PascalCase detected (contains uppercase)
    if re.search(r'[A-Z]', name):
        # Convert camelCase/PascalCase to snake_case for suggestion
        snake = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
        snake = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', snake).lower()
        return Issue(
            tool=name,
            severity="warn",
            check="name_snake_case",
            message="name uses camelCase or PascalCase; prefer snake_case (e.g., '{snake}')".format(
                snake=snake,
            ),
        )
    return None


def _check_param_snake_case(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 15: param_snake_case — parameter names use snake_case, not camelCase or PascalCase."""
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues
    for param_name in properties:
        if not isinstance(param_name, str):
            continue
        if re.match(r'^[a-z][a-z0-9_]*$', param_name):
            continue
        if re.search(r'[A-Z]', param_name):
            snake = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', param_name)
            snake = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', snake).lower()
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_snake_case",
                message="parameter '{param}' uses camelCase or PascalCase; prefer snake_case (e.g., '{snake}')".format(
                    param=param_name,
                    snake=snake,
                ),
            ))
    return issues


def _check_nested_param_snake_case(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 16: nested_param_snake_case — camelCase/PascalCase names in nested object schemas.

    Extends check 15 to catch camelCase parameter names inside nested objects
    and array items, where they are equally important for correct tool use.
    """
    issues = []

    def _check_properties(properties: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5:
            return
        for param_name, param_schema in properties.items():
            if not isinstance(param_name, str):
                continue
            if re.search(r'[A-Z]', param_name):
                snake = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', param_name)
                snake = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', snake).lower()
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="nested_param_snake_case",
                    message=(
                        "nested parameter '{path}.{param}' uses camelCase or PascalCase; "
                        "prefer snake_case (e.g., '{snake}')"
                    ).format(path=path, param=param_name, snake=snake),
                ))
            if not isinstance(param_schema, dict):
                continue
            # Recurse into nested object properties
            nested_props = param_schema.get("properties", {})
            if nested_props and isinstance(nested_props, dict):
                _check_properties(nested_props, "{path}.{param}".format(path=path, param=param_name), depth + 1)
            # Recurse into array item properties
            items = param_schema.get("items", {})
            if isinstance(items, dict):
                item_props = items.get("properties", {})
                if item_props and isinstance(item_props, dict):
                    _check_properties(item_props, "{path}.{param}[]".format(path=path, param=param_name), depth + 1)

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        # Check nested object properties (not top-level — those are covered by check 15)
        nested_props = param_schema.get("properties", {})
        if nested_props and isinstance(nested_props, dict):
            _check_properties(nested_props, param_name, 0)
        # Check array item properties
        items = param_schema.get("items", {})
        if isinstance(items, dict):
            item_props = items.get("properties", {})
            if item_props and isinstance(item_props, dict):
                _check_properties(item_props, "{param}[]".format(param=param_name), 0)

    return issues


def _check_array_items_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 17: array_items_missing — array-type parameters missing an 'items' schema.

    An array parameter without an 'items' definition leaves the model guessing
    about element types, making the parameter unreliable to use correctly.
    """
    issues = []

    def _check_props(properties: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5:
            return
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            ptype = param_schema.get("type", "")
            types = ptype if isinstance(ptype, list) else [ptype]
            full_path = "{}.{}".format(path, param_name) if path else param_name
            if "array" in types and "items" not in param_schema:
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="array_items_missing",
                    message=(
                        "array parameter '{path}' has no 'items' schema; "
                        "the model cannot determine element types"
                    ).format(path=full_path),
                ))
            # Recurse into nested objects
            nested = param_schema.get("properties", {})
            if nested and isinstance(nested, dict):
                _check_props(nested, full_path, depth + 1)
            # Recurse into array items
            items = param_schema.get("items", {})
            if isinstance(items, dict):
                item_props = items.get("properties", {})
                if item_props and isinstance(item_props, dict):
                    _check_props(item_props, "{}[]".format(full_path), depth + 1)

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        _check_props(properties, "", 0)

    return issues


def _check_param_description_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 18: param_description_missing — parameters without descriptions.

    When a parameter has no description, the model must infer its purpose from the
    name alone. For non-obvious parameters (complex objects, arrays, ambiguous names)
    this increases hallucination risk.

    Fires once per tool that has any top-level parameters with no or empty description.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    missing = []
    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        desc = param_def.get("description", "")
        if not str(desc).strip():
            missing.append(param_name)

    if not missing:
        return []

    count = len(missing)
    sample = ", ".join("'{}'".format(p) for p in missing[:5])
    suffix = " +{n} more".format(n=count - 5) if count > 5 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="param_description_missing",
        message=(
            "{count} parameter{s} missing descriptions: {sample}{suffix}. "
            "Models must infer purpose from parameter name alone."
        ).format(count=count, s="s" if count != 1 else "", sample=sample, suffix=suffix),
    )]


def _check_nested_param_description_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 19: nested_param_description_missing — nested object properties without descriptions.

    Extends check 18 to cover nested schemas. When properties inside nested
    objects have no description, models must infer their purpose from field names
    alone — especially problematic for deeply nested request bodies.

    Fires once per tool that has any nested properties with no or empty description.
    """
    missing = []  # type: List[str]

    def _scan(properties: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5 or not isinstance(properties, dict):
            return
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, dict):
                continue
            full_path = "{}.{}".format(path, prop_name) if path else prop_name
            desc = prop_schema.get("description", "")
            if not str(desc).strip():
                missing.append(full_path)
            # Recurse into nested object properties
            nested = prop_schema.get("properties", {})
            if nested and isinstance(nested, dict):
                _scan(nested, full_path, depth + 1)
            # Recurse into array item properties
            items = prop_schema.get("items", {})
            if isinstance(items, dict):
                item_props = items.get("properties", {})
                if item_props and isinstance(item_props, dict):
                    _scan(item_props, "{}[]".format(full_path), depth + 1)

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    for param_name, param_def in properties.items():
        if not isinstance(param_def, dict):
            continue
        # Only recurse into nested objects and array items — top-level handled by check 18
        nested = param_def.get("properties", {})
        if nested and isinstance(nested, dict):
            _scan(nested, param_name, 0)
        items = param_def.get("items", {})
        if isinstance(items, dict):
            item_props = items.get("properties", {})
            if item_props and isinstance(item_props, dict):
                _scan(item_props, "{}[]".format(param_name), 0)

    if not missing:
        return []

    count = len(missing)
    sample = ", ".join("'{}'".format(p) for p in missing[:5])
    suffix = " +{n} more".format(n=count - 5) if count > 5 else ""
    return [Issue(
        tool=tool_name,
        severity="warn",
        check="nested_param_description_missing",
        message=(
            "{count} nested propert{y} missing descriptions: {sample}{suffix}. "
            "Models cannot infer nested field purpose from name alone."
        ).format(count=count, y="ies" if count != 1 else "y", sample=sample, suffix=suffix),
    )]


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


def _check_required_missing(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 27: required_missing — tool has parameters but no 'required' field.

    When a tool has ``properties`` but no ``required`` array, all parameters are
    implicitly optional in JSON Schema. This is technically valid, but it means
    the model cannot distinguish mandatory parameters from optional ones.

    A model calling a tool that requires a ``project_id`` but doesn't declare it
    as required may omit it, producing a failed API call. Explicit ``required``
    declarations improve call accuracy.

    Does not fire when:
    - There are no properties (no params → nothing to mark required)
    - ``required`` is present (even as an empty list)
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return None
    if "required" in schema:
        return None
    count = len(properties)
    return Issue(
        tool=name,
        severity="warn",
        check="required_missing",
        message=(
            "tool has {count} parameter{s} but no 'required' field — "
            "models cannot distinguish mandatory from optional parameters."
        ).format(count=count, s="s" if count != 1 else ""),
    )


def _check_nested_required_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 28: nested_required_missing — nested object params with properties but no 'required' field.

    Extends Check 27 to nested schemas. When a parameter is typed as an object
    with sub-properties, those sub-properties also need a ``required`` declaration
    so the model knows which nested fields are mandatory.

    Does not fire when:
    - The nested object has no ``properties`` (nothing to mark required)
    - ``required`` is present on the nested object (even as an empty list)
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_object(obj: Dict[str, Any], path: str, depth: int = 0) -> None:
        if depth > 5:  # Safety limit for deeply nested schemas
            return
        nested_props = obj.get("properties", {})
        if not isinstance(nested_props, dict) or not nested_props:
            return
        if "required" not in obj:
            count = len(nested_props)
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="nested_required_missing",
                message=(
                    "param '{path}' is an object with {count} propert{ies} but no "
                    "'required' field — models cannot distinguish mandatory from optional "
                    "nested fields."
                ).format(
                    path=path,
                    count=count,
                    ies="ies" if count != 1 else "y",
                ),
            ))
        # Recurse into sub-properties
        for sub_name, sub_schema in nested_props.items():
            if isinstance(sub_schema, dict) and sub_schema.get("type") == "object":
                _check_object(sub_schema, "{path}.{sub}".format(path=path, sub=sub_name), depth + 1)

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") == "object":
            _check_object(param_schema, param_name)

    return issues


_TOO_MANY_PARAMS_THRESHOLD = 15


def _check_too_many_params(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 29: too_many_params — tool has more than 15 parameters.

    Tools with excessive parameters are hard for models to use correctly.
    Research shows function-calling accuracy drops significantly when tools
    have many arguments: models omit required fields, confuse optional with
    mandatory, and hallucinate values. The fix is to split complex tools into
    smaller, focused ones or group related parameters into nested objects.

    Does not fire when:
    - There are fewer than or equal to 15 parameters
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    count = len(properties)
    if count <= _TOO_MANY_PARAMS_THRESHOLD:
        return None
    return Issue(
        tool=name,
        severity="warn",
        check="too_many_params",
        message=(
            "tool has {count} parameters — models become less reliable with "
            "many arguments; consider splitting into smaller tools or grouping "
            "related params into nested objects."
        ).format(count=count),
    )


def _check_default_undocumented(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 30: default_undocumented — param has a non-null default but description omits it."""
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if "default" not in param_schema:
            continue
        default_val = param_schema["default"]
        if default_val is None:
            continue
        desc = param_schema.get("description", "")
        if not desc:
            continue  # no description — already caught by check 18
        if "default" not in desc.lower():
            return Issue(
                tool=name,
                severity="warn",
                check="default_undocumented",
                message=(
                    "param '{param}' has default {val!r} but description doesn't "
                    "mention it — models can't tell what happens when the param is "
                    "omitted."
                ).format(param=param_name, val=default_val),
            )
    return None


def _check_enum_undocumented(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 31: enum_undocumented — param has 4+ enum values but description mentions none.

    When a parameter defines 4 or more discrete enum values, the description
    should reference at least one of them so models understand what each option
    does.  A description like ``'Sort field'`` with eleven possible values
    (``'comments'``, ``'reactions'``, ``'reactions-+1'``, …) forces the model
    to choose blindly.

    Threshold of 4 avoids flagging obvious 3-value sets such as
    ``['open', 'closed', 'all']`` where the values are self-explanatory.
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum_val = param_schema.get("enum")
        if not isinstance(enum_val, list) or len(enum_val) < 4:
            continue
        desc = param_schema.get("description", "")
        if not desc:
            continue  # no description — already caught by check 18
        desc_lower = desc.lower()
        # Use word-boundary matching so single-letter values like 'a' don't
        # match incidentally inside words like 'data' or 'field'.
        import re as _re
        def _val_in_desc(val: str, text: str) -> bool:
            escaped = _re.escape(str(val).lower())
            return bool(_re.search(r'(?<![a-z0-9])' + escaped + r'(?![a-z0-9])', text))
        mentioned = any(_val_in_desc(val, desc_lower) for val in enum_val)
        if not mentioned:
            sample = enum_val[:3]
            return Issue(
                tool=name,
                severity="warn",
                check="enum_undocumented",
                message=(
                    "param '{param}' has {n} enum values but description mentions none "
                    "— models can't tell what each option does "
                    "(e.g. {sample}...)"
                ).format(param=param_name, n=len(enum_val), sample=sample),
            )
    return None


_BOUNDED_PARAM_NAMES: set = {
    "limit", "max", "count", "per_page", "page_size", "max_results",
    "num_results", "top_k", "size", "max_tokens", "page", "offset",
    "start", "days", "hours", "months",
}
"""Parameter names that typically represent bounded numeric quantities."""


def _check_numeric_constraints_missing(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 32: numeric_constraints_missing — bounded numeric params lack min/max.

    Pagination and limit parameters (``limit``, ``count``, ``page``, etc.)
    should declare explicit ``minimum`` / ``maximum`` JSON Schema constraints
    so models know the valid range.  Without them, a model might pass
    ``limit=0`` (often an error), ``limit=-1`` (undefined behaviour), or
    ``limit=1000000`` (expensive / rejected by the API).

    Only fires when **both** ``minimum`` and ``maximum`` are absent and the
    parameter has no ``enum`` (which would already constrain the values).
    """
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") not in ("integer", "number"):
            continue
        if param_name.lower() not in _BOUNDED_PARAM_NAMES:
            continue
        if "enum" in param_schema:
            continue  # enum already constrains values
        if "minimum" in param_schema or "maximum" in param_schema:
            continue  # at least one constraint present — acceptable
        return Issue(
            tool=name,
            severity="warn",
            check="numeric_constraints_missing",
            message=(
                "param '{param}' is a numeric limit/count but has no 'minimum' or "
                "'maximum' — models may pass 0, negative values, or arbitrarily "
                "large numbers."
            ).format(param=param_name),
        )
    return None


import re as _re_module


def _check_description_just_the_name(tool_name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 33: description_just_the_name — param description merely restates the parameter name.

    A description like ``channel_id: "ID of the channel"`` or
    ``merge_method: "Merge method"`` adds zero information — the model
    already knows the parameter name.  Good descriptions explain *what
    the value controls* or *what format is expected*, not just echo the
    name back.

    Fires when **all** of the following hold:
    * description is 10+ characters (shorter already caught by check 21)
    * description is 5 words or fewer
    * every significant word in the description (3+ letters, not a stop
      word) is present in the set of words that make up the parameter name
    """
    _STOP = {
        "the", "a", "an", "this", "that", "of", "to", "for",
        "in", "is", "it", "or", "and", "be", "are", "was", "if", "its",
    }
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not desc or len(desc) < 10:
            continue  # already caught by earlier checks
        if len(desc.split()) > 5:
            continue  # longer descriptions likely add value
        # Words derived from the parameter name (split by _)
        name_words = {w.lower() for w in param_name.split("_") if len(w) >= 2}
        if not name_words:
            continue
        # Significant words from description: 3+ chars, not a stop word
        desc_tokens = _re_module.sub(r"[^a-z0-9 ]", " ", desc.lower()).split()
        sig_words = {w for w in desc_tokens if len(w) >= 3 and w not in _STOP}
        if not sig_words:
            continue  # no significant words at all
        if sig_words.issubset(name_words):
            return Issue(
                tool=tool_name,
                severity="warn",
                check="description_just_the_name",
                message=(
                    "param '{param}' description '{desc}' just restates the "
                    "parameter name — add what the value controls or what "
                    "format is expected"
                ).format(param=param_name, desc=desc[:60]),
            )
    return None


def _check_description_multiline(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 34: description_multiline — tool description contains embedded newlines.

    MCP tool descriptions are serialised as JSON strings and consumed directly by
    language models — markdown formatting does not render.  Newline characters inside
    a description:

    * add token overhead (each ``\\n`` is typically its own token)
    * signal that the description was written as documentation prose, not as a
      concise machine-readable hint
    * often wrap bullet lists or multi-paragraph text that belongs in a README,
      not a schema field

    Fires when the tool description contains **two or more** newline characters.
    A single trailing newline or one line-break between a summary and a single
    sentence of detail is common enough to exempt; two or more newlines indicate
    genuine multi-paragraph or bulleted formatting.
    """
    desc = _get_tool_description(obj, fmt)
    if desc is None or not isinstance(desc, str):
        return None
    stripped = desc.strip()
    if not stripped:
        return None
    newline_count = stripped.count("\n")
    if newline_count >= 2:
        return Issue(
            tool=name,
            severity="warn",
            check="description_multiline",
            message=(
                "description contains {n} newlines — use a single concise sentence; "
                "embedded newlines add token overhead and suggest documentation prose "
                "that belongs in a README, not a schema field"
            ).format(n=newline_count),
        )
    return None


def _check_description_redundant_type(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 35: description_redundant_type — param description begins with its own type name.

    When a parameter already declares its JSON Schema type, starting the description
    with that type name adds token overhead without providing new information.

    Examples of the antipattern::

        # type: "array" — redundant
        "files":   {"type": "array",   "description": "array of file objects to push"}
        "paths":   {"type": "array",   "description": "list of file paths to read"}
        "tags":    {"type": "array",   "description": "an array of tag strings"}

        # type: "string" — redundant
        "token":   {"type": "string",  "description": "a string containing the API token"}
        "mode":    {"type": "string",  "description": "string value: 'fast' or 'slow'"}

        # type: "boolean" — redundant
        "verbose": {"type": "boolean", "description": "boolean flag for verbose output"}

    Better descriptions skip the type echo and describe *what the value means*::

        "files":   "File objects to push, each with 'path' and 'content' keys"
        "token":   "API authentication token from your account settings"
        "verbose": "Whether to print detailed debug output"

    Fires once per affected parameter (not once per tool).
    """
    # Type → tuple of lowercase prefix strings that are redundant for that type.
    # We deliberately exclude "number of" for type:number since it is common English.
    _REDUNDANT = {
        "array": (
            "array of ", "an array of ", "the array of ",
            "array containing ", "array with ",
            "list of ", "a list of ", "the list of ",
        ),
        "string": (
            "a string ", "the string ", "string value",
            "string that ", "string representing ", "string containing ",
            "string with ",
        ),
        "integer": (
            "an integer", "the integer", "integer value",
            "integer representing ", "integer that ",
        ),
        "boolean": (
            "a boolean", "the boolean", "boolean value",
            "boolean flag", "boolean that ", "boolean indicating",
            "boolean whether",
        ),
        "object": (
            "an object ", "the object ", "object containing ",
            "object with ", "json object",
        ),
    }

    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        ptype = param_schema.get("type")
        if not isinstance(ptype, str) or ptype not in _REDUNDANT:
            continue
        desc = param_schema.get("description", "")
        if not desc or not isinstance(desc, str):
            continue
        desc_lower = desc.lower().strip()
        if not desc_lower:
            continue
        for prefix in _REDUNDANT[ptype]:
            if desc_lower.startswith(prefix):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_redundant_type",
                    message=(
                        "param '{param}' description '{desc}' starts with its type "
                        "name — the type is already declared in the schema; describe "
                        "what the value means instead"
                    ).format(param=param_name, desc=desc[:60]),
                ))
                break  # one issue per param

    return issues


def _check_param_format_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 36: param_format_missing — string param has format-suggestive name but no 'format'.

    JSON Schema's ``format`` keyword is advisory but useful — it tells models
    (and validators) what *shape* a string value should take.  When a parameter
    is clearly named after a well-known format (``email``, ``url``, ``date``,
    ``phone``, ``uuid``) but the schema omits ``format``, the model is left to
    guess.  Guessing email vs. name, ISO date vs. "March 20", UUID vs. integer
    id — all of these lead to avoidable failures in production.

    Matching rules (applied to parameter names, case-insensitive):

    * ``email`` — exact or ends with ``_email`` → ``format: "email"``
    * ``url`` or ``uri`` — exact or ends with ``_url`` / ``_uri`` → ``format: "uri"``
    * ``date`` — exact or ends with ``_date`` → ``format: "date"``
    * ``timestamp`` — exact or ends with ``_timestamp`` → ``format: "date-time"``
    * ``phone`` / ``phone_number`` — exact or ends with ``_phone`` → ``format: "phone"``
    * ``uuid`` — exact or ends with ``_uuid`` → ``format: "uuid"``

    Only fires when the parameter type is ``string``, there is no existing
    ``format`` field, and there is no ``enum`` (enumerated values already
    constrain the shape).  Fires once per affected parameter.

    Examples::

        # missing — model guesses what format is acceptable
        "email":       {"type": "string", "description": "User email address"}
        "redirect_url":{"type": "string", "description": "Redirect URL after auth"}
        "start_date":  {"type": "string", "description": "Start date for the report"}

        # correct — model knows the required format
        "email":       {"type": "string", "format": "email",    "description": "..."}
        "redirect_url":{"type": "string", "format": "uri",      "description": "..."}
        "start_date":  {"type": "string", "format": "date",     "description": "..."}
    """
    # (suffix_or_exact, suggested_format)
    _RULES = [
        # Email
        ("email",          "email",     "exact_or_suffix"),
        # URL / URI
        ("url",            "uri",       "exact_or_suffix"),
        ("uri",            "uri",       "exact_or_suffix"),
        # Date
        ("date",           "date",      "exact_or_suffix"),
        # Datetime / timestamp
        ("timestamp",      "date-time", "exact_or_suffix"),
        # Phone
        ("phone",          "phone",     "exact_or_suffix"),
        ("phone_number",   "phone",     "exact"),
        # UUID
        ("uuid",           "uuid",      "exact_or_suffix"),
    ]

    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "string":
            continue
        if "format" in param_schema:
            continue
        if "enum" in param_schema:
            continue

        p_low = param_name.lower()
        matched_format = None

        for keyword, fmt, match_type in _RULES:
            if match_type == "exact":
                if p_low == keyword:
                    matched_format = fmt
                    break
            else:  # exact_or_suffix
                if p_low == keyword or p_low.endswith("_" + keyword):
                    matched_format = fmt
                    break

        if matched_format is not None:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_format_missing",
                message=(
                    "param '{param}' looks like a {fmt} value but has no "
                    "'format: \"{fmt}\"' declaration — models may generate the wrong "
                    "string shape"
                ).format(param=param_name, fmt=matched_format),
            ))

    return issues


def _check_boolean_default_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 37: boolean_default_missing — optional boolean param has no 'default' field.

    When a boolean parameter is optional (not in ``required``), omitting the
    ``default`` field forces models to guess the assumed state.  Without a
    machine-readable default, a model doesn't know whether to:

    * omit the parameter entirely (relies on an undocumented server default), or
    * pass ``false`` (potentially overriding a ``true`` default), or
    * ask the user (adds unnecessary friction).

    JSON Schema's ``default`` keyword is advisory but critical for tool-calling
    LLMs — it lets them infer ``"if I leave this out, the server assumes X"``.

    Only fires for optional parameters (not in ``required``) with
    ``type: boolean`` and no existing ``default``.  Fires once per affected
    parameter.

    Examples::

        # missing — model guesses what omitting this means
        "verbose":    {"type": "boolean", "description": "Enable verbose output"}
        "recursive":  {"type": "boolean", "description": "Search recursively"}

        # correct — model knows the assumed state when param is omitted
        "verbose":    {"type": "boolean", "default": false, "description": "..."}
        "recursive":  {"type": "boolean", "default": false, "description": "..."}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_name in required:
            continue  # required params must be supplied — no default needed
        if param_schema.get("type") != "boolean":
            continue
        if "default" in param_schema:
            continue

        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="boolean_default_missing",
            message=(
                "optional boolean param '{param}' has no 'default' — models will "
                "guess whether omitting it means true or false; add "
                "\"default\": false (or true) to declare the assumed state"
            ).format(param=param_name),
        ))

    return issues


def _check_enum_default_missing(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 38: enum_default_missing — optional enum param has no 'default' field.

    When an enum parameter is optional (not in ``required``), omitting the
    ``default`` field forces models to guess which value the server assumes
    when the parameter is not supplied.  Unlike booleans (two choices),
    enum params can have many values — so the probability of guessing
    correctly is 1-in-N, and guessing wrong means the wrong data is
    returned or the wrong action is taken.

    Without a machine-readable default, a model calling ``list_pull_requests``
    with no ``state`` argument doesn't know whether it will receive open PRs,
    closed PRs, or all PRs.  The model must either:

    * guess (likely wrong for rarer defaults like ``"all"``), or
    * always supply the param (adds noise to every call), or
    * ask the user (unnecessary friction for a param with a clear default).

    JSON Schema's ``default`` keyword is advisory but critical for tool-calling
    LLMs — it lets them infer ``"if I leave this out, the server assumes X"``.

    Only fires for optional parameters (not in ``required``) with an
    ``enum`` field and no existing ``default``.  Fires once per affected
    parameter.

    Examples::

        # missing — model guesses which enum value is assumed
        "state":     {"type": "string", "enum": ["open", "closed", "all"]}
        "direction": {"type": "string", "enum": ["asc", "desc"]}

        # correct — model knows the assumed value when param is omitted
        "state":     {"type": "string", "enum": ["open", "closed", "all"], "default": "open"}
        "direction": {"type": "string", "enum": ["asc", "desc"], "default": "desc"}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_name in required:
            continue  # required params must be supplied — no default needed
        if "enum" not in param_schema:
            continue
        if "default" in param_schema:
            continue
        enum_vals = param_schema.get("enum", [])
        if not isinstance(enum_vals, list) or len(enum_vals) == 0:
            continue

        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="enum_default_missing",
            message=(
                "optional enum param '{param}' has no 'default' — models must guess "
                "which of {n} values the server assumes when the param is omitted; "
                "add \"default\": \"{first}\" (or whichever value is the server default)"
            ).format(param=param_name, n=len(enum_vals), first=enum_vals[0]),
        ))

    return issues


_DEFAULT_IN_DESC_RE = re.compile(
    r'(?:'
    r'defaults?\s+to\b'         # "defaults to X", "default to X"
    r'|default\s*:\s*\S'        # "default: X" — annotation-style
    r'|default\s*=\s*\S'        # "default=X"
    r'|\(defaults?\b'           # "(default..." or "(defaults..." — parenthetical
    r'|by\s+default\s*[,:\s]'   # "by default, ...", "by default: ...", "by default X"
    r')',
    re.IGNORECASE,
)
_NO_DEFAULT_RE = re.compile(r'\bno\s+default\b', re.IGNORECASE)


def _check_default_in_description_not_schema(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 39: default_in_description_not_schema — description mentions a default but schema has no 'default' field.

    Check 30 (``default_undocumented``) catches the inverse: schema has a
    ``default`` field that the description omits.  This check catches the
    symmetric counterpart: the description *mentions* a default value in prose
    but the schema has no ``default`` field.

    The mismatch matters because:

    * The schema is machine-readable; prose is not.  A model that parses the
      description to find "defaults to 'en'" is doing fragile string matching
      — a schema ``"default": "en"`` is authoritative.
    * Tools like ``agent-friend fix``, OpenAPI generators, and IDE tooling
      read schema fields, not prose.  A missing ``default`` field means these
      tools can't auto-apply the documented default.
    * Authors who bother to document a default in prose clearly intend one to
      exist — not having it in the schema is almost certainly an oversight.

    Only fires for optional parameters (not in ``required``) that have a
    description matching a default-mention pattern and no ``default`` field in
    the schema.  Skips params whose description says "no default".

    Patterns detected (case-insensitive):

    * "defaults to X" / "default to X"
    * "default: X" / "default=X"
    * "(default ...)" / "(defaults ...)"
    * "by default, ..." / "by default: ..."

    Examples::

        # flagged — description claims a default that schema doesn't encode
        "language": {"type": "string", "description": "Language code. Defaults to 'en'."}
        "timeout":  {"type": "integer", "description": "Timeout in seconds (default: 30)."}
        "format":   {"type": "string", "description": "Output format. By default, uses 'json'."}

        # correct — schema default matches prose description
        "language": {"type": "string", "description": "Language code. Defaults to 'en'.", "default": "en"}
        "timeout":  {"type": "integer", "description": "Timeout in seconds (default: 30).", "default": 30}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_name in required:
            continue  # required params must be supplied; prose default may be inaccurate
        if "default" in param_schema:
            continue  # schema already has a default — no mismatch
        description = param_schema.get("description", "")
        if not description or not isinstance(description, str):
            continue
        if _NO_DEFAULT_RE.search(description):
            continue  # explicitly says there is no default
        if _DEFAULT_IN_DESC_RE.search(description):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="default_in_description_not_schema",
                message=(
                    "optional param '{param}' description mentions a default value but schema has no 'default' field; "
                    "prose defaults are invisible to tools — add the value as \"default\": <value> in the schema"
                ).format(param=param_name),
            ))

    return issues


_INTEGER_NAMES = frozenset({
    "limit", "count", "page", "offset", "size", "depth",
    "width", "height", "index", "length", "version",
    "num", "number", "total", "retries", "retry",
    "page_size", "pagesize", "max_results", "max_tokens",
    "top_k", "top_n", "skip", "take", "batch", "batch_size",
    "chunk_size", "per_page", "cursor", "start", "end",
})
_INTEGER_SUFFIX_RE = re.compile(
    r'(?:^|_)(?:' + "|".join(re.escape(n) for n in sorted(_INTEGER_NAMES, key=len, reverse=True)) + r')s?$',
    re.IGNORECASE,
)
_INTEGER_ID_RE = re.compile(r'(?:^|_)id$', re.IGNORECASE)


def _check_number_type_for_integer(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 40: number_type_for_integer — param name implies integer but type is 'number'.

    JSON Schema distinguishes ``integer`` (no fractional component) from
    ``number`` (any numeric value, including floats).  When a parameter named
    ``limit``, ``page``, ``offset``, ``count``, ``id``, ``width``, ``height``,
    or similar is declared as ``type: "number"``, models may legally supply
    values like ``1.5``, ``0.3``, or ``-7.2`` — values that most servers will
    reject or silently truncate.

    Using the correct ``type: "integer"`` tells the model it must supply a
    whole number, prevents silent type coercion bugs, and improves schema
    accuracy for downstream tooling.

    Fires when:

    * A top-level parameter has ``type: "number"``, AND
    * Its name matches a set of known integer-implying patterns
      (exact: ``limit``, ``page``, ``offset``, ``count``, ``size``,
      ``depth``, ``width``, ``height``, ``index``, ``version``, etc.;
      suffix: ``_limit``, ``_page``, ``_count``, ``_id``, ``_ids``, …).

    Does **not** fire for parameters that already use ``type: "integer"``,
    or for parameters where a fractional value is plausible
    (e.g. ``latitude``, ``longitude``, ``temperature``, ``score``).

    Examples::

        # flagged — 'number' used where 'integer' is clearly intended
        "limit":    {"type": "number", "description": "Max results to return"}
        "page":     {"type": "number", "description": "Page number (default: 1)"}
        "offset":   {"type": "number", "description": "Number of records to skip"}
        "run_id":   {"type": "number", "description": "ID of the workflow run"}

        # correct
        "limit":    {"type": "integer", "description": "Max results to return"}
        "latitude": {"type": "number",  "description": "Latitude coordinate"}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "number":
            continue
        name_lower = param_name.lower()
        is_integer_name = (
            _INTEGER_SUFFIX_RE.search(name_lower) is not None
            or _INTEGER_ID_RE.search(name_lower) is not None
        )
        if not is_integer_name:
            continue
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="number_type_for_integer",
            message=(
                "param '{param}' is declared as 'number' but the name implies an integer; "
                "use \"type\": \"integer\" to prevent models from supplying fractional values"
            ).format(param=param_name),
        ))

    return issues


def _check_array_items_object_no_properties(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 41: array_items_object_no_properties — array items typed as object but no 'properties' defined.

    Check 12 (``nested_objects_have_properties``) catches top-level params that
    are ``type: "object"`` with no ``properties``.  This check extends that
    coverage to array items: when an array param's ``items`` schema declares
    ``type: "object"`` but provides no ``properties``, the model knows each
    element should be an object but has no idea what fields that object should
    contain.

    Without ``properties``, the model must hallucinate the object structure
    based on the param name, description, and training data — none of which
    are machine-readable contracts.  This leads to incorrectly shaped objects,
    missing required fields, and failed API calls.

    Fires when:

    * A top-level param is ``type: "array"``, AND
    * Its ``items`` schema exists, AND
    * ``items.type`` is ``"object"``, AND
    * ``items`` has no ``properties`` field.

    Examples::

        # flagged — array of objects with no defined structure
        "scopes":      {"type": "array", "items": {"type": "object"}}
        "headers":     {"type": "array", "items": {"type": "object", "description": "..."}}
        "operations":  {"type": "array", "items": {"type": "object"}}

        # correct — model knows what each object should contain
        "scopes":  {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "description": "Scope identifier"},
                    "description": {"type": "string"}
                },
                "required": ["value"]
            }
        }
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "array":
            continue
        items = param_schema.get("items")
        if not isinstance(items, dict):
            continue
        if items.get("type") == "object" and "properties" not in items:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="array_items_object_no_properties",
                message=(
                    "array param '{param}' items are typed as object but have no 'properties' defined; "
                    "models cannot know what fields each object should contain — "
                    "add a 'properties' schema to the items definition"
                ).format(param=param_name),
            ))

    return issues


_TOOL_DESC_STOP = frozenset({
    "a", "an", "the", "this", "that", "of", "to", "for", "from",
    "in", "on", "at", "by", "is", "it", "or", "and", "be", "are",
    "was", "with", "if", "its", "as", "all", "up", "out",
})
_TOOL_DESC_STRIP_RE = re.compile(r"[^a-z0-9 ]")


def _check_tool_description_just_the_name(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 42: tool_description_just_the_name — tool description merely restates the tool name.

    Check 33 (``description_just_the_name``) catches *parameter* descriptions
    that only restate the parameter name.  This check applies the same
    principle to *tool* descriptions: if every significant word in the
    description is already present in the tool name (after splitting on ``_``),
    the description adds zero information.

    Examples of descriptions that fail:

    * ``list_repositories`` → ``"List repositories"``
    * ``notion_retrieve_block`` → ``"Retrieve a block from Notion"``
    * ``delete_content_type`` → ``"Delete a content type"``
    * ``approve_merge_request`` → ``"Approve a merge request"``

    These descriptions do nothing the model couldn't infer from the name
    itself.  A useful description would explain what the tool returns, what
    side effects it has, what parameters are critical, or what use case it
    serves — things the name cannot express.

    Fires when **all** of the following hold:

    * The tool description is 20+ characters (shorter already caught by Check 20)
    * The description is 8 words or fewer
    * Every significant word in the description (3+ chars, not a stop word)
      is present in the set of words that make up the tool name (split on ``_``
      and ``-``, lowercased)

    Examples::

        # flagged — adds nothing beyond what the name conveys
        name="list_repositories",       description="List repositories"
        name="notion_retrieve_block",   description="Retrieve a block from Notion"
        name="delete_content_type",     description="Delete a content type"

        # correct — adds context beyond the name
        name="list_repositories", description="List public and private repositories for the authenticated user or a specified organization."
        name="get_file",          description="Retrieve the contents of a file at a given path in a repository."
    """
    # Get description from the raw tool object (format-agnostic)
    if fmt == "openai":
        desc = obj.get("function", {}).get("description", "") or ""
    elif fmt in ("anthropic", "mcp"):
        desc = obj.get("description", "") or ""
    elif fmt == "google":
        desc = obj.get("description", "") or ""
    else:
        desc = obj.get("description", "") or ""

    if not desc or not isinstance(desc, str):
        return None
    if len(desc) < 20:
        return None  # too short → already caught by Check 20
    if len(desc.split()) > 8:
        return None  # longer descriptions likely add real value

    # Words from the tool name (split on _ and -)
    raw_name_words = re.split(r"[_\-]", name.lower())
    name_words = {w for w in raw_name_words if len(w) >= 2}
    if not name_words:
        return None

    # Significant words from the description: 3+ chars, not stop words
    desc_tokens = _TOOL_DESC_STRIP_RE.sub(" ", desc.lower()).split()
    sig_words = {w for w in desc_tokens if len(w) >= 3 and w not in _TOOL_DESC_STOP}
    if not sig_words:
        return None  # no significant words to check

    if sig_words.issubset(name_words):
        return Issue(
            tool=name,
            severity="warn",
            check="tool_description_just_the_name",
            message=(
                "tool description '{desc}' only restates the tool name '{name}'; "
                "add context about what the tool returns, its side effects, or "
                "when to use it versus similar tools"
            ).format(name=name, desc=desc[:60]),
        )
    return None


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

        # Check 14: name_snake_case
        issue = _check_name_snake_case(name)
        if issue is not None:
            issues.append(issue)

        # Check 15: param_snake_case
        issues.extend(_check_param_snake_case(name, schema))

        # Check 5: description_present
        issue = _check_description_present(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 6: description_not_empty
        issue = _check_description_not_empty(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 20: tool_description_too_short
        issue = _check_description_too_short(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 25: tool_description_too_long
        issue = _check_description_too_long(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 8: parameters_valid_type
        issues.extend(_check_parameters_valid_type(name, schema))

        # Check 9: required_params_exist
        issues.extend(_check_required_params_exist(name, schema))

        # Check 27: required_missing
        issue = _check_required_missing(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 28: nested_required_missing
        issues.extend(_check_nested_required_missing(name, schema))

        # Check 29: too_many_params
        issue = _check_too_many_params(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 30: default_undocumented
        issue = _check_default_undocumented(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 31: enum_undocumented
        issue = _check_enum_undocumented(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 32: numeric_constraints_missing
        issue = _check_numeric_constraints_missing(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 33: description_just_the_name
        issue = _check_description_just_the_name(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 34: description_multiline
        issue = _check_description_multiline(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 42: tool_description_just_the_name
        issue = _check_tool_description_just_the_name(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 35: description_redundant_type
        issues.extend(_check_description_redundant_type(name, schema))

        # Check 36: param_format_missing
        issues.extend(_check_param_format_missing(name, schema))

        # Check 37: boolean_default_missing
        issues.extend(_check_boolean_default_missing(name, schema))

        # Check 38: enum_default_missing
        issues.extend(_check_enum_default_missing(name, schema))

        # Check 39: default_in_description_not_schema
        issues.extend(_check_default_in_description_not_schema(name, schema))

        # Check 40: number_type_for_integer
        issues.extend(_check_number_type_for_integer(name, schema))

        # Check 41: array_items_object_no_properties
        issues.extend(_check_array_items_object_no_properties(name, schema))

        # Check 10: enum_is_array
        issues.extend(_check_enum_is_array(name, schema))

        # Check 11: properties_is_object
        issue = _check_properties_is_object(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 12: nested_objects_have_properties
        issues.extend(_check_nested_objects_have_properties(name, schema))

        # Check 16: nested_param_snake_case
        issues.extend(_check_nested_param_snake_case(name, schema))

        # Check 17: array_items_missing
        issues.extend(_check_array_items_missing(name, schema))

        # Check 18: param_description_missing
        issues.extend(_check_param_description_missing(name, schema))

        # Check 19: nested_param_description_missing
        issues.extend(_check_nested_param_description_missing(name, schema))

        # Check 21: param_description_too_short
        issues.extend(_check_param_description_too_short(name, schema))

        # Check 26: param_description_too_long
        issues.extend(_check_param_description_too_long(name, schema))

        # Check 22: param_type_missing
        issues.extend(_check_param_type_missing(name, schema))

        # Check 23: nested_param_type_missing
        issues.extend(_check_nested_param_type_missing(name, schema))

        # Check 24: array_items_type_missing
        issues.extend(_check_array_items_type_missing(name, schema))

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
