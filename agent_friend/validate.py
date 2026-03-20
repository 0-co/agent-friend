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


def _check_description_duplicate(tool_descs: List[Tuple[str, str]]) -> List[Issue]:
    """Check 61: description_duplicate — two or more tools share the same description.

    Identical descriptions across different tools indicate copy-paste
    documentation that was never updated.  Each tool does something distinct;
    its description should be equally distinct.  When two tools share the same
    sentence, the model cannot tell which tool to use — the schema signal is
    diluted.

    This check fires for every tool that shares its description with at least
    one other tool in the same server.  The issue is reported on each
    duplicated tool (not just the second occurrence).

    Fires when:

    * Two or more tools have exactly the same non-empty description string
      (after stripping leading/trailing whitespace), AND
    * The description is at least 10 characters (to avoid penalising trivially
      short placeholder descriptions already caught by Check 20).

    Does **not** fire on:

    * Tools with empty or missing descriptions (Check 5/6 handles those).
    * Descriptions under 10 characters (too short to be meaningful duplicates).

    Examples::

        # flagged — two tools have the same description
        {"name": "search_users",    "description": "Search the database."}
        {"name": "search_products", "description": "Search the database."}

        # ok — descriptions are distinct
        {"name": "search_users",    "description": "Search users by name or email."}
        {"name": "search_products", "description": "Search products by keyword or SKU."}
    """
    # Build a map from description → list of tool names
    desc_map = {}  # type: Dict[str, List[str]]
    for tool_name, desc in tool_descs:
        if desc and len(desc) >= 10:
            desc_map.setdefault(desc, []).append(tool_name)

    issues = []
    for desc, tool_names in desc_map.items():
        if len(tool_names) < 2:
            continue
        others_str = ", ".join(f"'{t}'" for t in tool_names[:3])
        if len(tool_names) > 3:
            others_str += f" and {len(tool_names) - 3} more"
        for tool_name in tool_names:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="description_duplicate",
                message=(
                    "tool description is identical to {count} other tool{s} ({others}) — "
                    "each tool should have a unique description; "
                    "copy-pasted descriptions dilute the signal for the model."
                ).format(
                    count=len(tool_names) - 1,
                    s="s" if len(tool_names) - 1 != 1 else "",
                    others=others_str,
                ),
            ))
    return issues


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


# ---------------------------------------------------------------------------
# Check 53: tool_name_redundant_prefix
# ---------------------------------------------------------------------------

_VERB_PREFIXES = frozenset({
    "get", "set", "list", "create", "update", "delete", "search", "find",
    "run", "execute", "manage", "generate", "fetch", "read", "write",
    "send", "add", "remove", "check", "handle", "process", "load", "save",
    "query", "call", "post", "put", "patch", "edit", "batch", "export",
    "import", "upload", "download", "analyze", "parse", "validate", "test",
    "start", "stop", "reset", "init", "configure", "show", "view", "make",
    "build", "deploy", "describe", "enable", "disable", "toggle", "copy",
    "move", "rename", "clone", "merge", "split", "convert", "transform",
    "do", "use", "apply", "open", "close", "lock", "unlock", "sync",
    "push", "pull", "submit", "cancel", "archive", "restore", "refresh",
    "resolve", "assign", "unassign", "invite", "revoke", "grant", "deny",
})


def _check_tool_name_redundant_prefix(names: List[str]) -> List[Issue]:
    """Check 53: tool_name_redundant_prefix — most tools share a redundant service-name prefix.

    When a developer names all their tools with the same prefix — e.g.,
    ``auth0_list_applications``, ``auth0_create_application``,
    ``auth0_get_application`` — the ``auth0_`` prefix is pure overhead.
    The MCP client already namespaces tools by server; models see the server
    name from the protocol layer, not from individual tool names.  A prefix
    like ``hubspot_``, ``asana_``, ``chroma_`` repeats the server identity on
    every tool, wasting tokens on every call and making tool names harder to
    read at a glance.

    Fires when **all** of the following hold:

    * At least 3 tools are present, AND
    * 80 % or more of tools share the same first word (split on ``_``), AND
    * That first word is **not** a common action verb (``get``, ``list``,
      ``create``, ``delete``, etc.) — verbs are meaningful groupings, not
      redundant namespace prefixes, AND
    * The shared prefix is 3–15 characters long.

    Fires **once** per tool list (not per tool), with the first matching tool
    used as the issue anchor.

    Examples::

        # flagged — 'auth0_' prefix repeats the server name
        ["auth0_list_applications", "auth0_create_application", "auth0_delete_application"]

        # flagged — 'hubspot_' prefix repeats the server name
        ["hubspot_create_company", "hubspot_get_company", "hubspot_search_contacts"]

        # ok — 'get_' is an action verb, not a redundant service name
        ["get_weather", "get_forecast", "get_alerts"]

        # ok — tools have different prefixes, no single dominant one
        ["search_users", "create_issue", "list_labels", "get_repo"]

    The fix is to remove the shared prefix.  The server name provides context;
    ``list_applications`` in an Auth0 MCP server is unambiguous.
    """
    from collections import Counter

    first_words = Counter()  # type: Counter
    for name in names:
        if "_" in name:
            first_word = name.split("_")[0].lower()
            first_words[first_word] += 1

    if not first_words:
        return []

    top_word, count = first_words.most_common(1)[0]
    total = len(names)

    if (
        count >= 3
        and count >= int(total * 0.8)
        and top_word not in _VERB_PREFIXES
        and 3 <= len(top_word) <= 15
    ):
        # Find first example tool with this prefix for the issue anchor
        sample = next(
            (n for n in names if "_" in n and n.split("_")[0].lower() == top_word),
            names[0],
        )
        stripped = sample[len(top_word) + 1:]  # remove prefix + underscore

        return [Issue(
            tool=sample,
            severity="warn",
            check="tool_name_redundant_prefix",
            message=(
                "{count}/{total} tools share the '{prefix}_' prefix — this repeats the "
                "server name and wastes tokens on every call; the MCP client already "
                "namespaces tools by server. Example: rename '{old}' → '{new}'"
            ).format(
                count=count,
                total=total,
                prefix=top_word,
                old=sample,
                new=stripped,
            ),
        )]

    return []


# ---------------------------------------------------------------------------
# Check 54: optional_string_no_minlength
# ---------------------------------------------------------------------------

_OPTIONAL_STRING_CONTENT_PARTS = frozenset({
    "query", "sql", "statement", "script", "command",
    "message", "prompt", "instruction", "expression", "formula", "template",
    "text", "search", "keyword", "term", "phrase",
})
"""Name-part keywords for optional string params that cannot meaningfully be empty."""

_OPTIONAL_ID_SUFFIX_RE = re.compile(r'(?:_|-)?ids?$', re.IGNORECASE)
_OPTIONAL_TYPE_SUFFIX_RE = re.compile(r'(?:_|-)type$', re.IGNORECASE)


def _check_optional_string_no_minlength(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 54: optional_string_no_minlength — optional content-like string param has no minLength.

    Complements check 49 (``required_string_no_minlength``), which only covers
    required params.  An *optional* string param named ``query``, ``text``,
    ``message``, ``prompt``, ``command``, etc. may be omitted entirely — but
    if the model *does* supply it, an empty string ``""`` is almost always
    semantically wrong and will cause a downstream API error.

    JSON Schema allows ``""`` by default for any unconstrained string, so the
    model cannot distinguish "omit this param" from "pass empty string".
    Adding ``"minLength": 1`` closes the gap: validators reject ``""`` and the
    model can only omit the param or supply a non-empty value.

    Fires when **all** of the following hold:

    * Param is **not** in the ``required`` array (optional), AND
    * Param type is ``string`` (or unset, no nested properties), AND
    * Param name contains a content-like keyword part (see list below), AND
    * Param has no ``minLength``, ``enum``, ``pattern``, or ``const``, AND
    * Param name does not end with ``_id`` / ``_ids`` (identifier), AND
    * Param name does not end with ``_type`` (type discriminator)

    Keywords that trigger: query, sql, statement, script, command, message,
    prompt, instruction, expression, formula, template, text, search, keyword,
    term, phrase.

    Special case: bare ``code`` or ``code_*`` (but not ``*_code`` suffixes like
    ``country_code``) is also included.

    Examples::

        # flagged — optional query string allows empty
        "query":   {"type": "string"}
        "search_query": {"type": "string", "description": "Search term"}
        "message": {"type": "string"}

        # correct — minLength prevents empty string
        "query":   {"type": "string", "minLength": 1}
        "message": {"type": "string", "minLength": 1}

        # not flagged — already constrained
        "status":  {"type": "string", "enum": ["active", "inactive"]}
        "country_code": {"type": "string"}  # _code suffix, not content

        # not flagged — required params handled by check 49
        required: ["query"]
    """
    issues = []
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    required_set = set(required)

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if param_name in required_set:
            continue  # required params handled by check 49
        if not isinstance(param_schema, dict):
            continue
        # Skip ID params
        if _OPTIONAL_ID_SUFFIX_RE.search(param_name):
            continue
        # Skip type-discriminator params (query_type, filter_type, etc.)
        if _OPTIONAL_TYPE_SUFFIX_RE.search(param_name):
            continue

        ptype = param_schema.get("type")
        if ptype not in (None, "string"):
            continue
        if ptype is None and "properties" in param_schema:
            continue  # object without explicit type — skip

        # Already constrained
        if param_schema.get("minLength") is not None:
            continue
        if param_schema.get("enum") is not None:
            continue
        if param_schema.get("pattern") is not None:
            continue
        if param_schema.get("const") is not None:
            continue

        # Check name parts against content keywords
        parts = set(re.split(r'[_\-\s]+', param_name.lower()))
        matched_kw = parts & _OPTIONAL_STRING_CONTENT_PARTS

        # Special case for 'code': only match standalone or as first part
        # (not as a suffix like country_code, zip_code, currency_code)
        if not matched_kw:
            name_parts = re.split(r'[_\-\s]+', param_name.lower())
            if "code" in name_parts and name_parts.index("code") == 0:
                matched_kw = {"code"}

        if not matched_kw:
            continue

        matched = next(iter(matched_kw))
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="optional_string_no_minlength",
            message=(
                "optional param '{param}' is a content string ('{kw}') with no "
                "'minLength' — the schema allows '\"\"' (empty string); add "
                "'minLength: 1' so the model cannot accidentally pass an empty value"
            ).format(param=param_name, kw=matched),
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


def _check_required_array_empty(name: str, schema: Dict[str, Any]) -> Optional[Issue]:
    """Check 46: required_array_empty — tool has ``required: []`` but parameters exist with no defaults.

    ``required: []`` is an *explicit* declaration: "I thought about this and
    decided nothing is required."  But when parameters have no ``default``
    value, the model cannot know whether they are truly optional or just
    forgotten.  Calling the tool without those parameters may fail silently.

    This is the complement of Check 27 (``required_missing``), which fires when
    ``required`` is absent.  Here ``required`` is present but intentionally
    empty — which Check 27 exempts — yet the schema still offers no guidance
    on which parameters are safe to omit.

    Does not fire when:
    - ``required`` is absent (Check 27 handles that)
    - ``required`` is non-empty (parameters are explicitly marked)
    - All parameters have ``default`` values (optionality is documented)
    - There are no parameters (nothing to mark required)

    Fix: add ``default`` values to confirm parameters are intentionally
    optional, or move genuinely required parameters into ``required``.

    Examples::

        # flags — required: [] but no defaults (check fires)
        "required": [],
        "properties": {
            "paths":  {"type": "array", "description": "Files to upload"},
            "format": {"type": "string", "description": "Output format"},
        }

        # ok — required: [] and all params have defaults
        "required": [],
        "properties": {
            "format": {"type": "string", "default": "json", "description": "Output format"},
            "limit":  {"type": "integer", "default": 10, "description": "Max results"},
        }
    """
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 0:
        return None
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or not properties:
        return None
    no_default = [p for p, ps in properties.items()
                  if isinstance(ps, dict) and "default" not in ps]
    if not no_default:
        return None
    count = len(no_default)
    return Issue(
        tool=name,
        severity="warn",
        check="required_array_empty",
        message=(
            "tool has 'required: []' but {count} parameter{s} ({params}) "
            "have no 'default' value — models cannot tell which are safe to omit; "
            "add defaults to confirm optionality or move required params into 'required'."
        ).format(
            count=count,
            s="s" if count != 1 else "",
            params=", ".join(f"'{p}'" for p in no_default[:3])
            + (" ..." if count > 3 else ""),
        ),
    )


_TOOL_DESC_MARKDOWN_RE = re.compile(
    r'`[^`\n]{1,60}`'                         # `code` spans in tool descriptions
    r'|\*\*[A-Za-z][A-Za-z0-9 _,!:.#-]+\*\*'  # **bold text**
    r'|```'                                     # ``` code fences
    r'|\n\s*#{1,4}\s+\w',                       # markdown headers
    re.MULTILINE,
)
_PARAM_DESC_MARKDOWN_RE = re.compile(
    r'\*\*[A-Za-z][A-Za-z0-9 _,!:.#-]+\*\*'  # **bold text** in param descriptions
    r'|```'                                     # ``` code fences
    r'|\n\s*#{1,4}\s+\w',                       # markdown headers
    re.MULTILINE,
)
"""Patterns that indicate markdown formatting in descriptions."""


def _check_description_markdown_formatting(tool_name: str, tool_description: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 47: description_markdown_formatting — markdown syntax in tool or param descriptions.

    Markdown formatting (backtick code spans, **bold**, headers, fenced code
    blocks) does not render in most LLM runtimes when reading tool schemas.
    The asterisks, backticks, and pound signs appear as literal characters,
    adding noise tokens without conveying extra meaning.

    Tool descriptions with markdown bold like ``**IMPORTANT:**`` or headers
    like ``### When to use this tool`` add token overhead.  Inline backtick
    spans such as `` `find_organizations()` `` are common in AI-generated
    schemas but the formatting is invisible to the model.

    Fires on:
    - **Tool descriptions** with backtick code spans, ``**bold**``, ````` ``` `````
      fences, or ``##`` headers.
    - **Param descriptions** with ``**bold**``, fences, or headers (single
      backtick spans are allowed in param descriptions as they often show
      expected values like `` `true` `` or `` `json` ``).

    Does not fire when there is no markdown formatting detected.

    Fix: use plain prose.  State what the tool does without markdown
    formatting.  The model reads descriptions as plain text.

    Examples::

        # flags — markdown in tool description
        {
          "name": "init",
          "description": "**IMPORTANT:** Call this first. ### Setup\\nRequired step.",
        }

        # flags — code fence in param description
        {
          "name": "format",
          "description": "Output format: ```json\\n{ ... }\\n```"
        }

        # ok — plain text
        {
          "name": "init",
          "description": "Initialize the session. Must be called before other tools.",
        }
    """
    issues = []

    # Check tool description
    if isinstance(tool_description, str) and _TOOL_DESC_MARKDOWN_RE.search(tool_description):
        matches = _TOOL_DESC_MARKDOWN_RE.findall(tool_description)
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_markdown_formatting",
            message=(
                "tool description contains markdown formatting ({sample}) — "
                "markdown does not render in LLM tool schemas; use plain text."
            ).format(sample=repr(matches[0][:40])),
        ))

    # Check param descriptions (only bold/headers/fences, not single backtick spans)
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            desc = param_schema.get("description")
            if not isinstance(desc, str):
                continue
            if _PARAM_DESC_MARKDOWN_RE.search(desc):
                matches = _PARAM_DESC_MARKDOWN_RE.findall(desc)
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_markdown_formatting",
                    message=(
                        "param '{param}' description contains markdown formatting ({sample}) — "
                        "markdown does not render in LLM tool schemas; use plain text."
                    ).format(param=param_name, sample=repr(matches[0][:40])),
                ))

    return issues


_TOOL_MODEL_INSTRUCTIONS_RE = re.compile(
    r'\byou (?:must|should|need to|have to|always|never)\b'
    r'|\balways (?:call|use|pass|include|check|verify|ensure|start|begin|run)\b'
    r'|\bnever (?:call|use|pass|include|skip|omit)\b'
    r'|\bIMPORTANT:\s*(?:call|use|this tool must|always)\b'
    r'|\bmust be called (?:before|first|after|instead)\b'
    r'|\bprioritize (?:this|using)\b'
    # Orchestration-hint patterns (added v0.104.0)
    r'|\buse\s+this\s+tool\s+when\b'
    r'|\bwhen\s+to\s+use\b'
    r'|\bdo\s+not\s+use\s+this\b'
    r'|\bcall\s+this\s+(?:tool\s+)?(?:first|before|after)\b'
    r'|\bthis\s+tool\s+should\s+(?:only|never|not)\s+be\b'
    r'|\bonly\s+(?:call|use)\s+this\s+(?:tool\s+)?when\b',
    re.IGNORECASE,
)
"""Patterns that indicate model-directing language in tool descriptions."""


def _check_description_model_instructions(tool_name: str, tool_description: str) -> Optional[Issue]:
    """Check 48: description_model_instructions — tool description contains language
    directing model behavior rather than describing what the tool does.

    Tool descriptions should describe what a tool does.  Phrases like
    "you must call X first", "always pass Y", "never skip Z", or
    "Use this tool when:" sections are operational instructions for the
    model — they belong in the system prompt, not in the schema.
    Putting them here:

    * wastes tokens on every tool call (the model re-reads the full
      schema each time)
    * mixes "what" (schema) with "how" (system prompt)
    * may conflict with system-level instructions from the host application

    Fires on (examples):
    - "you must/should/need to/have to/always/never ..."
    - "always call/use/pass/include ..."
    - "never call/use/pass/include/skip ..."
    - "IMPORTANT: call/use ..."
    - "must be called before/first/after ..."
    - "Use this tool when: ..."
    - "When to use: ..."
    - "Do not use this ..."
    - "Call this first/before/after ..."
    - "This tool should only/never/not be ..."
    - "Only call/use this tool when ..."

    Does not fire on:
    - "always" used as a modifier unrelated to model behavior
      ("always-on service", "always returns a list")
    - descriptive use of "should" meaning a normal outcome
      ("the response should contain...")
    - Check 13 already covers malicious override patterns.
    """
    if not isinstance(tool_description, str) or not tool_description:
        return None
    m = _TOOL_MODEL_INSTRUCTIONS_RE.search(tool_description)
    if not m:
        return None
    return Issue(
        tool=tool_name,
        severity="warn",
        check="description_model_instructions",
        message=(
            "tool description contains model-directing language ({sample}) — "
            "instructions on how to use the tool belong in the system prompt, "
            "not in the schema."
        ).format(sample=repr(m.group(0)[:50])),
    )


_REQUIRED_STRING_CONTENT_PARTS = frozenset({
    "query", "sql", "statement", "code", "script", "program", "command",
    "message", "prompt", "instruction", "expression", "formula", "template",
})
"""Name-part keywords that strongly imply a string value cannot be empty."""


def _check_required_string_no_minlength(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 49: required_string_no_minlength — required content-like string param has no minLength.

    A required string parameter with a content-like name (query, code, message,
    command, prompt, script, etc.) that omits ``minLength`` technically allows
    an empty string ``""`` to pass JSON Schema validation.  At runtime the
    downstream API will almost always reject the empty value, making this a
    latent correctness bug.

    Adding ``"minLength": 1`` makes the contract explicit: the model cannot
    accidentally pass an empty string and get a confusing runtime error.

    Does not fire when:
    - The parameter is not in the ``required`` array
    - The parameter type is not ``string`` (or unset string)
    - The parameter already has ``minLength`` set
    - The parameter has an ``enum`` (values already constrained)
    - The parameter has a ``pattern`` (format already constrained)
    - The parameter name ends with ``_id`` or ``_ids`` (it's an identifier, not content)
    - The parameter name contains none of the content-like keywords
    """
    issues = []
    required = schema.get("required", [])
    if not isinstance(required, list) or not required:
        return issues
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues
    _id_suffix_re = re.compile(r'(?:_|-)?ids?$', re.IGNORECASE)
    for param_name, param_schema in properties.items():
        if param_name not in required:
            continue
        if not isinstance(param_schema, dict):
            continue
        # Skip ID params — they should use format/pattern constraints, not minLength
        if _id_suffix_re.search(param_name):
            continue
        ptype = param_schema.get("type")
        if ptype not in (None, "string"):
            continue
        if ptype is None and "properties" in param_schema:
            # likely an object without explicit type — skip
            continue
        if param_schema.get("minLength") is not None:
            continue
        if param_schema.get("enum") is not None:
            continue
        if param_schema.get("pattern") is not None:
            continue
        # Check if any name part is a content keyword
        parts = set(re.split(r'[_\-\s]+', param_name.lower()))
        if not parts & _REQUIRED_STRING_CONTENT_PARTS:
            continue
        matched = next(iter(parts & _REQUIRED_STRING_CONTENT_PARTS))
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="required_string_no_minlength",
            message=(
                "required param '{param}' is a content string ('{kw}') with no 'minLength' — "
                "the schema allows '\"\"' (empty string); add 'minLength: 1' to reject "
                "empty values at validation time."
            ).format(param=param_name, kw=matched),
        ))
    return issues


# ---------------------------------------------------------------------------
# Check 50: param_description_says_optional
# ---------------------------------------------------------------------------

_SAYS_OPTIONAL_RE = re.compile(
    r'^(\(optional\)\s*|optional\s*[:\-–]\s*|optional\s+)',
    re.IGNORECASE,
)


def _check_param_description_says_optional(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 50: param_description_says_optional — param description starts with 'Optional:' or '(optional)'.

    JSON Schema already expresses optionality through the ``required`` array: if
    a parameter is not listed there, it is optional.  Repeating that fact in the
    description prefix (``"Optional: ..."`` or ``"(optional) ..."`` etc.) is pure
    redundancy — the model can read the schema; it does not need the prose to
    re-state what the schema already encodes.  This also wastes tokens on every
    call that includes the tool definition.

    Common forms that fire:

    * ``"Optional: Language code for the output"``
    * ``"(Optional) Filter by status"``
    * ``"Optional - maximum number of results"``
    * ``"Optional. Overrides the default timeout."``

    Fires when:

    * Param is **not** listed in the tool's ``required`` array, AND
    * Param description starts with the above patterns (case-insensitive)

    Does **not** fire for required params (even if they mistakenly say
    "optional" — that is a different, more severe issue) because the focus is
    on the common redundancy pattern in optional params.

    The fix is simply to remove the prefix and let the ``required`` array
    communicate optionality.

    Examples::

        # flagged — 'optional' prefix is redundant with the schema required array
        "lang":   {"type": "string", "description": "Optional: Language code (e.g. 'en', 'fr')"}
        "limit":  {"type": "integer", "description": "Optional: Maximum results to return"}
        "filter": {"type": "string", "description": "(optional) Filter expression"}

        # correct — description goes straight to the point
        "lang":   {"type": "string", "description": "Language code (e.g. 'en', 'fr')", "default": "en"}
        "limit":  {"type": "integer", "description": "Maximum results to return (default: 10)", "default": 10}
        "filter": {"type": "string", "description": "Filter expression"}
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
            continue  # only flag optional params (non-required ones)
        description = param_schema.get("description", "")
        if not description or not isinstance(description, str):
            continue
        if _SAYS_OPTIONAL_RE.match(description.strip()):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_description_says_optional",
                message=(
                    "optional param '{param}' description starts with 'Optional' — "
                    "redundant with the schema's 'required' array; remove the prefix "
                    "and let the schema structure communicate optionality"
                ).format(param=param_name),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 55: required_param_has_default
# ---------------------------------------------------------------------------


def _check_required_param_has_default(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 55: required_param_has_default — required param also declares a ``default``.

    In JSON Schema the ``required`` array means the caller **must** supply the
    parameter — the server will reject the call if it is absent.  The
    ``default`` keyword means "use this value when the caller omits the
    parameter."  These two signals contradict each other: if the caller must
    always provide the parameter, the default can never be applied; if the
    default is meaningful, the parameter should be optional.

    This contradiction confuses LLMs: the model sees ``required`` and infers it
    must supply a value, but also sees ``default: "gpt-4o"`` and may assume it
    can omit the param.  The resulting behaviour is unpredictable.

    Common causes:

    * Copy-paste from optional params where the ``default`` made sense.
    * Required + default used as documentation (showing the typical value).
    * Lazy schema migration where optionality changed but ``required`` was not
      updated.

    Fires when:

    * Param is listed in the tool's ``required`` array, AND
    * Param schema contains a ``"default"`` key (any non-null value)

    Does **not** fire when the ``default`` value is ``null`` — a ``null``
    default on a required param is unusual but may indicate nullability rather
    than optionality.

    Examples::

        # flagged — required param with a contradictory default
        properties:
          model:    {type: string, default: "gpt-4o"}
          format:   {type: string, default: "json"}
        required: [model, format]

        # correct — required param with no default (caller always provides it)
        properties:
          model:    {type: string, description: "Model ID to use"}
        required: [model]

        # correct — optional param with a default
        properties:
          format:   {type: string, default: "json", description: "Output format"}
        required: []   # format is optional, default applies when omitted
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    if not required:
        return issues

    for param_name in required:
        param_schema = properties.get(param_name)
        if not isinstance(param_schema, dict):
            continue
        if "default" not in param_schema:
            continue
        default_value = param_schema["default"]
        if default_value is None:
            continue  # null default is an edge case, skip
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="required_param_has_default",
            message=(
                "required param '{param}' has 'default: {default}' — "
                "required params must always be supplied by the caller so the "
                "default can never apply; either remove the param from 'required' "
                "(if the default is meaningful) or remove the 'default' field "
                "(if the param is truly mandatory)"
            ).format(param=param_name, default=repr(default_value)),
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 58: description_allows_you_to
# ---------------------------------------------------------------------------

_ALLOWS_YOU_TO_RE = re.compile(
    r'^(?:'
    r'Allows?\s+(?:you|the\s+(?:model|user|agent|caller|client))\s+to\s+'
    r'|Enables?\s+(?:you|the\s+(?:model|user|agent|caller|client))\s+to\s+'
    r'|Lets?\s+(?:you|the\s+(?:model|user|agent|caller|client))\s+'
    r'|Helps?\s+(?:you|the\s+(?:model|user|agent|caller|client))\s+(?:to\s+)?'
    r'|Used\s+to\s+'
    r'|Can\s+be\s+used\s+to\s+'
    r'|Provides?\s+(?:you|the\s+(?:model|user|agent|caller|client))\s+(?:with\s+)?(?:the\s+ability|a\s+way|the\s+capability)\s+to\s+'
    r')',
    re.IGNORECASE,
)
"""Patterns indicating weak 'allows you to X' preambles in tool descriptions."""


def _check_description_allows_you_to(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 58: description_allows_you_to — tool description starts with an indirect
    'Allows you to X', 'Enables you to X', 'Lets you X', or 'Used to X' preamble
    instead of directly stating the action in imperative mood.

    These phrasings describe the tool from the user's perspective ("you get to do X")
    rather than directly naming the action ("do X").  They waste tokens, add noise, and
    weaken the signal — the model must extract the real action from the preamble.

    Fires when the description starts with:

    * ``Allows you to …`` / ``Allows the model to …``
    * ``Enables you to …`` / ``Enables the agent to …``
    * ``Lets you …``
    * ``Helps you to …`` / ``Helps the caller …``
    * ``Used to …``
    * ``Can be used to …``
    * ``Provides you with the ability to …``

    Does **not** fire on descriptions that simply *mention* these phrases mid-sentence.

    Examples::

        # flagged — indirect preamble
        "Allows you to search for files by name."
        "Enables you to create new records in the database."
        "Used to retrieve the current user session."
        "Can be used to send messages to a Slack channel."

        # correct — direct imperative
        "Search for files by name."
        "Create new records in the database."
        "Retrieve the current user session."
        "Send messages to a Slack channel."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _ALLOWS_YOU_TO_RE.match(desc)
    if not m:
        return None
    preamble = m.group(0).strip()
    return Issue(
        tool=name,
        severity="warn",
        check="description_allows_you_to",
        message=(
            "tool description starts with indirect preamble '{preamble}' — "
            "drop the preamble and use imperative mood directly "
            "(e.g. 'Search for files' not 'Allows you to search for files')."
        ).format(preamble=preamble),
    )


# ---------------------------------------------------------------------------
# Check 57: description_this_tool
# ---------------------------------------------------------------------------

_THIS_TOOL_PREAMBLE_RE = re.compile(
    r'^This\s+(?:tool|function|API|endpoint|command|method|operation|action|call|request|route)\b',
    re.IGNORECASE,
)
"""Patterns matching redundant 'This tool/function/API...' preambles."""


def _check_description_this_tool(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 57: description_this_tool — tool description starts with a redundant
    'This tool/function/API...' preamble instead of jumping straight to the action.

    The model reading a tool schema already knows it is reading a tool description.
    Starting with "This tool creates a record" wastes tokens on noise.  The
    imperative form — "Create a record" — is shorter, clearer, and consistent
    with how the best-documented tools are written.

    Fires when the tool description starts with:

    * ``This tool …``
    * ``This function …``
    * ``This API …``
    * ``This endpoint …``
    * ``This command …``
    * ``This method …``
    * ``This operation …``
    * ``This action …``
    * ``This call …``
    * ``This request …``
    * ``This route …``

    Does **not** fire on:
    * "The tool…" (different pattern; not flagged)
    * "This returns…" (bare "This" without a tool-type noun)
    * Descriptions that happen to contain "this tool" mid-sentence

    Examples::

        # flagged — redundant preamble
        "This tool creates a new user account."
        "This function retrieves all active sessions."
        "This API allows you to search for records."

        # correct — no preamble needed
        "Create a new user account."
        "List all active sessions."
        "Search for records."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _THIS_TOOL_PREAMBLE_RE.match(desc)
    if not m:
        return None
    return Issue(
        tool=name,
        severity="warn",
        check="description_this_tool",
        message=(
            "tool description starts with redundant preamble '{preamble}' — "
            "the model already knows it's reading a tool description; "
            "start with the action directly (e.g. 'Create a user' not 'This tool creates a user')."
        ).format(preamble=m.group(0)),
    )


# ---------------------------------------------------------------------------
# Check 56: tool_description_non_imperative
# ---------------------------------------------------------------------------

_NON_IMPERATIVE_PREFIXES = re.compile(
    r'^(?:'
    r'Returns'
    r'|Provides'
    r'|Retrieves'
    r'|Fetches'
    r'|Gets'
    r'|Gives'
    r'|Lists'
    r'|Shows'
    r'|Displays'
    r'|Describes'
    r'|Allows'
    r'|Enables'
    r'|Outputs'
    r'|Checks'
    r'|Reads'
    r'|Counts'
    r')\s',
    re.IGNORECASE,
)
"""Patterns indicating a tool description starts with a non-imperative verb."""


def _check_tool_description_non_imperative(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 56: tool_description_non_imperative — tool description starts with an
    output-focused (non-imperative) verb rather than an action-imperative verb.

    MCP tool descriptions are injected into an LLM's context to explain what
    each tool does.  Best practice is to write them in imperative mood:
    "Search for files", "Create a record", "Delete the session" — these read
    as instructions and clearly convey the action.

    Descriptions that start with 3rd-person singular output verbs like
    "Returns a list of…", "Provides the current…", "Retrieves all…" describe
    *what comes back* rather than *what the tool does*.  They are technically
    accurate but subtly less effective: the model must infer the action from
    the output description rather than having it stated directly.

    Fires when:

    * The tool description's first word is a 3rd-person-singular present-tense
      verb from a known non-imperative set (Returns, Provides, Retrieves,
      Fetches, Gets, Gives, Lists, Shows, Displays, Describes, Allows,
      Enables, Outputs, Checks, Reads, Counts), AND
    * The word is followed by whitespace (avoiding false positives on words
      like "Reset", "Resolve", etc.)

    Does **not** fire on:
    * Imperative equivalents: "Return", "Provide", "Retrieve", "Fetch", etc.
    * Descriptions that begin with a noun phrase ("A utility that…")
    * Short descriptions that happen to start with these words as nouns

    Examples::

        # flagged — output-focused, 3rd-person
        "Returns the current user session token."
        "Provides a list of available models."
        "Retrieves all matching records from the index."

        # correct — imperative, action-focused
        "Get the current user session token."
        "List available models."
        "Retrieve all matching records from the index."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _NON_IMPERATIVE_PREFIXES.match(desc)
    if not m:
        return None
    first_word = m.group(0).strip()
    return Issue(
        tool=name,
        severity="warn",
        check="tool_description_non_imperative",
        message=(
            "tool description starts with non-imperative verb '{verb}' — "
            "use imperative mood (e.g. 'Get X' not 'Gets X', 'List X' not 'Lists X'). "
            "Imperative descriptions tell the model what action to take."
        ).format(verb=first_word),
    )


# ---------------------------------------------------------------------------
# Checks 72-73: name length
# ---------------------------------------------------------------------------

_TOOL_NAME_MAX_LENGTH = 60
_PARAM_NAME_MAX_LENGTH = 40


def _check_tool_name_too_long(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 72: tool_name_too_long — tool name exceeds 60 characters.

    Excessively long tool names waste tokens on every request that includes
    the tool definition.  They also hinder model reasoning — the model must
    process and remember long identifiers.

    A well-named tool is a concise, descriptive verb phrase:

    * ``get_user`` (8 chars) ✓
    * ``create_document_with_metadata`` (31 chars) ✓
    * ``get_all_user_profile_information_with_preferences_and_settings`` (62 chars) ✗

    Fires when: tool name length > 60 characters.

    Examples::

        # flagged
        "get_all_user_profile_information_with_preferences_and_settings"  # 62 chars

        # correct — concise name
        "get_user_profile"  # 16 chars
    """
    if not name or len(name) <= _TOOL_NAME_MAX_LENGTH:
        return None
    return Issue(
        tool=name,
        severity="warn",
        check="tool_name_too_long",
        message=(
            "tool name '{name}' is {n} characters (max {mx}) — use a concise "
            "verb phrase."
        ).format(name=name, n=len(name), mx=_TOOL_NAME_MAX_LENGTH),
    )


def _check_param_name_too_long(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 73: param_name_too_long — parameter name exceeds 40 characters.

    Long parameter names waste tokens and make the schema harder for the model
    to parse.  Parameter names should be concise identifiers.

    Fires when: any parameter name length > 40 characters.

    Examples::

        # flagged
        "maximum_number_of_results_to_return_per_page"  # 44 chars

        # correct
        "max_results"  # 11 chars
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if len(param_name) > _PARAM_NAME_MAX_LENGTH:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_too_long",
                message=(
                    "param name '{param}' is {n} characters (max {mx}) — use a concise "
                    "identifier."
                ).format(param=param_name, n=len(param_name), mx=_PARAM_NAME_MAX_LENGTH),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 74: description_word_repetition
# ---------------------------------------------------------------------------

_WORD_REPEAT_RE = re.compile(r'\b(\w{3,})\s+\1\b', re.IGNORECASE)


def _check_description_word_repetition(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 74: description_word_repetition — tool description contains
    consecutive repeated words (e.g. "search search", "the the").

    Repeated words are almost always a copy-paste or editing error.  They
    waste tokens and reduce description clarity.

    Fires when: the tool description contains two adjacent identical words
    (≥ 3 characters each, case-insensitive).

    Examples::

        # flagged
        "Searches the the repository for matching files"
        "Execute execute the given shell command"

        # correct
        "Searches the repository for matching files"
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _WORD_REPEAT_RE.search(desc)
    if not m:
        return None
    word = m.group(1)
    return Issue(
        tool=name,
        severity="warn",
        check="description_word_repetition",
        message=(
            "tool description contains repeated word '{word}' — remove the duplicate."
        ).format(word=word.lower()),
    )


# ---------------------------------------------------------------------------
# Check 75: default_type_mismatch
# ---------------------------------------------------------------------------

# Maps JSON Schema type names to the Python types that are valid for that type.
# "null" default is always allowed (means "not set") so it's excluded from checks.
_JS_TYPE_TO_PYTHON: Dict[str, tuple] = {
    "string":  (str,),
    "boolean": (bool,),
    "array":   (list,),
    "object":  (dict,),
    # integer and number: both int and float are fine for "number"; only int for "integer"
    # We handle these separately because bool is a subclass of int in Python.
    "integer": (int,),
    "number":  (int, float),
}


def _default_type_mismatch(declared_type: str, default_value: Any) -> bool:
    """Return True if default_value's Python type is incompatible with declared_type."""
    if default_value is None:
        return False  # null default is always acceptable
    allowed = _JS_TYPE_TO_PYTHON.get(declared_type)
    if allowed is None:
        return False  # unknown type, skip
    # bool is a subclass of int — treat it as boolean, not integer/number
    if isinstance(default_value, bool):
        return declared_type not in ("boolean",)
    return not isinstance(default_value, allowed)


def _check_default_type_mismatch(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 75: default_type_mismatch — a parameter's ``default`` value has a
    different JSON type than the parameter's declared ``type``.

    This is a schema correctness bug.  Examples:

    * ``{"type": "integer", "default": "5"}``   — string default for integer
    * ``{"type": "boolean", "default": "false"}`` — string default for boolean
    * ``{"type": "array",   "default": {}}``    — object default for array
    * ``{"type": "object",  "default": []}``    — array default for object
    * ``{"type": "string",  "default": 0}``     — number default for string

    A ``null`` default is always acceptable (means "use no default if omitted").

    Common cause: copy-paste from prose documentation, or auto-generated schemas
    that serialise defaults as strings regardless of the declared type.

    Severity: error — the schema is incorrect; the model may pass wrong-typed
    values or ignore the default entirely.
    """
    if schema is None:
        return []

    issues: List[Issue] = []
    properties = schema.get("properties") or {}

    def _check_props(props: Any, path: str) -> None:
        if not isinstance(props, dict):
            return
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            declared_type = param_schema.get("type")
            if not isinstance(declared_type, str):
                continue  # array type or no type — skip
            if "default" not in param_schema:
                continue
            default_val = param_schema["default"]
            if _default_type_mismatch(declared_type, default_val):
                param_path = f"{path}.{param_name}" if path else param_name
                issues.append(Issue(
                    tool=tool_name,
                    severity="error",
                    check="default_type_mismatch",
                    message=(
                        "param '{param}' declares type '{type}' but default value "
                        "{default!r} is {actual_type} — default must match declared type."
                    ).format(
                        param=param_path,
                        type=declared_type,
                        default=default_val,
                        actual_type=type(default_val).__name__,
                    ),
                ))
            # Recurse into nested objects
            nested_props = param_schema.get("properties") or {}
            if nested_props:
                _check_props(nested_props, f"{path}.{param_name}" if path else param_name)
            # Recurse into array items
            items = param_schema.get("items") or {}
            if isinstance(items, dict):
                nested_item_props = items.get("properties") or {}
                if nested_item_props:
                    item_path = f"{path}.{param_name}[]" if path else f"{param_name}[]"
                    _check_props(nested_item_props, item_path)

    _check_props(properties, "")
    return issues


# ---------------------------------------------------------------------------
# Check 76: param_name_implies_boolean
# ---------------------------------------------------------------------------

# Name prefixes that, by near-universal convention, signal a boolean parameter.
# Restricted to verb/auxiliary forms that are unambiguously boolean.
# Excluded: enable_, disable_, use_, show_, hide_, allow_, include_ — these are
# commonly used as verb-noun pairs that accept arrays or other types
# (e.g. include_domains: array, allow_users: array, enable_features: array).
_BOOL_PREFIX_RE = re.compile(
    r'^(is|has|should|can|was|will|did|are|were)_',
    re.IGNORECASE,
)


def _check_param_name_implies_boolean(
    tool_name: str, schema: Dict[str, Any]
) -> List[Issue]:
    """Check 76: param_name_implies_boolean — a parameter name starts with a
    conventional boolean prefix (``is_``, ``has_``, ``should_``, ``can_``,
    ``was_``, ``will_``, etc.) but the declared ``type`` is not ``"boolean"``.

    These prefixes signal boolean semantics across virtually all languages and
    API conventions: ``is_active``, ``has_permissions``, ``should_retry``,
    ``can_upload``.  When the declared type is ``"string"`` or ``"integer"``
    instead, the model receives conflicting signals — the name says boolean but
    the schema says otherwise — which degrades tool-selection accuracy.

    Common causes:

    * Auto-generated schema from a typed language where the bool was
      accidentally mapped to ``"string"`` (e.g., JSON serialisation quirk).
    * Developer intended a boolean flag but declared ``"string"`` accepting
      ``"true"``/``"false"`` strings instead of a proper boolean.
    * Copy-paste from a string param with the name left unchanged.

    Fires when:

    * A parameter name matches the boolean-prefix pattern, AND
    * The declared ``type`` is not ``"boolean"`` (and not absent — check 22
      handles missing type), AND
    * The type is not ``"null"``

    Examples::

        # flagged — 'is_' prefix but type is "string"
        {"name": "is_enabled", "type": "string", "description": "Whether enabled."}

        # flagged — 'has_' prefix but type is "integer"
        {"name": "has_access", "type": "integer", "description": "Access flag."}

        # correct
        {"name": "is_enabled", "type": "boolean", "description": "Whether enabled."}
    """
    issues: List[Issue] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_props(props: Dict[str, Any], path: str) -> None:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            full_path = f"{path}.{param_name}" if path else param_name
            declared_type = param_schema.get("type")
            if (
                declared_type is not None
                and declared_type not in ("boolean", "null")
                and _BOOL_PREFIX_RE.match(param_name)
            ):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="param_name_implies_boolean",
                    message=(
                        "param '{param}' name implies boolean ('{prefix}_' prefix) "
                        "but declares type '{type}' — use type 'boolean' or rename the param."
                    ).format(
                        param=full_path,
                        prefix=_BOOL_PREFIX_RE.match(param_name).group(1),
                        type=declared_type,
                    ),
                ))
            # Recurse into nested objects
            nested_props = param_schema.get("properties") or {}
            if isinstance(nested_props, dict) and nested_props:
                _check_props(nested_props, full_path)
            # Recurse into array items
            items = param_schema.get("items") or {}
            if isinstance(items, dict):
                item_props = items.get("properties") or {}
                if isinstance(item_props, dict) and item_props:
                    _check_props(item_props, f"{full_path}[]")

    _check_props(properties, "")
    return issues


# ---------------------------------------------------------------------------
# Check 77: anyof_null_should_be_optional
# ---------------------------------------------------------------------------


def _check_anyof_null_should_be_optional(
    tool_name: str, schema: Dict[str, Any]
) -> List[Issue]:
    """Check 77: anyof_null_should_be_optional — a parameter uses
    ``anyOf: [{type: "..."}, {type: "null"}]`` to express nullability, but
    the simpler form is to just declare the real type and make the parameter
    optional (not in ``required``).

    This pattern is extremely common in auto-generated schemas produced by
    Pydantic, FastAPI, and similar Python libraries that translate
    ``Optional[str]`` to ``anyOf: [{type: "string"}, {type: "null"}]``.

    In MCP tool calling, LLMs do not explicitly pass ``null`` for optional
    parameters — they simply omit them.  The null branch of the ``anyOf`` is
    therefore dead schema that wastes tokens on every request that includes
    this tool's definition.

    **Token cost example** — for a tool with 10 optional parameters of this
    form, every request carries ~150–200 extra tokens for null branches that
    are never used.

    **Fix**: Remove the ``{type: "null"}`` schema from ``anyOf``, inline the
    real type directly, and ensure the parameter is not in ``required``.

    Fires when:

    * A parameter has ``anyOf`` with exactly 2 entries, AND
    * One entry is ``{type: "null"}`` (the null branch), AND
    * The other entry declares a concrete ``type`` (string/integer/number/
      boolean/array/object)

    Does NOT fire for:

    * ``anyOf`` with 3+ entries (legitimate union types — keep those)
    * ``anyOf`` containing ``$ref`` entries (references need special handling)
    * The non-null schema lacks an explicit ``type`` (too complex to simplify)

    Examples::

        # flagged — anyOf with null is just Optional[string]
        {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "..."}

        # correct — just declare the type and leave it optional
        {"type": "string", "description": "..."}
    """
    issues: List[Issue] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    _CONCRETE_TYPES = {"string", "integer", "number", "boolean", "array", "object"}

    def _check_props(props: Dict[str, Any], path: str) -> None:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            full_path = f"{path}.{param_name}" if path else param_name
            anyof = param_schema.get("anyOf")
            if not isinstance(anyof, list) or len(anyof) != 2:
                pass  # only check exactly-2-schema anyOf
            else:
                null_schema = None
                real_schema = None
                for sub in anyof:
                    if not isinstance(sub, dict):
                        real_schema = None  # has non-dict, skip
                        break
                    if "$ref" in sub:
                        real_schema = None  # has $ref, skip
                        break
                    if sub.get("type") == "null":
                        null_schema = sub
                    elif sub.get("type") in _CONCRETE_TYPES:
                        real_schema = sub
                if null_schema is not None and real_schema is not None:
                    issues.append(Issue(
                        tool=tool_name,
                        severity="warn",
                        check="anyof_null_should_be_optional",
                        message=(
                            "param '{param}' uses anyOf with null to express nullability "
                            "(type='{real_type}' | null) — declare type '{real_type}' "
                            "directly and make the param optional (omit from required)."
                        ).format(
                            param=full_path,
                            real_type=real_schema.get("type", "?"),
                        ),
                    ))
            # Recurse into nested objects
            nested_props = param_schema.get("properties") or {}
            if isinstance(nested_props, dict) and nested_props:
                _check_props(nested_props, full_path)
            # Recurse into array items
            items = param_schema.get("items") or {}
            if isinstance(items, dict):
                item_props = items.get("properties") or {}
                if isinstance(item_props, dict) and item_props:
                    _check_props(item_props, f"{full_path}[]")

    _check_props(properties, "")
    return issues


# ---------------------------------------------------------------------------
# Check 78: name_uses_hyphen
# ---------------------------------------------------------------------------


def _check_tool_name_uses_hyphen(tool_name: str) -> Optional[Issue]:
    """Check 78 (tool): name_uses_hyphen — tool name contains a hyphen.

    MCP convention (and the broader API ecosystem) uses ``snake_case`` for
    tool names.  Hyphens create problems:

    * Many code generators map tool names to function/method names; a hyphen
      is invalid in most identifier syntaxes, requiring quoting or replacement.
    * HTTP header–style naming (``Content-Type``) bleeds into parameter names
      from auto-generated schemas.
    * Inconsistency with ``snake_case`` tools in the same server confuses
      models about which naming convention to expect.

    Note: hyphens are technically *allowed* by the JSON Schema spec and most
    LLM providers accept them, but they are a consistency and ergonomics smell.

    Fix: replace hyphens with underscores (``create-issue`` → ``create_issue``).
    """
    if '-' not in tool_name:
        return None
    fixed = tool_name.replace('-', '_')
    return Issue(
        tool=tool_name,
        severity="warn",
        check="name_uses_hyphen",
        message=(
            "tool name '{name}' uses hyphens; prefer snake_case "
            "(e.g., '{fixed}')."
        ).format(name=tool_name, fixed=fixed),
    )


# ---------------------------------------------------------------------------
# Check 94: param_name_is_reserved_word
# ---------------------------------------------------------------------------

# Python + JS keywords that commonly appear as param names and break codegen
_RESERVED_WORDS = frozenset({
    # Python keywords
    "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
    "or", "pass", "raise", "return", "try", "while", "with", "yield",
    # JS reserved words that overlap and are commonly seen
    "delete", "export", "extends", "function", "instanceof", "let",
    "new", "static", "super", "switch", "this", "throw", "typeof",
    "var", "void",
})


def _check_param_name_is_reserved_word(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 94: param_name_is_reserved_word — a parameter name is a Python
    or JavaScript reserved word (e.g., ``type``, ``class``, ``from``,
    ``import``).

    Reserved-word parameter names break code generators that map tool
    parameters to function arguments, require quoting in many contexts,
    and cause syntax errors in generated client code::

        # bad — "from" is a Python keyword
        {"from": {"type": "string", "description": "Sender email address."}}

        # good — unambiguous name
        {"from_address": {"type": "string", "description": "Sender email address."}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if param_name.lower() in _RESERVED_WORDS:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_is_reserved_word",
                message=(
                    "param name '{param}' is a reserved word — rename to "
                    "avoid conflicts in generated code (e.g., "
                    "'{param}_value' or '{param}_field')."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 102: tool_name_too_generic
# ---------------------------------------------------------------------------

_GENERIC_TOOL_NAMES = frozenset({
    "run", "execute", "process", "call", "invoke", "do", "handle",
    "perform", "trigger", "fire", "dispatch", "send", "submit",
    "action", "task", "job", "work", "operation",
})


def _check_tool_name_too_generic(tool_name: str) -> Optional[Issue]:
    """Check 102: tool_name_too_generic — tool name is a single generic verb
    that gives no information about what the tool actually does.

    Names like ``run``, ``execute``, ``process``, ``call``, ``invoke`` are
    meaningless without context.  The LLM cannot distinguish between tools
    with these names, and users reading the schema learn nothing about the
    tool's purpose::

        # bad — what does it run? what does it execute?
        "name": "run"
        "name": "execute"

        # good — verb + object pair
        "name": "run_tests"
        "name": "execute_query"

    Fires when the tool name (after stripping underscores/hyphens) is a
    single generic word with no additional qualifier.

    Severity: ``warn``.
    """
    normalized = tool_name.lower().strip("_-")
    # Check if the whole name (no underscores) is a generic verb
    if normalized in _GENERIC_TOOL_NAMES:
        return Issue(
            tool=tool_name,
            severity="warn",
            check="tool_name_too_generic",
            message=(
                "tool name '{name}' is too generic — add an object to "
                "clarify what it operates on (e.g., '{name}_query', "
                "'{name}_command')."
            ).format(name=tool_name),
        )
    return None


# ---------------------------------------------------------------------------
# Check 128: schema_type_not_object
# ---------------------------------------------------------------------------


def _check_schema_type_not_object(
    tool_name: str,
    schema: Dict[str, Any],
) -> Optional[Issue]:
    """Check 128: schema_type_not_object — the tool's ``inputSchema`` has a
    ``type`` field set to something other than ``"object"``.

    MCP tool input schemas must be JSON Schema objects (``type: "object"``).
    Setting the type to ``"string"``, ``"array"``, ``"integer"``, etc. is
    invalid — the schema describes the entire set of parameters, not a single
    value::

        # bad — inputSchema type must be "object"
        {"type": "string", "properties": {...}}
        {"type": "array", "items": {...}}

        # good — inputSchema is always an object
        {"type": "object", "properties": {...}}

    Severity: ``error``.
    """
    stype = schema.get("type")
    if stype is not None and stype != "object":
        return Issue(
            tool=tool_name,
            severity="error",
            check="schema_type_not_object",
            message=(
                "tool '{name}' inputSchema has type '{t}' — tool input "
                "schemas must have type 'object'."
            ).format(name=tool_name, t=stype),
        )
    return None


# Check 127: param_name_has_space
# ---------------------------------------------------------------------------


def _check_param_name_has_space(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 127: param_name_has_space — a parameter name contains a
    whitespace character (space, tab, newline).

    Whitespace in parameter names is technically valid JSON but causes
    issues in virtually every programming environment — it can't be used
    as a Python variable, JavaScript property without bracket notation, or
    CLI argument.  It almost always indicates a schema generation error::

        # bad — spaces in param names
        {"first name": {"type": "string", ...}}
        {"search query": {"type": "string", ...}}
        {"max results": {"type": "integer", ...}}

        # good — snake_case names
        {"first_name": {"type": "string", ...}}
        {"search_query": {"type": "string", ...}}
        {"max_results": {"type": "integer", ...}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if isinstance(param_name, str) and any(c.isspace() for c in param_name):
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="param_name_has_space",
                message=(
                    "param '{param}' contains whitespace — parameter names "
                    "cannot have spaces; use snake_case instead."
                ).format(param=param_name),
            ))
    return issues


# Check 126: name_ends_with_underscore
# ---------------------------------------------------------------------------


def _check_name_ends_with_underscore(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 126: name_ends_with_underscore — a tool name or parameter name
    ends with an underscore.

    Trailing underscores are a Python convention for avoiding naming conflicts
    with reserved words (``class_``, ``type_``, ``from_``).  In MCP tool
    schemas, this convention is unnecessary and looks odd — use a synonym or
    different name instead::

        # bad — Python reserved-word workaround
        tool name: class_, from_
        param name: type_, id_, format_

        # good — use a synonym or meaningful alternative
        tool name: classify, fetch_from
        param name: entity_type, record_id, output_format

    Severity: ``warn``.
    """
    issues = []

    if tool_name.endswith("_"):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="name_ends_with_underscore",
            message=(
                "tool name '{name}' ends with an underscore — this is a "
                "Python reserved-word workaround; use a synonym instead."
            ).format(name=tool_name),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name in properties:
            if param_name.endswith("_"):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="name_ends_with_underscore",
                    message=(
                        "param '{param}' ends with an underscore — this is "
                        "a Python reserved-word workaround; use a synonym "
                        "instead."
                    ).format(param=param_name),
                ))
    return issues


# Check 125: param_name_is_number
# ---------------------------------------------------------------------------


def _check_param_name_is_number(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 125: param_name_is_number — a parameter name is a numeric string
    (e.g., ``"0"``, ``"1"``, ``"42"``).

    Numeric parameter names are invalid in most tool schema implementations.
    They cannot be used as Python keyword arguments, are confusing for models,
    and indicate the schema was auto-generated from an array or incorrect
    source structure::

        # bad — numeric param names
        {"0": {"type": "string", ...}}
        {"1": {"type": "integer", ...}}
        {"42": {"type": "boolean", ...}}

        # good — named params
        {"first_arg": {"type": "string", ...}}
        {"count": {"type": "integer", ...}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if isinstance(param_name, str) and param_name.lstrip("-").isdigit():
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="param_name_is_number",
                message=(
                    "param '{param}' is a numeric name — use a descriptive "
                    "name; numeric params cannot be passed as keyword "
                    "arguments."
                ).format(param=param_name),
            ))
    return issues


# Check 124: param_name_starts_with_underscore
# ---------------------------------------------------------------------------


def _check_param_name_starts_with_underscore(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 124: param_name_starts_with_underscore — a parameter name starts
    with an underscore (``_id``, ``_type``, ``_internal``).

    Leading underscores indicate private or internal members in Python and
    JavaScript conventions.  They should not appear in MCP tool schema
    parameter names, which are part of the public API surface::

        # bad — leading underscore suggests internal detail
        {"_id": {"type": "string", ...}}
        {"_type": {"type": "string", ...}}
        {"_internal_flag": {"type": "boolean", ...}}

        # good — public names
        {"id": {"type": "string", ...}}
        {"type": {"type": "string", ...}}
        {"debug_mode": {"type": "boolean", ...}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if param_name.startswith("_"):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_starts_with_underscore",
                message=(
                    "param '{param}' starts with an underscore — "
                    "leading underscores suggest private/internal fields "
                    "and should not appear in public API schemas."
                ).format(param=param_name),
            ))
    return issues


# Check 123: param_name_has_double_underscore
# ---------------------------------------------------------------------------


def _check_param_name_has_double_underscore(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 123: param_name_has_double_underscore — a parameter name contains
    a double underscore (``__``).

    Double underscores are a Python convention for "dunder" (double underscore)
    names like ``__init__`` and ``__class__``.  In MCP tool schemas, parameter
    names with ``__`` usually indicate a Python implementation detail leaking
    into the API surface.  Use single underscores::

        # bad — Python implementation details leaking
        {"__type": {"type": "string", ...}}
        {"field__name": {"type": "string", ...}}
        {"__meta__": {"type": "object", ...}}

        # good — clean names
        {"type": {"type": "string", ...}}
        {"field_name": {"type": "string", ...}}
        {"meta": {"type": "object", ...}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if "__" in param_name:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_has_double_underscore",
                message=(
                    "param '{param}' contains double underscores — this "
                    "leaks Python implementation conventions into the API; "
                    "use single underscores."
                ).format(param=param_name),
            ))
    return issues


# Check 122: required_param_not_in_properties
# ---------------------------------------------------------------------------


def _check_required_param_not_in_properties(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 122: required_param_not_in_properties — the ``required`` array
    lists param names that are not present in ``properties``.

    When a param name appears in ``required`` but not in ``properties``,
    the schema is internally inconsistent.  Clients cannot resolve the
    constraint — the parameter has no type or description, but is required::

        # bad — "format" in required but not in properties
        {
          "properties": {"query": {"type": "string"}},
          "required": ["query", "format"]
        }

        # good — all required params are defined
        {
          "properties": {"query": {"type": "string"}, "format": {"type": "string"}},
          "required": ["query", "format"]
        }

    Severity: ``error``.
    """
    issues = []
    required = schema.get("required")
    properties = schema.get("properties")

    if not isinstance(required, list) or not isinstance(properties, dict):
        return issues

    for param_name in required:
        if isinstance(param_name, str) and param_name not in properties:
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="required_param_not_in_properties",
                message=(
                    "param '{param}' is in 'required' but not in 'properties' — "
                    "add the param to properties or remove it from required."
                ).format(param=param_name),
            ))
    return issues


# Check 121: param_name_all_uppercase
# ---------------------------------------------------------------------------


def _check_param_name_all_uppercase(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 121: param_name_all_uppercase — a parameter name is written in
    ALL_CAPS or ALL_UPPERCASE_WITH_UNDERSCORES style.

    MCP parameter names must be lowercase snake_case.  ALL_CAPS names are a
    convention from environment variables and constants, not tool schemas::

        # bad — ALL_CAPS param names
        {"API_KEY": {"type": "string", ...}}
        {"MAX_RETRIES": {"type": "integer", ...}}
        {"USER_ID": {"type": "string", ...}}

        # good — snake_case param names
        {"api_key": {"type": "string", ...}}
        {"max_retries": {"type": "integer", ...}}
        {"user_id": {"type": "string", ...}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        # A name is ALL_CAPS if it has at least 2 chars, contains at least
        # one letter, all letters are uppercase, and no lowercase letters
        if (
            len(param_name) >= 2
            and any(c.isalpha() for c in param_name)
            and not any(c.islower() for c in param_name)
            and all(c.isupper() or c == "_" or c.isdigit() for c in param_name)
        ):
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="param_name_all_uppercase",
                message=(
                    "param '{param}' is ALL_CAPS — MCP parameter names must "
                    "be lowercase snake_case (e.g., '{lower}')."
                ).format(
                    param=param_name,
                    lower=param_name.lower(),
                ),
            ))
    return issues


# Check 120: required_not_array
# ---------------------------------------------------------------------------


def _check_required_not_array(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 120: required_not_array — the ``required`` field in an input
    schema (or nested object) is not an array.

    JSON Schema mandates that ``required`` is an array of strings.  Setting
    it to ``true``, ``false``, a single string, or any non-array value is a
    schema correctness error that will silently be ignored by most validators::

        # bad — required must be an array
        {"required": true}
        {"required": "param_name"}
        {"required": {"param_name": true}}

        # good — required is an array
        {"required": ["param_name", "other_param"]}

    Severity: ``error``.
    """
    issues = []

    def _check_obj(obj: Any, context: str) -> None:
        if not isinstance(obj, dict):
            return
        req = obj.get("required")
        if req is not None and not isinstance(req, list):
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="required_not_array",
                message=(
                    "'{ctx}' has 'required: {val}' — 'required' must be "
                    "an array of param name strings."
                ).format(ctx=context, val=repr(req)),
            ))
        # Recurse into nested properties
        props = obj.get("properties")
        if isinstance(props, dict):
            for pname, pschema in props.items():
                if isinstance(pschema, dict):
                    nested_req = pschema.get("required")
                    if nested_req is not None and not isinstance(nested_req, list):
                        issues.append(Issue(
                            tool=tool_name,
                            severity="error",
                            check="required_not_array",
                            message=(
                                "param '{param}' has 'required: {val}' — "
                                "'required' must be an array of strings."
                            ).format(param=pname, val=repr(nested_req)),
                        ))

    _check_obj(schema, tool_name)
    return issues


# Check 119: description_has_json_example
# ---------------------------------------------------------------------------

_JSON_EXAMPLE_RE = re.compile(
    r'\{[^{}]{5,}\}'        # inline JSON object (at least 5 chars between braces)
    r'|\[[^\[\]]{5,}\]',    # or inline JSON array (at least 5 chars between brackets)
)


def _check_description_has_json_example(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str,
) -> List[Issue]:
    """Check 119: description_has_json_example — a tool or parameter
    description contains an inline JSON example (object or array literal).

    Embedding JSON examples in descriptions wastes tokens (descriptions
    are included in every tool call) and mixes documentation with schema.
    Use the ``examples`` field instead::

        # bad — JSON example bloating the description
        {"query": {"description": "Search query. Example: {\"q\": \"cats\", \"limit\": 10}"}}
        {"tags": {"description": "List of tags, e.g. [\"foo\", \"bar\", \"baz\"]"}}

        # good — example in dedicated field
        {"query": {"description": "Search query.", "examples": ["{\"q\": \"cats\"}"]}}

    Severity: ``warn``.
    """
    issues = []
    desc = _get_tool_description(obj, fmt)
    if desc and _JSON_EXAMPLE_RE.search(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_has_json_example",
            message=(
                "tool '{name}' description contains an inline JSON example — "
                "move it to the 'examples' field to save tokens."
            ).format(name=tool_name),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if isinstance(pdesc, str) and pdesc and _JSON_EXAMPLE_RE.search(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_has_json_example",
                    message=(
                        "param '{param}' description contains an inline JSON "
                        "example — move it to the 'examples' field to save tokens."
                    ).format(param=param_name),
                ))
    return issues


# Check 118: description_uses_first_person
# ---------------------------------------------------------------------------

_FIRST_PERSON_RE = re.compile(
    r"\bI(?:'ll|'ve|'m|'d)?\b"
    r"|\bI\s+(?:will|can|am|was|have|had|would|should|could|might|must|shall)\b"
    r"|\bmy\s+(?:tool|function|api|endpoint|server|service|assistant|implementation)\b",
    re.IGNORECASE,
)


def _check_description_uses_first_person(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str,
) -> List[Issue]:
    """Check 118: description_uses_first_person — a tool or parameter
    description uses first-person language (``"I will"``, ``"I can"``,
    ``"I'll"``, ``"my tool"``).

    Tools are utilities, not agents.  First-person language in a tool
    description mixes the tool's role with the calling agent's identity
    and is confusing when the model reads the schema::

        # bad — first-person voice
        "I will search the database for matching records."
        "I can create, update, or delete items."
        "My tool returns a JSON object."

        # good — imperative or neutral voice
        "Search the database for matching records."
        "Create, update, or delete items."
        "Returns a JSON object."

    Severity: ``warn``.
    """
    issues = []
    desc = _get_tool_description(obj, fmt)
    if desc and _FIRST_PERSON_RE.search(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_uses_first_person",
            message=(
                "tool '{name}' description uses first-person language — "
                "use imperative or neutral voice instead."
            ).format(name=tool_name),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if isinstance(pdesc, str) and pdesc and _FIRST_PERSON_RE.search(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_uses_first_person",
                    message=(
                        "param '{param}' description uses first-person language — "
                        "use imperative or neutral voice instead."
                    ).format(param=param_name),
                ))
    return issues


# Check 117: param_name_too_generic
# ---------------------------------------------------------------------------

_GENERIC_PARAM_NAMES = frozenset({
    "data", "value", "val", "input", "output", "result", "response",
    "payload", "content", "info", "item", "object", "obj", "stuff",
    "thing", "param", "arg", "argument", "variable", "var",
})


def _check_param_name_too_generic(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 117: param_name_too_generic — a parameter name is a single generic
    word that gives no context about what value should be passed.

    Generic names like ``data``, ``value``, ``input``, ``output``,
    ``result``, ``payload``, ``content``, ``info`` force the model to rely
    entirely on the description to understand the parameter.  A descriptive
    name (even one word) is always better::

        # bad — generic names
        {"data": {"type": "string", "description": "The SQL query to run."}}
        {"value": {"type": "number", "description": "The temperature in Celsius."}}
        {"input": {"type": "object", "description": "The request body."}}

        # good — descriptive names
        {"query": {"type": "string", "description": "The SQL query to run."}}
        {"temperature_c": {"type": "number", "description": "Temperature in Celsius."}}
        {"request_body": {"type": "object", "description": "The request body."}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if param_name.lower() in _GENERIC_PARAM_NAMES:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_too_generic",
                message=(
                    "param '{param}' has a generic name — use a descriptive "
                    "name that conveys what value should be passed."
                ).format(param=param_name),
            ))
    return issues


# Check 116: enum_has_empty_value
# ---------------------------------------------------------------------------


def _check_enum_has_empty_value(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 116: enum_has_empty_value — an enum array contains an empty
    string ``""`` as a valid value.

    An empty-string enum value is almost always a schema generation error.
    If the intent is to allow "no selection", use an optional param with no
    default, or add a meaningful sentinel value like ``"none"``::

        # bad — empty string is an opaque sentinel
        {"status": {"type": "string", "enum": ["active", "inactive", ""]}}

        # good — meaningful values only
        {"status": {"type": "string", "enum": ["active", "inactive", "none"]}}
        # or: make it optional with no enum restriction

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum_vals = param_schema.get("enum")
        if not isinstance(enum_vals, list):
            continue
        if "" in enum_vals:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="enum_has_empty_value",
                message=(
                    "param '{param}' enum contains an empty string value — "
                    "use a named sentinel like 'none' or make the param "
                    "optional instead."
                ).format(param=param_name),
            ))
    return issues


# Check 115: param_type_is_null
# ---------------------------------------------------------------------------


def _check_param_type_is_null(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 115: param_type_is_null — a parameter's ``type`` is ``"null"``
    (or ``["null"]``), making the parameter useless.

    A parameter that can only be ``null`` can never carry a meaningful value.
    It should either be removed or given a real type.  This is almost always
    a copy-paste or schema generation error::

        # bad — param can only ever be null
        {"format": {"type": "null", "description": "Output format."}}
        {"options": {"type": ["null"], "description": "Extra options."}}

        # good — give it a real type or remove it
        {"format": {"type": "string", "description": "Output format."}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        ptype = param_schema.get("type")
        if ptype == "null" or ptype == ["null"]:
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="param_type_is_null",
                message=(
                    "param '{param}' has type 'null' — a null-only param "
                    "can never hold a meaningful value; remove it or give "
                    "it a real type."
                ).format(param=param_name),
            ))
    return issues


# Check 114: name_starts_with_uppercase
# ---------------------------------------------------------------------------


def _check_name_starts_with_uppercase(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 114: name_starts_with_uppercase — a tool name or parameter name
    starts with an uppercase letter (PascalCase or SCREAMING_SNAKE_CASE style).

    MCP tool and parameter names must be lowercase.  Names like ``GetUser``,
    ``CreateItem``, or ``MAX_RETRIES`` violate the naming convention and may
    cause compatibility issues with clients that validate name format::

        # bad — starts with uppercase
        tool name: GetUser, CreateItem, MAX_RETRIES
        param name: UserId, Format

        # good — starts with lowercase
        tool name: get_user, create_item
        param name: user_id, format

    Severity: ``error``.
    """
    issues = []

    if tool_name and tool_name[0].isupper():
        issues.append(Issue(
            tool=tool_name,
            severity="error",
            check="name_starts_with_uppercase",
            message=(
                "tool name '{name}' starts with an uppercase letter — "
                "MCP tool names must be lowercase snake_case."
            ).format(name=tool_name),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name in properties:
            if param_name and param_name[0].isupper():
                issues.append(Issue(
                    tool=tool_name,
                    severity="error",
                    check="name_starts_with_uppercase",
                    message=(
                        "param '{param}' starts with an uppercase letter — "
                        "MCP parameter names must be lowercase."
                    ).format(param=param_name),
                ))
    return issues


# Check 113: name_uses_camelcase
# ---------------------------------------------------------------------------

_CAMELCASE_RE = re.compile(r"[a-z][A-Z]")


def _check_name_uses_camelcase(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str,
) -> List[Issue]:
    """Check 113: name_uses_camelcase — a tool or parameter name uses
    camelCase style instead of the MCP-recommended snake_case.

    MCP tool and parameter names should use ``snake_case`` (lowercase words
    joined by underscores).  camelCase names (``userId``, ``maxRetries``,
    ``createdAt``) are idiomatic in JavaScript/Java but inconsistent with
    the MCP naming convention::

        # bad — camelCase names
        tool name: getUserById
        param name: userId, maxRetries, createdAt

        # good — snake_case names
        tool name: get_user_by_id
        param name: user_id, max_retries, created_at

    Severity: ``warn``.
    """
    issues = []

    # Check tool name
    if _CAMELCASE_RE.search(tool_name):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="name_uses_camelcase",
            message=(
                "tool name '{name}' uses camelCase — use snake_case instead "
                "(e.g., '{snake}')."
            ).format(
                name=tool_name,
                snake=re.sub(r"([a-z])([A-Z])", r"\1_\2", tool_name).lower(),
            ),
        ))

    # Check param names
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name in properties:
            if _CAMELCASE_RE.search(param_name):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="name_uses_camelcase",
                    message=(
                        "param '{param}' uses camelCase — use snake_case "
                        "instead (e.g., '{snake}')."
                    ).format(
                        param=param_name,
                        snake=re.sub(r"([a-z])([A-Z])", r"\1_\2", param_name).lower(),
                    ),
                ))
    return issues


# Check 112: enum_duplicate_values
# ---------------------------------------------------------------------------


def _check_enum_duplicate_values(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 112: enum_duplicate_values — an enum array contains duplicate
    values.

    JSON Schema requires enum values to be unique.  Duplicate entries are a
    copy-paste error and may confuse model inference about valid choices::

        # bad — "active" appears twice
        {"status": {"type": "string", "enum": ["active", "inactive", "active"]}}

        # good — unique values
        {"status": {"type": "string", "enum": ["active", "inactive", "archived"]}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_props(props: Dict[str, Any], parent: str) -> None:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            enum_vals = param_schema.get("enum")
            if not isinstance(enum_vals, list):
                continue
            # Deduplicate using JSON representation (handles None, bool, etc.)
            serialized = [json.dumps(v, sort_keys=True) for v in enum_vals]
            seen: set = set()
            dupes = []
            for s in serialized:
                if s in seen:
                    dupes.append(s)
                seen.add(s)
            if dupes:
                label = "{}/{}".format(parent, param_name) if parent else param_name
                issues.append(Issue(
                    tool=tool_name,
                    severity="error",
                    check="enum_duplicate_values",
                    message=(
                        "param '{param}' enum contains duplicate value(s): "
                        "{dupes} — enum values must be unique."
                    ).format(
                        param=label,
                        dupes=", ".join(sorted(set(dupes))),
                    ),
                ))
            # Recurse into nested object properties
            nested = param_schema.get("properties")
            if isinstance(nested, dict):
                _check_props(nested, param_name)

    _check_props(properties, "")
    return issues


# Check 111: param_name_ends_with_type
# ---------------------------------------------------------------------------

_TYPE_SUFFIX_RE = re.compile(
    r"_(?:string|integer|boolean|number|float|array|object|dict|list)$",
    re.IGNORECASE,
)


def _check_param_name_ends_with_type(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 111: param_name_ends_with_type — a parameter name ends with a
    type suffix such as ``_string``, ``_integer``, ``_boolean``, ``_array``,
    ``_object``, etc.

    Like Hungarian notation prefixes, type suffixes redundantly encode the
    type in the parameter name.  The type is already declared in the
    schema's ``type`` field::

        # bad — type redundantly encoded in the suffix
        {"query_string": {"type": "string", ...}}
        {"active_boolean": {"type": "boolean", ...}}
        {"limit_integer": {"type": "integer", ...}}
        {"tags_array": {"type": "array", ...}}

        # good — descriptive names; type is in the schema
        {"query": {"type": "string", ...}}
        {"active": {"type": "boolean", ...}}
        {"limit": {"type": "integer", ...}}
        {"tags": {"type": "array", ...}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if _TYPE_SUFFIX_RE.search(param_name):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_ends_with_type",
                message=(
                    "param '{param}' name ends with a type suffix — the "
                    "type is already in the schema; use a descriptive name "
                    "without the type suffix."
                ).format(param=param_name),
            ))
    return issues


# Check 110: param_name_starts_with_type
# ---------------------------------------------------------------------------

_TYPE_PREFIX_RE = re.compile(
    r"^(?:str|int|bool|num|float|arr|list|dict|obj)_",
    re.IGNORECASE,
)


def _check_param_name_starts_with_type(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 110: param_name_starts_with_type — a parameter name starts with a
    type prefix such as ``str_``, ``int_``, ``bool_``, ``arr_``, etc.

    The type is already declared in the schema's ``type`` field.  Prefixing the
    name with the type is a C-style Hungarian notation convention that adds noise
    without meaning::

        # bad — type redundantly encoded in the name
        {"str_query": {"type": "string", ...}}
        {"int_limit": {"type": "integer", ...}}
        {"bool_active": {"type": "boolean", ...}}
        {"arr_tags": {"type": "array", ...}}

        # good — descriptive names; type is in the schema
        {"query": {"type": "string", ...}}
        {"limit": {"type": "integer", ...}}
        {"active": {"type": "boolean", ...}}
        {"tags": {"type": "array", ...}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if _TYPE_PREFIX_RE.match(param_name):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_starts_with_type",
                message=(
                    "param '{param}' name starts with a type prefix — the "
                    "type is already in the schema; use a descriptive name "
                    "without the type prefix."
                ).format(param=param_name),
            ))
    return issues


# Check 109: description_has_parenthetical_type
# ---------------------------------------------------------------------------

_PAREN_TYPE_RE = re.compile(
    r"\(\s*(?:type[:\s]+)?(?:string|integer|number|boolean|array|object|null)\s*\)",
    re.IGNORECASE,
)


def _check_description_has_parenthetical_type(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 109: description_has_parenthetical_type — a parameter
    description contains the type in parentheses (e.g., ``"(string)"``,
    ``"(boolean)"``, ``"(type: integer)"``).

    The type is already declared in the schema's ``type`` field; repeating
    it in the description wastes tokens and goes out of sync if the type
    changes::

        # bad — type duplicated in description
        {"limit": {"type": "integer", "description": "(integer) Max results."}}
        {"active": {"type": "boolean", "description": "Flag (boolean) to enable."}}

        # good — description adds value, not redundant type info
        {"limit": {"type": "integer", "description": "Max results to return."}}
        {"active": {"type": "boolean", "description": "Enable or disable the feature."}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        if _PAREN_TYPE_RE.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="description_has_parenthetical_type",
                message=(
                    "param '{param}' description contains the type in "
                    "parentheses — the type is already in the schema; "
                    "remove it from the description."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 108: array_items_empty_schema
# ---------------------------------------------------------------------------


def _check_array_items_empty_schema(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 108: array_items_empty_schema — an array parameter has an empty
    ``items`` schema (``{}``), meaning the array can contain any value.

    An empty items schema provides no type or constraint information.
    The LLM must guess what to put in the array.  Even a simple
    ``"items": {"type": "string"}`` is better than nothing::

        # bad — items can be anything
        {"tags": {"type": "array", "items": {}}}

        # good — explicit item type
        {"tags": {"type": "array", "items": {"type": "string"}}}

    Severity: ``warn``.
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
        if items == {}:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="array_items_empty_schema",
                message=(
                    "param '{param}' has items: {{}} — specify the item type "
                    "(e.g., items: {{type: string}})."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 107: object_no_props_additional_false
# ---------------------------------------------------------------------------


def _check_object_no_props_additional_false(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 107: object_no_props_additional_false — a parameter has
    ``type: object``, no ``properties`` defined, and
    ``additionalProperties: false``.

    This combination means only the empty object ``{}`` is valid — no keys
    are allowed because no properties are defined and extra properties are
    forbidden.  This is almost certainly a mistake::

        # bad — only {} is valid
        {"config": {"type": "object", "additionalProperties": false}}

        # good — define what keys are allowed
        {"config": {"type": "object",
                    "properties": {"timeout": {"type": "integer"}},
                    "additionalProperties": false}}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "object":
            continue
        if "properties" in param_schema:
            continue  # has properties, fine
        if param_schema.get("additionalProperties") is False:
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="object_no_props_additional_false",
                message=(
                    "param '{param}' has type:object, no properties, and "
                    "additionalProperties:false — only {{}} is valid; define "
                    "the allowed properties or remove additionalProperties:false."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 106: description_ends_abruptly
# ---------------------------------------------------------------------------


def _check_description_ends_abruptly(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str = "mcp",
) -> List[Issue]:
    """Check 106: description_ends_abruptly — a tool or param description
    ends with a punctuation character that suggests the text was cut off
    (trailing comma, colon, semi-colon, or conjunction word).

    These patterns indicate the description was truncated before being
    published::

        # bad — description appears cut off
        "description": "Get user data including name, email, and"
        "description": "Supported formats: json, xml,"

        # good — complete sentence
        "description": "Get user data including name and email."

    Severity: ``warn``.
    """
    issues = []

    def _is_abrupt(text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.rstrip()
        if not t:
            return False
        # Ends with comma, colon, or semi-colon
        if t[-1] in (",", ";", ":"):
            return True
        # Ends with conjunction word
        words = t.lower().split()
        if words and words[-1] in ("and", "or", "but", "the", "a", "an", "for",
                                    "with", "of", "in", "to", "by", "from"):
            return True
        return False

    desc = _get_tool_description(obj, fmt)
    if _is_abrupt(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_ends_abruptly",
            message=(
                "tool '{name}' description appears to end abruptly "
                "(ends with '{ending}') — check for truncation."
            ).format(name=tool_name, ending=(desc or "").rstrip()[-10:].strip()),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if _is_abrupt(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_ends_abruptly",
                    message=(
                        "param '{param}' description appears to end abruptly "
                        "— check for truncation."
                    ).format(param=param_name),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 105: schema_has_definitions
# ---------------------------------------------------------------------------


def _check_schema_has_definitions(
    tool_name: str,
    obj: Dict[str, Any],
    fmt: str = "mcp",
) -> Optional[Issue]:
    """Check 105: schema_has_definitions — the tool's input schema includes
    a ``definitions`` or ``$defs`` section.

    In MCP context, tool schemas are self-contained blobs.  A
    ``definitions`` section implies ``$ref`` usage, which is unresolvable
    (see Check 101).  Even without ``$ref``, the definitions section adds
    tokens that the LLM will see but never directly use::

        # bad — definitions are wasted tokens
        {"type": "object", "properties": {...}, "definitions": {"User": {...}}}

        # good — inline every schema; omit definitions
        {"type": "object", "properties": {...}}

    Severity: ``warn``.
    """
    # Get the inputSchema object for the tool
    schema: Any = None
    if fmt == "mcp":
        schema = obj.get("inputSchema")
    elif fmt == "openai":
        schema = obj.get("parameters")
    elif fmt in ("anthropic", "google"):
        schema = obj.get("input_schema")

    if not isinstance(schema, dict):
        return None

    if "definitions" in schema or "$defs" in schema:
        return Issue(
            tool=tool_name,
            severity="warn",
            check="schema_has_definitions",
            message=(
                "tool '{name}' inputSchema has a 'definitions'/'$defs' "
                "section — in MCP context $refs are unresolvable; inline "
                "schemas and remove the definitions block."
            ).format(name=tool_name),
        )
    return None


# ---------------------------------------------------------------------------
# Check 104: enum_values_inconsistent_case
# ---------------------------------------------------------------------------


def _check_enum_values_inconsistent_case(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 104: enum_values_inconsistent_case — an enum array has string
    values with inconsistent casing (e.g., ``["Yes", "no", "MAYBE"]``).

    Mixed-case enum values are a data quality issue.  The LLM must guess
    the exact case for each value; mismatches cause validation failures.
    Pick one casing convention and apply it consistently::

        # bad — mixed casing
        {"status": {"type": "string", "enum": ["Active", "inactive", "PENDING"]}}

        # good — consistent lowercase
        {"status": {"type": "string", "enum": ["active", "inactive", "pending"]}}

    Only fires when the enum has 2+ string values with at least 2 distinct
    case patterns.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum_vals = param_schema.get("enum")
        if not isinstance(enum_vals, list) or len(enum_vals) < 2:
            continue
        str_vals = [v for v in enum_vals if isinstance(v, str) and len(v) > 1]
        if len(str_vals) < 2:
            continue

        def _case_pattern(s: str) -> str:
            if s == s.upper():
                return "UPPER"
            if s == s.lower():
                return "lower"
            if s[0].isupper() and s[1:] == s[1:].lower():
                return "Title"
            return "mixed"

        patterns = {_case_pattern(v) for v in str_vals}
        if len(patterns) > 1:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="enum_values_inconsistent_case",
                message=(
                    "param '{param}' enum values have inconsistent casing "
                    "({patterns}) — pick one convention (lowercase recommended)."
                ).format(
                    param=param_name,
                    patterns="/".join(sorted(patterns)),
                ),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 103: string_minlength_zero
# ---------------------------------------------------------------------------


def _check_string_minlength_zero(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 103: string_minlength_zero — a string parameter has an explicit
    ``minLength: 0``.

    The default value of ``minLength`` in JSON Schema is already ``0``, so
    ``minLength: 0`` is redundant and wastes schema tokens.  It also
    misses an opportunity to actually validate non-empty input::

        # bad — redundant; minLength: 0 is the default
        {"query": {"type": "string", "minLength": 0}}

        # good — remove the redundant field, or set a meaningful minimum
        {"query": {"type": "string"}}
        {"query": {"type": "string", "minLength": 1}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "string":
            continue
        if param_schema.get("minLength") == 0:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="string_minlength_zero",
                message=(
                    "param '{param}' has minLength: 0 which is the default "
                    "— remove it (redundant) or set minLength: 1 to prohibit "
                    "empty strings."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 101: param_uses_schema_ref
# ---------------------------------------------------------------------------


def _check_param_uses_schema_ref(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 101: param_uses_schema_ref — a parameter or nested property uses
    a JSON Schema ``$ref`` to reference another schema definition.

    MCP tool schemas are self-contained JSON blobs passed inline to the LLM.
    There is no base URI or ``$defs``/``definitions`` context to resolve
    ``$ref`` against.  Unresolved references cause LLMs to see the literal
    ``{"$ref": "#/definitions/..."}`` instead of the actual schema, which
    provides no type or constraint information::

        # bad — $ref is unresolvable in MCP context
        {"user": {"$ref": "#/definitions/User"}}

        # good — inline the schema
        {"user": {"type": "object", "properties": {"id": {"type": "integer"}}}}

    Fires on top-level parameters and nested properties that contain a
    ``$ref`` key.

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_obj(obj: Any, path: str) -> None:
        if not isinstance(obj, dict):
            return
        if "$ref" in obj:
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="param_uses_schema_ref",
                message=(
                    "param '{path}' uses $ref which is unresolvable in MCP "
                    "context — inline the schema definition instead."
                ).format(path=path),
            ))
            return  # don't recurse into $ref object
        for key, val in obj.items():
            if isinstance(val, dict):
                _check_obj(val, f"{path}.{key}")
            elif isinstance(val, list):
                for item in val:
                    _check_obj(item, path)

    for param_name, param_schema in properties.items():
        _check_obj(param_schema, param_name)

    return issues


# ---------------------------------------------------------------------------
# Check 100: param_accepts_secret_no_format
# ---------------------------------------------------------------------------

_SECRET_PARAM_RE = re.compile(
    r"\bpassword\b|\bpasswd\b"
    r"|\bsecret\b"
    r"|\bapi[_\-]?key\b"
    r"|\baccess[_\-]?token\b"
    r"|\bauth[_\-]?token\b"
    r"|\bprivate[_\-]?key\b"
    r"|\bapi[_\-]?secret\b"
    r"|\bclient[_\-]?secret\b",
    re.IGNORECASE,
)


def _check_param_accepts_secret_no_format(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 100: param_accepts_secret_no_format — a parameter whose name
    implies it holds a secret value (password, api_key, secret, access_token,
    private_key, etc.) is declared as ``type: string`` but does not have
    ``"format": "password"``.

    The ``password`` format signals to UIs and developer tools that the
    value should be masked/redacted.  Without it, secrets may be logged
    or displayed in plain text::

        # bad — no masking hint
        {"api_key": {"type": "string", "description": "Your API key."}}

        # good — format signals that the value should be masked
        {"api_key": {"type": "string", "format": "password",
                     "description": "Your API key."}}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "string":
            continue
        if param_schema.get("format") == "password":
            continue
        if _SECRET_PARAM_RE.search(param_name):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_accepts_secret_no_format",
                message=(
                    "param '{param}' looks like a secret value but is missing "
                    "\"format\": \"password\" — add this so UIs and tools "
                    "know to mask the value."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 99: description_has_internal_path
# ---------------------------------------------------------------------------

_INTERNAL_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])"
    r"(?:/(?:var|etc|home|usr|opt|tmp|srv|root|proc|sys|run)/\w"   # Unix absolute
    r"|[A-Z]:\\\\(?:Program Files|Users|Windows|System32)\b"       # Windows absolute
    r"|~/"                                                          # home-relative
    r")",
    re.IGNORECASE,
)


def _check_description_has_internal_path(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str = "mcp",
) -> List[Issue]:
    """Check 99: description_has_internal_path — a tool or param description
    contains a server filesystem path (e.g., ``/var/data/``, ``/etc/config``,
    ``C:\\\\Windows\\\\``).

    Filesystem paths in tool descriptions expose server internals to the
    LLM and any user who inspects the schema.  They also break when the
    deployment environment changes.  Paths belong in server configuration,
    not in schema descriptions::

        # bad — leaks server layout
        "description": "Path to config file (default: /etc/myapp/config.yaml)."

        # good — describe what the param does, not where it lives
        "description": "Path to a YAML configuration file."

    Severity: ``warn``.
    """
    issues = []

    desc = _get_tool_description(obj, fmt)
    if isinstance(desc, str) and _INTERNAL_PATH_RE.search(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_has_internal_path",
            message=(
                "tool '{name}' description contains a server filesystem path "
                "— paths leak server internals and break across deployments."
            ).format(name=tool_name),
        ))

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if isinstance(pdesc, str) and _INTERNAL_PATH_RE.search(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_has_internal_path",
                    message=(
                        "param '{param}' description contains a server "
                        "filesystem path — paths leak server internals."
                    ).format(param=param_name),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 98: description_says_see_docs
# ---------------------------------------------------------------------------

_SEE_DOCS_RE = re.compile(
    r"\bsee\s+(the\s+)?docs?\b"
    r"|\bsee\s+(the\s+)?documentation\b"
    r"|\bsee\s+(the\s+)?readme\b"
    r"|\brefer\s+to\s+(the\s+)?docs?\b"
    r"|\brefer\s+to\s+(the\s+)?documentation\b"
    r"|\bcheck\s+(the\s+)?docs?\b"
    r"|\bcheck\s+(the\s+)?documentation\b"
    r"|\bfor\s+details?,?\s+see\b"
    r"|\bmore\s+info(rmation)?\s+at\b",
    re.IGNORECASE,
)


def _check_description_says_see_docs(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str = "mcp",
) -> List[Issue]:
    """Check 98: description_says_see_docs — a tool or param description
    defers to external documentation ("see docs", "see the documentation",
    "refer to README", etc.) instead of describing the parameter inline.

    LLMs cannot follow documentation links at inference time.  Descriptions
    that say "see docs" provide no actionable information.  The relevant
    details must be in the schema itself::

        # bad — LLM cannot access the docs
        "description": "Filter options. See the documentation for details."

        # good — self-contained description
        "description": "Comma-separated list of status filters (open, closed, draft)."

    Severity: ``warn``.
    """
    issues = []

    # Tool description
    desc = _get_tool_description(obj, fmt)
    if isinstance(desc, str) and _SEE_DOCS_RE.search(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_says_see_docs",
            message=(
                "tool '{name}' description defers to external docs — "
                "LLMs cannot follow links at inference time; put the "
                "relevant details in the description."
            ).format(name=tool_name),
        ))

    # Param descriptions
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if isinstance(pdesc, str) and _SEE_DOCS_RE.search(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_says_see_docs",
                    message=(
                        "param '{param}' description defers to external docs "
                        "— put the relevant details inline."
                    ).format(param=param_name),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 97: array_max_items_zero
# ---------------------------------------------------------------------------


def _check_array_max_items_zero(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 97: array_max_items_zero — an array parameter has ``maxItems: 0``.

    When ``maxItems`` is 0, only the empty array ``[]`` is valid.  This is
    almost always a mistake (a typo, a leftover from a template, or a
    misunderstanding of the field).  If an empty array is truly the only
    acceptable value, use ``const: []`` instead::

        # bad — only [] is valid, but expressed confusingly
        {"tags": {"type": "array", "maxItems": 0}}

        # good — express intent with const
        {"tags": {"type": "array", "const": []}}

    Severity: ``error``.
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
        max_items = param_schema.get("maxItems")
        if max_items == 0:
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="array_max_items_zero",
                message=(
                    "param '{param}' has maxItems: 0 — only [] is valid; "
                    "this is almost certainly a mistake. Use const: [] if "
                    "intentional."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 96: description_has_todo_marker
# ---------------------------------------------------------------------------

_TODO_MARKER_RE = re.compile(
    r"\bTODO\b|\bFIXME\b|\bHACK\b|\bXXX\b|\bNOTE:\s",
    # Note: case-sensitive — TODO/FIXME/HACK/XXX are developer convention
)


def _check_description_has_todo_marker(
    tool_name: str,
    obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str = "mcp",
) -> List[Issue]:
    """Check 96: description_has_todo_marker — a tool or param description
    contains an inline developer marker (TODO, FIXME, HACK, XXX, NOTE:).

    These markers indicate the description was not finished.  Unlike
    Check 70 (which catches descriptions that are *only* a placeholder),
    this check fires when a marker appears *within* a longer string::

        # bad — incomplete description with embedded marker
        "description": "Get user data. TODO: add pagination details."
        "description": "FIXME: this doesn't handle errors yet."

        # good — complete description
        "description": "Get paginated user data."

    Severity: ``warn``.
    """
    issues = []

    # Tool description
    desc = _get_tool_description(obj, fmt)
    if isinstance(desc, str) and _TODO_MARKER_RE.search(desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_has_todo_marker",
            message=(
                "tool '{name}' description contains a developer marker "
                "(TODO/FIXME/HACK/XXX/NOTE) — finish the description before "
                "publishing."
            ).format(name=tool_name),
        ))

    # Param descriptions
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            pdesc = param_schema.get("description", "")
            if isinstance(pdesc, str) and _TODO_MARKER_RE.search(pdesc):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_has_todo_marker",
                    message=(
                        "param '{param}' description contains a developer "
                        "marker (TODO/FIXME/HACK/XXX/NOTE) — finish the "
                        "description before publishing."
                    ).format(param=param_name),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 95: description_has_version_info
# ---------------------------------------------------------------------------

_DESC_VERSION_RE = re.compile(
    r"\bapi\s+v\d+\b"           # API v1, API v2
    r"|\bv\d+\.\d+\b"           # v1.0, v2.3
    r"|\bversion\s+\d+\b"       # version 2, version 3
    r"|\bapi[\s-]version\b",    # api-version, api version
    re.IGNORECASE,
)


def _check_description_has_version_info(
    tool_name: str,
    obj: Dict[str, Any],
    fmt: str = "mcp",
) -> Optional[Issue]:
    """Check 95: description_has_version_info — a tool description mentions
    an API version (e.g., "API v2", "v1.0", "version 2").

    Version strings in tool descriptions go stale when APIs change and
    clutter context with information that belongs in the server manifest
    or deployment URL, not in individual tool definitions::

        # bad — version info will drift
        "description": "Get user data using the Users API v2."

        # good — omit API version from individual tool descriptions
        "description": "Get user data."

    Severity: ``warn``.
    """
    desc = _get_tool_description(obj, fmt)
    if not isinstance(desc, str):
        return None
    if _DESC_VERSION_RE.search(desc):
        return Issue(
            tool=tool_name,
            severity="warn",
            check="description_has_version_info",
            message=(
                "tool '{name}' description contains an API version string — "
                "version info belongs in server metadata, not tool "
                "descriptions."
            ).format(name=tool_name),
        )
    return None


# ---------------------------------------------------------------------------
# Check 93: tool_name_contains_version
# ---------------------------------------------------------------------------

_VERSION_IN_NAME_RE = re.compile(
    r"(?:^|_)v\d+(?:_|$)"          # _v1_, _v2, v1_, etc.
    r"|(?:^|_)\d{4}(?:_|$)"        # _2024_, _2023 etc.
    r"|_v\d+\.\d+"                  # _v1.2
    r"|_version_?\d+",              # _version1, _version_2
    re.IGNORECASE,
)


def _check_tool_name_contains_version(tool_name: str) -> Optional[Issue]:
    """Check 93: tool_name_contains_version — tool name contains a version
    number (e.g., ``get_user_v2``, ``list_items_v1``, ``create_record_2024``).

    Version numbers in tool names leak implementation details into the
    schema and break backwards compatibility for clients that hard-code
    tool names.  Versioning belongs in the server metadata (manifest
    version, endpoint URL), not individual tool names.

    Fix: remove the version suffix and use server-level versioning instead.
    """
    if _VERSION_IN_NAME_RE.search(tool_name):
        return Issue(
            tool=tool_name,
            severity="warn",
            check="tool_name_contains_version",
            message=(
                "tool name '{name}' contains a version number — versioning "
                "belongs in server metadata, not tool names."
            ).format(name=tool_name),
        )
    return None


def _check_param_name_uses_hyphen(
    tool_name: str, schema: Dict[str, Any]
) -> List[Issue]:
    """Check 78 (param): name_uses_hyphen — a parameter name contains a hyphen.

    Same rationale as the tool-level check: hyphens in parameter names conflict
    with ``snake_case`` conventions, cause identifier-mapping problems in code
    generators, and produce inconsistent schemas.

    Fix: replace hyphens with underscores (``user-id`` → ``user_id``).
    """
    issues: List[Issue] = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_props(props: Dict[str, Any], path: str) -> None:
        for param_name, param_schema in props.items():
            full_path = f"{path}.{param_name}" if path else param_name
            if '-' in param_name:
                fixed = param_name.replace('-', '_')
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="name_uses_hyphen",
                    message=(
                        "param '{param}' uses hyphens; prefer snake_case "
                        "(e.g., '{fixed}')."
                    ).format(param=full_path, fixed=fixed),
                ))
            if not isinstance(param_schema, dict):
                continue
            # Recurse into nested objects
            nested_props = param_schema.get("properties") or {}
            if isinstance(nested_props, dict) and nested_props:
                _check_props(nested_props, full_path)

    _check_props(properties, "")
    return issues


# ---------------------------------------------------------------------------
# Check 79: description_has_example
# ---------------------------------------------------------------------------

import re as _re

_EXAMPLE_PATTERNS = _re.compile(
    r"\be\.g\.[\s,]"          # e.g., or e.g. followed by space
    r"|\bfor example\b"       # for example
    r"|\bexample:\s"          # example: <value>
    r"|\(e\.g[.,]"            # (e.g. or (e.g,
    r"|\bsuch as\b"           # such as
    r"|\blike\s+['\"`]"       # like 'foo' or like "bar"
    r"|\be\.g\.$",            # e.g. at end of string
    _re.IGNORECASE,
)


def _check_description_has_example(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 79: description_has_example — a parameter description contains
    an inline example (``e.g.``, ``for example``, ``example:``, ``such as``,
    ``like 'x'``) instead of using the JSON Schema ``examples`` field.

    Inline examples inside descriptions waste tokens, mix semantics with
    usage hints, and become stale.  The ``examples`` array is the correct
    place for them::

        # bad — 60+ tokens consumed describing the example
        {"description": "The city name, e.g. 'London' or 'New York'"}

        # good — description is semantic, example is structured
        {"description": "City to look up", "examples": ["London", "New York"]}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        if _EXAMPLE_PATTERNS.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="description_has_example",
                message=(
                    "param '{param}' description contains an inline example "
                    "(e.g. / for example / such as); use the 'examples' field "
                    "instead."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 80: description_lists_enum_values
# ---------------------------------------------------------------------------

_LISTS_ENUM_PATTERNS = _re.compile(
    r"\bone of\b[^.]{0,60}['\"`]"          # one of 'x', 'y'
    r"|\bmust be\b[^.]{0,40}['\"`]"         # must be 'x' or 'y'
    r"|\bvalid values?\b"                    # valid value(s): ...
    r"|\bpossible values?\b"                 # possible value(s): ...
    r"|\baccepted values?\b"                 # accepted value(s): ...
    r"|\bsupported values?\b"               # supported value(s): ...
    r"|\bcan be\b[^.]{0,40}['\"`]"          # can be 'x' or 'y'
    r"|\beither\b[^.]{0,40}['\"`]"          # either 'x' or 'y'
    r"|\ballowed values?\b"                  # allowed value(s): ...
    r"|\buse ['\"`][a-z_]+['\"`] for\b",    # use 'x' for ..., use 'y' for ...
    _re.IGNORECASE,
)


def _check_description_lists_enum_values(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 80: description_lists_enum_values — a parameter description
    explicitly enumerates valid values instead of using a JSON Schema
    ``enum`` field.

    When a parameter only accepts a finite set of values, the schema should
    declare an ``enum`` so the model receives a formal constraint::

        # bad — valid values buried in prose
        {"type": "string", "description": "Sort order. One of 'asc', 'desc'."}

        # good — explicit constraint, no token waste
        {"type": "string", "enum": ["asc", "desc"], "description": "Sort order."}

    This check fires on non-enum params whose descriptions contain phrases
    like "one of", "must be", "valid values:", "possible values:", "can be",
    "either", "allowed values:", etc.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        # Skip if an enum is already declared
        if "enum" in param_schema:
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        if _LISTS_ENUM_PATTERNS.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="description_lists_enum_values",
                message=(
                    "param '{param}' description lists valid values in prose "
                    "(one of / must be / valid values / etc.); declare an "
                    "'enum' field instead."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 81: param_description_says_ignored
# ---------------------------------------------------------------------------

_SAYS_IGNORED_PATTERNS = _re.compile(
    r"\bignored\b"                           # this field is ignored
    r"|\bnot used\b"                         # not used
    r"|\bnot currently used\b"              # not currently used
    r"|\bcurrently unused\b"                 # currently unused
    r"|\bunused\b"                           # unused
    r"|\breserved for future\b"             # reserved for future use
    r"|\breserved\b"                         # reserved
    r"|\bno[\-\s]op\b"                       # no-op / noop
    r"|\bdeprecated,? (?:use|replaced)\b",  # deprecated, use X instead
    _re.IGNORECASE,
)


def _check_param_description_says_ignored(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 81: param_description_says_ignored — a parameter description
    says the parameter is ignored, unused, reserved, or a no-op.

    If a parameter is truly unused, it should be removed from the schema.
    Including dead parameters:

    * Wastes tokens on every call that sends the schema.
    * Confuses models into passing values that have no effect.
    * Makes the schema harder to understand.

    Common patterns::

        # should be removed
        {"name": "format", "description": "Ignored. Always returns JSON."}
        {"name": "version", "description": "Reserved for future use."}
        {"name": "legacy_id", "description": "Not used. Kept for backwards compatibility."}

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        if _SAYS_IGNORED_PATTERNS.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_description_says_ignored",
                message=(
                    "param '{param}' description says it is ignored/unused/"
                    "reserved; remove unused params from the schema."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 82: enum_boolean_string
# ---------------------------------------------------------------------------

_BOOL_STRING_SETS = (
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"on", "off"}),
    frozenset({"1", "0"}),
    frozenset({"enabled", "disabled"}),
)


def _check_enum_boolean_string(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 82: enum_boolean_string — a string parameter's enum contains
    only boolean-like values (``true``/``false``, ``yes``/``no``,
    ``on``/``off``, ``1``/``0``, ``enabled``/``disabled``).

    These parameters should use ``type: boolean`` instead::

        # bad — boolean disguised as string enum
        {"type": "string", "enum": ["true", "false"]}
        {"type": "string", "enum": ["yes", "no"]}

        # good — use the actual boolean type
        {"type": "boolean"}

    Using string booleans forces consumers to handle both boolean and string
    representations, increases parsing complexity, and wastes tokens.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "string":
            continue
        enum_vals = param_schema.get("enum")
        if not isinstance(enum_vals, list) or len(enum_vals) != 2:
            continue
        # Normalize to lowercase strings
        lower_set = frozenset(
            str(v).lower() for v in enum_vals if isinstance(v, (str, int, bool))
        )
        if any(lower_set == bool_set for bool_set in _BOOL_STRING_SETS):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="enum_boolean_string",
                message=(
                    "param '{param}' uses enum {vals} as a string type; "
                    "use 'type: boolean' instead."
                ).format(param=param_name, vals=sorted(enum_vals)),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 83: param_nullable_field
# ---------------------------------------------------------------------------


def _check_param_nullable_field(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 83: param_nullable_field — a parameter uses the OpenAPI 3.0
    ``nullable: true`` extension keyword instead of proper JSON Schema syntax.

    ``nullable`` is not a JSON Schema keyword.  It is an OpenAPI 3.0
    extension.  JSON Schema validators ignore it; most tool-calling runtimes
    ignore it.  The correct JSON Schema patterns for nullable types are:

    * ``{"type": ["string", "null"]}``  (JSON Schema draft-07+)
    * ``{"anyOf": [{"type": "string"}, {"type": "null"}]}`` (handled by Check 77)

    Common source: auto-generated schemas from OpenAPI → MCP converters that
    do not translate ``nullable`` to proper JSON Schema.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_props(props: Dict[str, Any]) -> None:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            if param_schema.get("nullable") is True:
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="param_nullable_field",
                    message=(
                        "param '{param}' uses 'nullable: true' (OpenAPI extension); "
                        "use type: [\"{t}\", \"null\"] or anyOf instead."
                    ).format(
                        param=param_name,
                        t=param_schema.get("type", "string"),
                    ),
                ))
            # Recurse into nested object properties
            nested_props = param_schema.get("properties")
            if isinstance(nested_props, dict):
                _check_props(nested_props)

    _check_props(properties)
    return issues


# ---------------------------------------------------------------------------
# Check 84: schema_has_x_field
# ---------------------------------------------------------------------------


def _check_schema_has_x_field(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 84: schema_has_x_field — the inputSchema or a parameter
    definition contains OpenAPI extension fields (``x-*`` keys).

    ``x-`` prefixed fields are OpenAPI extension conventions.  They are not
    part of JSON Schema and are ignored by JSON Schema validators and most
    tool-calling runtimes (Anthropic, OpenAI, Google, MCP).  Including them
    wastes tokens on every request.

    Common sources: OpenAPI → MCP schema converters that copy extension
    fields verbatim.

    Examples of fields that trigger this check:

    * ``x-order``, ``x-hidden``, ``x-deprecated``
    * ``x-example``, ``x-display-name``, ``x-nullable``

    Severity: ``warn``.
    """
    issues = []

    # Check inputSchema-level x- fields
    x_keys_schema = [k for k in schema if isinstance(k, str) and k.startswith("x-")]
    if x_keys_schema:
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="schema_has_x_field",
            message=(
                "inputSchema contains OpenAPI extension field(s) {keys}; "
                "remove x-* fields — they are ignored by JSON Schema and "
                "tool-calling runtimes."
            ).format(keys=sorted(x_keys_schema)),
        ))

    # Check parameter-level x- fields
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    def _check_props(props: Dict[str, Any]) -> None:
        for param_name, param_schema in props.items():
            if not isinstance(param_schema, dict):
                continue
            x_keys = [k for k in param_schema if isinstance(k, str) and k.startswith("x-")]
            if x_keys:
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="schema_has_x_field",
                    message=(
                        "param '{param}' has OpenAPI extension field(s) {keys}; "
                        "remove x-* fields."
                    ).format(param=param_name, keys=sorted(x_keys)),
                ))
            # Recurse into nested objects
            nested_props = param_schema.get("properties")
            if isinstance(nested_props, dict):
                _check_props(nested_props)

    _check_props(properties)
    return issues


# ---------------------------------------------------------------------------
# Check 85: default_violates_minimum
# ---------------------------------------------------------------------------


def _check_default_violates_minimum(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 85: default_violates_minimum — a parameter's ``default`` value
    falls outside the declared ``minimum``/``maximum`` range.

    This is a schema correctness bug: if the default is invalid under the
    schema's own constraints, the schema is self-contradictory::

        # bad — minimum is 1 but default is 0 (invalid!)
        {"type": "integer", "minimum": 1, "default": 0}

        # bad — maximum is 100 but default is 200 (invalid!)
        {"type": "number", "maximum": 100, "default": 200}

        # good — default is within the allowed range
        {"type": "integer", "minimum": 1, "maximum": 100, "default": 10}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") not in ("integer", "number"):
            continue
        default = param_schema.get("default")
        if default is None or not isinstance(default, (int, float)):
            continue
        minimum = param_schema.get("minimum")
        maximum = param_schema.get("maximum")
        if minimum is not None and isinstance(minimum, (int, float)):
            if default < minimum:
                issues.append(Issue(
                    tool=tool_name,
                    severity="error",
                    check="default_violates_minimum",
                    message=(
                        "param '{param}' has default={default} but minimum={min}; "
                        "the default value violates the minimum constraint."
                    ).format(param=param_name, default=default, min=minimum),
                ))
                continue  # Don't double-report for same param
        if maximum is not None and isinstance(maximum, (int, float)):
            if default > maximum:
                issues.append(Issue(
                    tool=tool_name,
                    severity="error",
                    check="default_violates_minimum",
                    message=(
                        "param '{param}' has default={default} but maximum={max}; "
                        "the default value violates the maximum constraint."
                    ).format(param=param_name, default=default, max=maximum),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 86: param_name_single_char
# ---------------------------------------------------------------------------


def _check_param_name_single_char(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 86: param_name_single_char — a parameter name is a single
    character.

    Single-character parameter names (``n``, ``q``, ``k``, ``x``) are
    opaque.  Models infer parameter meaning from the name; a one-letter name
    forces the model to rely entirely on the description — and descriptions
    are often short or missing.  Clear names improve accuracy::

        # bad — opaque; what is 'n', 'q'?
        {"n": {"type": "integer"}, "q": {"type": "string"}}

        # good — self-documenting
        {"limit": {"type": "integer"}, "query": {"type": "string"}}

    Exception: single-character names that are standard abbreviations in
    their domain (``x``, ``y``, ``z`` in a coordinate system or math
    context) are acceptable, but such cases are rare and can be suppressed
    manually.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name in properties:
        if not isinstance(param_name, str):
            continue
        if len(param_name) == 1:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_name_single_char",
                message=(
                    "param '{param}' has a single-character name; use a "
                    "descriptive name instead (e.g., 'query', 'limit', 'offset')."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 87: allof_single_schema
# ---------------------------------------------------------------------------


def _check_allof_single_schema(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 87: allof_single_schema — a parameter or inputSchema uses
    ``allOf``, ``oneOf``, or ``anyOf`` with exactly one element.

    A combiner keyword with a single schema is a no-op and should be
    simplified::

        # bad — allOf with one element adds no meaning, wastes tokens
        {"allOf": [{"type": "string", "minLength": 1}]}

        # good — just use the schema directly
        {"type": "string", "minLength": 1}

    Common source: code generators that wrap every schema in ``allOf`` or
    ``oneOf`` for extensibility.

    This check fires on the inputSchema and each parameter that uses a
    single-element combiner.

    Severity: ``warn``.
    """
    issues = []

    def _check_schema_node(name: str, node: Dict[str, Any], is_param: bool = False) -> None:
        for combiner in ("allOf", "oneOf", "anyOf"):
            val = node.get(combiner)
            if isinstance(val, list) and len(val) == 1:
                label = "param '{}'".format(name) if is_param else "inputSchema"
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="allof_single_schema",
                    message=(
                        "{label} uses {combiner} with a single schema; "
                        "use the schema directly instead."
                    ).format(label=label, combiner=combiner),
                ))

    # Check inputSchema itself (top level)
    _check_schema_node(tool_name, schema, is_param=False)

    # Check each parameter
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for param_name, param_schema in properties.items():
            if isinstance(param_schema, dict):
                _check_schema_node(param_name, param_schema, is_param=True)

    return issues


# ---------------------------------------------------------------------------
# Check 88: enum_has_duplicates
# ---------------------------------------------------------------------------


def _check_enum_has_duplicates(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 88: enum_has_duplicates — a parameter's ``enum`` array contains
    duplicate values.

    Duplicate values in an ``enum`` are a schema correctness bug.  JSON Schema
    (draft-06+) states that enum values SHOULD be unique.  Duplicates most
    likely result from copy-paste errors or merging lists without deduplication::

        # bad — 'active' appears twice
        {"enum": ["active", "inactive", "active"]}

        # good — unique values
        {"enum": ["active", "inactive"]}

    Severity: ``error``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum_vals = param_schema.get("enum")
        if not isinstance(enum_vals, list):
            continue
        # Convert to hashable form to detect duplicates
        try:
            hashable = [json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for v in enum_vals]
        except (TypeError, ValueError):
            continue
        if len(hashable) != len(set(map(str, hashable))):
            from collections import Counter
            counts = Counter(str(v) for v in hashable)
            dupes = [v for v, c in counts.items() if c > 1]
            issues.append(Issue(
                tool=tool_name,
                severity="error",
                check="enum_has_duplicates",
                message=(
                    "param '{param}' enum has duplicate values: {dupes}."
                ).format(param=param_name, dupes=dupes),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 89: description_has_html
# ---------------------------------------------------------------------------

_HTML_TAG_RE = _re.compile(r"<(/?\w[\w.-]*)\s*[^>]*>", _re.IGNORECASE)


def _check_description_has_html(
    tool_name: str,
    raw_obj: Dict[str, Any],
    schema: Dict[str, Any],
    fmt: str = "mcp",
) -> List[Issue]:
    """Check 89: description_has_html — a tool or parameter description
    contains HTML tags.

    Tool-calling descriptions are plain text.  HTML tags render as literal
    angle-bracket sequences in LLM contexts and waste tokens::

        # bad — HTML renders as literal text, not formatted output
        {"description": "Fetch the <b>resource</b>. See <a href='...'>docs</a>."}

        # good — plain prose
        {"description": "Fetch the resource."}

    Severity: ``warn``.
    """
    issues = []
    tool_desc = _get_tool_description(raw_obj, fmt) or ""
    if _HTML_TAG_RE.search(tool_desc):
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="description_has_html",
            message="tool description contains HTML tags; use plain text instead.",
        ))

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if isinstance(desc, str) and _HTML_TAG_RE.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="description_has_html",
                message=(
                    "param '{param}' description contains HTML tags; "
                    "use plain text instead."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 90: description_starts_with_param_name
# ---------------------------------------------------------------------------


def _check_description_starts_with_param_name(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 90: description_starts_with_param_name — a parameter description
    begins with the parameter name (e.g., ``"limit: Max results"`` for param
    ``limit``).

    When a description starts with the parameter name followed by a colon,
    dash, or similar separator, the name is being repeated.  The model
    already sees the parameter name separately — duplicating it in the
    description wastes tokens and adds no information::

        # bad — "limit" is already the param name
        {"limit": {"description": "limit: Maximum number of results."}}
        {"query": {"description": "query - The search string."}}

        # good — description adds meaning without restating the name
        {"limit": {"description": "Maximum number of results to return."}}
        {"query": {"description": "Full-text search query."}}

    Fires when the description starts with the param name (case-insensitive)
    followed by a colon, dash, em-dash, or space+colon.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        # Check if description starts with param_name followed by separator
        lower_name = param_name.lower().replace("_", "").replace("-", "")
        lower_desc = desc.lower().lstrip()
        # Strip underscores/hyphens from description start too
        desc_start = _re.sub(r"[\s_-]", "", lower_desc[:len(param_name) + 5])
        if desc_start.startswith(lower_name):
            # Check what follows the name in the original description
            rest = desc.lstrip()[len(param_name):].lstrip()
            if rest and rest[0] in (":", "-", "–", "—", "="):
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="description_starts_with_param_name",
                    message=(
                        "param '{param}' description starts with the param "
                        "name ('{name}: ...')  — the name is already shown "
                        "separately; start with what the value represents."
                    ).format(param=param_name, name=param_name),
                ))
    return issues


# ---------------------------------------------------------------------------
# Check 91: string_type_describes_json
# ---------------------------------------------------------------------------

_JSON_STRING_RE = _re.compile(
    r"\bjson[\s\-]?string\b"
    r"|\bjson[\s\-]?encoded\b"
    r"|\bjson[\s\-]?formatted\b"
    r"|\bjson[\s\-]?serialized\b"
    r"|\bstringified[\s\-]?json\b"
    r"|\bpassed?\s+as\s+json\b"
    r"|\bencoded\s+as\s+json\b",
    _re.IGNORECASE,
)


def _check_string_type_describes_json(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 91: string_type_describes_json — a parameter has ``type: string``
    but its description describes it as a JSON value ("JSON string",
    "JSON-encoded", "JSON-formatted", "stringified JSON", etc.).

    Using ``type: string`` to pass structured data defeats the purpose of
    JSON Schema.  LLMs cannot validate the structure, serialisation is
    error-prone, and token cost increases.  Use ``type: object`` or
    ``type: array`` and define the schema properly::

        # bad — structure hidden in a string
        {"filters": {"type": "string",
                      "description": "A JSON string of filter conditions."}}

        # good — structure expressed in schema
        {"filters": {"type": "object", "properties": {...}}}

    Fires when param type is exactly ``"string"`` (not array/object)
    and description matches a JSON-string phrase.

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "string":
            continue
        desc = param_schema.get("description", "")
        if not isinstance(desc, str) or not desc:
            continue
        if _JSON_STRING_RE.search(desc):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="string_type_describes_json",
                message=(
                    "param '{param}' is type string but description says "
                    "it contains JSON — use type object/array with a proper "
                    "schema instead of encoding structure as a string."
                ).format(param=param_name),
            ))
    return issues


# ---------------------------------------------------------------------------
# Check 92: object_param_no_properties
# ---------------------------------------------------------------------------


def _check_object_param_no_properties(
    tool_name: str,
    schema: Dict[str, Any],
) -> List[Issue]:
    """Check 92: object_param_no_properties — a parameter has ``type: object``
    but no ``properties`` field is defined.

    An unstructured object gives the LLM no guidance on what keys to provide.
    Without a properties schema the LLM must guess, leading to hallucinated
    keys, missing required fields, and no validation.  Even a minimal
    properties definition is better than none::

        # bad — completely opaque; LLM must guess the structure
        {"config": {"type": "object", "description": "Configuration options."}}

        # good — structure declared
        {"config": {"type": "object",
                    "properties": {"timeout": {"type": "integer"}},
                    "required": ["timeout"]}}

    Does not fire when ``additionalProperties`` is set (the author is
    intentionally allowing a free-form map) or when the param also defines
    ``oneOf``/``anyOf``/``allOf`` (which imply structure).

    Severity: ``warn``.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "object":
            continue
        # Skip if already has properties
        if "properties" in param_schema:
            continue
        # Skip if additionalProperties is explicitly set (free-form map)
        if "additionalProperties" in param_schema:
            continue
        # Skip if complex composition used
        if any(k in param_schema for k in ("oneOf", "anyOf", "allOf")):
            continue
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="object_param_no_properties",
            message=(
                "param '{param}' has type object but no properties defined "
                "— add a properties schema so the model knows what keys to "
                "provide."
            ).format(param=param_name),
        ))
    return issues


# ---------------------------------------------------------------------------
# Check 71: schema_has_title_field
# ---------------------------------------------------------------------------


def _check_schema_has_title_field(tool_name: str, schema: Dict[str, Any], fmt: str = "mcp") -> List[Issue]:
    """Check 71: schema_has_title_field — the inputSchema or a parameter
    definition contains a ``title`` field.

    In tool-calling schemas, the ``title`` JSON Schema keyword is redundant:

    * At the **schema level**: the tool's ``name`` already identifies it.
    * At the **parameter level**: the parameter name already serves as the
      identifier; ``description`` provides the prose explanation.

    The ``title`` field is valid JSON Schema and useful in UI form generators
    (e.g., json-schema-form), but tool-calling LLMs do not use it.  Including
    it wastes tokens on every request that carries the schema.

    Common cause: auto-generated schemas from OpenAPI specs (where ``title``
    is a standard field at every level).

    Fires when:

    * ``inputSchema.title`` is present, OR
    * Any top-level parameter definition has a ``title`` key

    Examples::

        # flagged — schema-level title
        {"inputSchema": {"type": "object", "title": "Search parameters", ...}}

        # flagged — param-level title
        {"query": {"type": "string", "title": "Query string", "description": "..."}}

        # correct — no title fields
        {"inputSchema": {"type": "object", "properties": {...}}}
    """
    issues = []

    # Schema-level title — skip for json_schema format where title IS the tool name
    if isinstance(schema, dict) and "title" in schema and fmt != "json_schema":
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="schema_has_title_field",
            message=(
                "inputSchema has a 'title' field ('{title}') — redundant with the tool "
                "name; remove it to reduce token cost."
            ).format(title=schema.get("title", "")),
        ))

    # Param-level titles
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if not isinstance(properties, dict):
        return issues
    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if "title" in param_schema:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="schema_has_title_field",
                message=(
                    "param '{param}' has a 'title' field ('{title}') — redundant with the "
                    "param name; remove it to reduce token cost."
                ).format(param=param_name, title=param_schema.get("title", "")),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 70: description_is_placeholder
# ---------------------------------------------------------------------------

_PLACEHOLDER_DESCRIPTION_RE = re.compile(
    r'^(n/?a|none|todo|tbd|description|placeholder|undefined|no\s+description\.?|tba|coming\s+soon|fill\s+in|to\s+be\s+added|example\s+description)$',
    re.IGNORECASE,
)
"""Pattern matching placeholder descriptions that were never filled in."""


def _check_description_is_placeholder(name: str, obj: Dict[str, Any], fmt: str) -> List[Issue]:
    """Check 70: description_is_placeholder — tool or param description is a
    known placeholder value that was never filled in.

    These appear in auto-generated or carelessly maintained schemas where the
    description field was left as a template default.

    Fires on tool descriptions that exactly match (case-insensitive):

    * ``N/A``, ``NA``
    * ``None``
    * ``TODO``, ``TBD``, ``TBA``
    * ``description`` (the word itself)
    * ``placeholder``
    * ``undefined``
    * ``No description``
    * ``Coming soon``
    * ``Fill in``
    * ``To be added``
    * ``Example description``

    Also fires on parameter descriptions that match the same set.

    Examples::

        # flagged — placeholder tool description
        {"description": "TODO", "inputSchema": {...}}
        {"description": "N/A", "inputSchema": {...}}

        # flagged — placeholder param description
        {"query": {"type": "string", "description": "placeholder"}}

        # correct — actual description
        {"description": "Search documents by keyword."}
    """
    issues = []

    # Check tool-level description
    desc = _get_tool_description(obj, fmt)
    if desc and isinstance(desc, str) and _PLACEHOLDER_DESCRIPTION_RE.match(desc.strip()):
        issues.append(Issue(
            tool=name,
            severity="warn",
            check="description_is_placeholder",
            message=(
                "tool description '{desc}' is a placeholder — write an actual "
                "description that explains what the tool does."
            ).format(desc=desc.strip()),
        ))

    # Check param descriptions
    schema = _get_tool_schema(obj, fmt)
    if schema and isinstance(schema, dict):
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for param_name, param_schema in properties.items():
                if not isinstance(param_schema, dict):
                    continue
                param_desc = param_schema.get("description", "")
                if param_desc and isinstance(param_desc, str) and _PLACEHOLDER_DESCRIPTION_RE.match(param_desc.strip()):
                    issues.append(Issue(
                        tool=name,
                        severity="warn",
                        check="description_is_placeholder",
                        message=(
                            "param '{param}' description '{desc}' is a placeholder — "
                            "write an actual description."
                        ).format(param=param_name, desc=param_desc.strip()),
                    ))

    return issues


# ---------------------------------------------------------------------------
# Check 69: contradictory_min_max
# ---------------------------------------------------------------------------

_MIN_MAX_PAIRS = [
    ("minimum", "maximum"),
    ("minLength", "maxLength"),
    ("minItems", "maxItems"),
    ("exclusiveMinimum", "exclusiveMaximum"),
]


def _check_contradictory_min_max(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 69: contradictory_min_max — parameter has ``minimum > maximum``,
    ``minLength > maxLength``, or ``minItems > maxItems``.

    This is a schema correctness bug: the allowed range is empty — no valid
    value can satisfy both constraints simultaneously.  The model may produce
    a value that passes the intent check but fails schema validation, leading
    to runtime errors.

    Fires when any of the following are violated for a parameter:

    * ``minimum > maximum``
    * ``minLength > maxLength``
    * ``minItems > maxItems``
    * ``exclusiveMinimum > exclusiveMaximum``

    Examples::

        # flagged — minimum exceeds maximum (empty range)
        {"type": "integer", "minimum": 100, "maximum": 10}

        # flagged — minLength exceeds maxLength
        {"type": "string", "minLength": 50, "maxLength": 10}

        # correct
        {"type": "integer", "minimum": 1, "maximum": 100}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        for min_key, max_key in _MIN_MAX_PAIRS:
            if min_key not in param_schema or max_key not in param_schema:
                continue
            try:
                min_val = float(param_schema[min_key])
                max_val = float(param_schema[max_key])
            except (TypeError, ValueError):
                continue
            if min_val > max_val:
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="contradictory_min_max",
                    message=(
                        "param '{param}' has {mn}={mn_val} > {mx}={mx_val} — "
                        "the allowed range is empty; no valid value can satisfy both constraints."
                    ).format(
                        param=param_name,
                        mn=min_key, mn_val=param_schema[min_key],
                        mx=max_key, mx_val=param_schema[max_key],
                    ),
                ))

    return issues


# ---------------------------------------------------------------------------
# Check 68: const_param_should_be_removed
# ---------------------------------------------------------------------------


def _check_const_param_should_be_removed(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 68: const_param_should_be_removed — parameter uses the JSON Schema
    ``const`` keyword, meaning it always has a fixed value.

    The model can never provide a different value for a ``const`` parameter.
    The server should hardcode this value instead of exposing it as a parameter.
    Exposing it:

    * **Wastes tokens** — the model reads a param definition it cannot influence.
    * **Wastes model attention** — the model may waste reasoning about a param
      that has a single valid value.
    * **Suggests auto-generated schema** — often comes from tools that auto-export
      API specs without pruning fixed-value fields.

    Note: Check 44 (``enum_single_const``) catches the ``enum: ["value"]``
    anti-pattern (single-value enum should use ``const``).  This check catches
    the complementary case: where ``const`` is correctly used syntactically but
    should be removed from the public schema entirely.

    Fires when any top-level parameter contains a ``const`` key.

    Examples::

        # flagged — const param is a fixed value the model can't change
        "api_version": {"type": "string", "const": "v2"}
        "format": {"type": "string", "const": "json"}

        # correct — remove the parameter; hardcode the value server-side
        (no parameter at all)
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if "const" in param_schema:
            const_val = param_schema["const"]
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="const_param_should_be_removed",
                message=(
                    "param '{param}' uses 'const: {val!r}' — the model cannot "
                    "provide a different value; hardcode it server-side and remove "
                    "the parameter from the schema."
                ).format(param=param_name, val=const_val),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 67: enum_default_not_in_enum
# ---------------------------------------------------------------------------


def _check_enum_default_not_in_enum(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 67: enum_default_not_in_enum — parameter has both an ``enum`` and a
    ``default``, but the default value is not listed in the enum.

    This is a schema correctness bug: the declared default can never be a
    valid value because it lies outside the allowed set.

    Common causes:

    * Copy-paste error: the default was not updated after the enum was narrowed.
    * Typo: e.g., default is ``"asc"`` but enum requires ``"ascending"``.
    * Case mismatch: default ``"None"`` (string) vs enum ``[null]``.
    * Sentinel value: default ``"auto"`` used as a magic sentinel, not declared
      in the enum.

    Fires when:

    * A parameter has ``"enum"`` (non-empty list), AND
    * The parameter has a ``"default"`` field, AND
    * The default value is **not** in the enum list

    Examples::

        # flagged — default 'asc' not in enum
        {"type": "string", "enum": ["ascending", "descending"], "default": "asc"}

        # flagged — default null not in enum (use null as enum value to allow it)
        {"type": "string", "enum": ["active", "inactive"], "default": null}

        # correct — default is in enum
        {"type": "string", "enum": ["asc", "desc"], "default": "asc"}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        enum = param_schema.get("enum")
        if not isinstance(enum, list) or not enum:
            continue
        if "default" not in param_schema:
            continue
        default = param_schema["default"]
        if default not in enum:
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="enum_default_not_in_enum",
                message=(
                    "param '{param}' default {default!r} is not in its enum {enum!r} — "
                    "the default can never be a valid value."
                ).format(param=param_name, default=default, enum=enum),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 66: param_description_says_required
# ---------------------------------------------------------------------------

_SAYS_REQUIRED_RE = re.compile(
    r'^(\(required\)\s*|required\s*[:\-–]\s*|required\s+field\s*[:\-–]\s*)',
    re.IGNORECASE,
)
"""Pattern matching 'Required:' / '(required)' labels at the start of param descriptions."""


def _check_param_description_says_required(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 66: param_description_says_required — param description starts with a
    'Required' label.

    Two cases, both are smells:

    * **Required param** says "required" in its description: redundant — the
      ``required`` array already communicates this.
    * **Optional param** says "required" in its description: contradictory — the
      description contradicts the schema structure.

    The label conveys no information to the model beyond what the schema already
    encodes (or mis-encodes), and wastes tokens on every call that includes
    the tool definition.

    Fires when a param description starts with:

    * ``"Required: ..."`` / ``"Required - ..."``
    * ``"(Required) ..."``
    * ``"Required field: ..."``

    Does not fire when "required" appears mid-description (e.g., "The ID
    required for authentication") — only the label pattern is flagged.

    Examples::

        # flagged — 'Required' prefix is redundant with the required array
        "user_id": {"description": "Required: the user's ID", ...}
        "query":   {"description": "(Required) Search query string", ...}

        # correct — description goes straight to the point
        "user_id": {"description": "The user's ID", ...}
        "query":   {"description": "Search query string", ...}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        description = param_schema.get("description", "")
        if not description or not isinstance(description, str):
            continue
        if _SAYS_REQUIRED_RE.match(description.strip()):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="param_description_says_required",
                message=(
                    "param '{param}' description starts with 'Required' — "
                    "the schema's 'required' array already encodes this; "
                    "remove the label and let the schema structure speak."
                ).format(param=param_name),
            ))

    return issues


# ---------------------------------------------------------------------------
# Check 65: description_says_deprecated
# ---------------------------------------------------------------------------

_DEPRECATED_RE = re.compile(
    r'\b(deprecated|do\s+not\s+use|will\s+be\s+removed|being\s+removed|obsolete|no\s+longer\s+supported)\b',
    re.IGNORECASE,
)
"""Pattern matching deprecation language in tool descriptions."""


def _check_description_says_deprecated(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 65: description_says_deprecated — tool description signals the tool
    is deprecated, obsolete, or should not be used.

    When a description says "deprecated", "do not use", "will be removed", etc.,
    the schema is announcing that this tool should not exist.  The fix is to
    remove the tool — not to document it as deprecated.

    A deprecation label in the schema is actively harmful:

    * The model may still call the tool if it appears to satisfy the task —
      the word "deprecated" does not prevent calls.
    * It pollutes the token budget for every request that includes the schema.
    * It signals that the schema is not being maintained.

    Fires on:

    * ``deprecated`` / ``DEPRECATED``
    * ``do not use`` / ``DO NOT USE``
    * ``will be removed``
    * ``being removed``
    * ``obsolete``
    * ``no longer supported``

    Examples::

        # flagged — deprecation label
        "DEPRECATED: use get_user_v2 instead."
        "Fetch legacy data. This tool is deprecated."
        "Will be removed in v2. Use create_order instead."

        # correct — tool removed or description updated
        # (remove the tool from the schema entirely)
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _DEPRECATED_RE.search(desc)
    if not m:
        return None
    term = m.group(0)
    return Issue(
        tool=name,
        severity="warn",
        check="description_says_deprecated",
        message=(
            "tool description contains '{term}' — deprecated tools should be "
            "removed from the schema, not documented as deprecated."
        ).format(term=term),
    )


# ---------------------------------------------------------------------------
# Check 63: description_has_note_label
# ---------------------------------------------------------------------------

_NOTE_LABEL_RE = re.compile(
    r'(?<![A-Za-z])(?:Note|Important|Warning|Caution|Tip|Caveat)\s*:',
    re.IGNORECASE,
)
"""Pattern matching labeled meta-sections in tool descriptions."""


def _check_description_has_note_label(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 63: description_has_note_label — tool description contains a labeled
    meta-section like "Note:", "Important:", "Warning:", "Tip:", or "Caution:".

    These labels interrupt the flow of the description with inline commentary
    that should either be:

    * Integrated into the description prose: ``"Delete the record (irreversible)."``
    * Moved to the system prompt for operational guidance.

    Labeled notes are a form of structured prose in what should be a concise,
    single-sentence description.  They expand the description with meta-content
    the model must parse past to understand the core action.  They also tokenise
    poorly — "Note:" takes a token, the colon takes a token, the space takes
    a token — before any useful information is conveyed.

    Fires when the description contains any of:

    * ``Note:`` / ``NOTE:``
    * ``Important:`` / ``IMPORTANT:``
    * ``Warning:`` / ``WARNING:``
    * ``Caution:`` / ``CAUTION:``
    * ``Tip:`` / ``TIP:``
    * ``Caveat:`` / ``CAVEAT:``

    Does **not** fire on:
    * These words without a colon (e.g., "important parameter", "warning signs")
    * Words that start with these but are different (e.g., "Notation:", "Importantly,")
    * Check 48 (`description_model_instructions`) already catches "IMPORTANT: call..."
      directives; this check catches the broader labeled-section pattern

    Examples::

        # flagged — labeled meta-section
        "Delete all records. Note: This operation is irreversible."
        "Fetch the current user. Important: Requires authentication."
        "Run the command. Warning: High resource usage."

        # correct — integrated prose
        "Delete all records. This operation cannot be undone."
        "Fetch the current authenticated user's profile."
        "Run the command (may use significant CPU/memory)."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _NOTE_LABEL_RE.search(desc)
    if not m:
        return None
    label = m.group(0).rstrip(':').strip()
    return Issue(
        tool=name,
        severity="warn",
        check="description_has_note_label",
        message=(
            "tool description contains labeled meta-section '{label}:' — "
            "integrate the note into the prose or move operational guidance "
            "to the system prompt."
        ).format(label=label),
    )


# ---------------------------------------------------------------------------
# Check 64: description_contains_url
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r'https?://')
"""Pattern matching http:// or https:// URLs in tool descriptions."""


def _check_description_contains_url(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 64: description_contains_url — tool description contains a URL.

    URLs embedded in tool descriptions are a schema smell:

    * **Maintenance burden** — links rot; when the URL changes, the schema
      must be updated and re-deployed.
    * **Token waste** — a URL can consume 10-30+ tokens for zero semantic
      value to the model.
    * **Wrong layer** — documentation links belong in the README or docs
      site, not in the schema that the model reads at inference time.

    The fix is to remove the URL entirely or replace it with a short prose
    phrase that conveys the same intent without a link.

    Fires when the tool description contains ``http://`` or ``https://``.

    Examples::

        # flagged — URL in description
        "Fetch weather data. See https://api.weather.gov/docs for details."
        "Call the payments API (https://stripe.com/docs/api)."

        # correct — prose only
        "Fetch current weather conditions for a location."
        "Create a payment intent using the Stripe payments API."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    if not _URL_RE.search(desc):
        return None
    return Issue(
        tool=name,
        severity="warn",
        check="description_contains_url",
        message=(
            "tool description contains a URL — documentation links belong in "
            "the README or docs site, not the schema; remove the URL or replace "
            "with a short prose phrase."
        ),
    )


# Check 62: description_3p_action_verb
# ---------------------------------------------------------------------------

_THIRD_PERSON_ACTION_VERBS_RE = re.compile(
    r'^(?:'
    r'Creates|Updates|Deletes|Modifies|Removes|Adds|Inserts|Replaces'
    r'|Searches|Queries|Filters|Sorts|Finds'
    r'|Sends|Posts|Submits|Uploads|Downloads'
    r'|Sets|Saves|Opens|Closes'
    r'|Runs|Executes|Starts|Stops|Triggers|Cancels'
    r'|Parses|Converts|Transforms|Formats'
    r'|Archives|Restores|Syncs'
    r'|Manages|Handles|Processes|Performs'
    r'|Connects|Disconnects'
    r'|Builds|Deploys|Publishes'
    r'|Appends|Prepends|Merges|Joins|Splits'
    r'|Configures|Initializes|Resets|Refreshes'
    r'|Tracks|Logs|Records|Monitors'
    r'|Validates|Authenticates|Authorizes'
    r')\s',
    re.IGNORECASE,
)
"""Patterns matching 3rd-person singular action verbs that signal non-imperative mood."""


def _check_description_3p_action_verb(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 62: description_3p_action_verb — tool description starts with a common
    action verb in 3rd-person singular form instead of imperative mood.

    Check 56 (`tool_description_non_imperative`) catches output-focused 3rd-person
    verbs like "Returns", "Provides", "Retrieves".  This check catches the
    complementary set: common CRUD and operation verbs that also commonly appear
    in 3rd-person form in auto-generated or careless documentation.

    When a description says "Creates a new record" instead of "Create a new record",
    it reads as a statement about what the tool does from the outside ("it creates...")
    rather than a command ("create...").  The imperative form is the universal
    convention for tool descriptions.

    Fires when the tool description's first word is a recognized action verb in
    3rd-person singular form (ending in ``s``, e.g. ``Creates``, ``Updates``,
    ``Deletes``, ``Searches``, ``Sends``, ``Sets``, ``Runs``, ``Processes``...).

    Does **not** fire on:
    * Imperative base forms: "Create...", "Update...", "Delete..."
    * Check 56 already handles output-focused verbs (Returns, Gets, Lists, etc.)
    * Short tool names that happen to end in ``s`` but aren't action verbs

    Examples::

        # flagged — 3rd-person action verb
        "Creates a new user account in the system."
        "Updates the existing record with the provided values."
        "Searches all documents matching the query."
        "Sends an email notification to the specified recipient."
        "Validates the input schema against the rules."

        # correct — imperative
        "Create a new user account."
        "Update the existing record."
        "Search all documents matching the query."
        "Send an email notification."
        "Validate the input schema."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _THIRD_PERSON_ACTION_VERBS_RE.match(desc)
    if not m:
        return None
    verb = m.group(0).strip()
    # Suggest imperative by stripping trailing 's'
    imperative = verb[:-1] if verb.endswith('s') else verb
    return Issue(
        tool=name,
        severity="warn",
        check="description_3p_action_verb",
        message=(
            "tool description starts with 3rd-person verb '{verb}' — "
            "use imperative mood ('{imp}...' not '{verb}...'). "
            "Tool descriptions should command the action, not describe it."
        ).format(verb=verb, imp=imperative),
    )


# ---------------------------------------------------------------------------
# Check 60: description_starts_with_gerund
# ---------------------------------------------------------------------------

_GERUND_START_RE = re.compile(
    r'^[A-Z][a-z]+ing\s',
)
"""Pattern matching tool descriptions that start with a gerund (present participle)."""


def _check_description_starts_with_gerund(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 60: description_starts_with_gerund — tool description starts with a
    gerund (present participle verb form ending in -ing) instead of an imperative verb.

    Gerund forms describe the tool as if it were an action in progress rather
    than commanding the action.  "Creating a new user" says what the tool is
    doing; "Create a new user" says what to do.  The imperative form is the
    universal convention for tool/API documentation because it is shorter,
    clearer, and consistent with how commands are written.

    Common culprits (typically AI-generated):

    * ``"Creating a new record in the database."``  → ``"Create a new record."``
    * ``"Searching for files matching the query."`` → ``"Search for files."``
    * ``"Updating the user's profile settings."``   → ``"Update user profile settings."``
    * ``"Retrieving all active sessions."``         → ``"Retrieve all active sessions."``
    * ``"Listing available integrations."``         → ``"List available integrations."``
    * ``"Generating a summary of the text."``       → ``"Generate a text summary."``

    Fires when:

    * The tool description's first word is a capitalized word ending in ``-ing``
      followed by whitespace (e.g. ``Creating``, ``Searching``, ``Updating``).

    Does **not** fire on:

    * Short words that end in ``-ing`` but are not gerunds by context
      (e.g. ``"Ping the server"`` — "Ping" ends in ``g`` but not ``-ing``)
    * Mid-description gerunds (only the first word is checked)
    * Checks 56–59 cover other non-imperative patterns; this is the gerund case.

    Examples::

        # flagged — gerund preamble
        "Creating a new user account."
        "Updating the existing record."
        "Searching for matching documents."
        "Generating text using the AI model."

        # correct — imperative verb
        "Create a new user account."
        "Update the existing record."
        "Search for matching documents."
        "Generate text using the AI model."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _GERUND_START_RE.match(desc)
    if not m:
        return None
    gerund = m.group(0).strip()
    # Strip the trailing 'ing' to form the imperative hint
    imperative = gerund[:-3] if gerund.endswith('ing') else gerund
    return Issue(
        tool=name,
        severity="warn",
        check="description_starts_with_gerund",
        message=(
            "tool description starts with gerund '{gerund}' — "
            "use imperative mood (e.g. '{imp}...' not '{gerund}...'). "
            "Gerund forms describe what the tool is doing; imperative forms tell the model what to do."
        ).format(gerund=gerund, imp=imperative),
    )


# ---------------------------------------------------------------------------
# Check 59: description_starts_with_article
# ---------------------------------------------------------------------------

_ARTICLE_START_RE = re.compile(
    r'^(A|An|The)\s',
    re.IGNORECASE,
)
"""Pattern matching tool descriptions that start with an English article."""


def _check_description_starts_with_article(name: str, obj: Dict[str, Any], fmt: str) -> Optional[Issue]:
    """Check 59: description_starts_with_article — tool description starts with
    an article ('A', 'An', or 'The') instead of an imperative verb.

    Tool descriptions written in the style "A utility that searches records" or
    "The current user's session data" or "An endpoint for creating posts" are
    noun phrases, not action statements.  They describe *what the tool is*
    rather than *what it does*, which is a weaker signal for the model trying
    to decide which tool to call.

    Imperative mood ("Search for records", "Get the current session",
    "Create a post") is shorter, unambiguous, and consistent with how the
    best-documented MCP servers are written.

    Fires when the tool description's first word is "A", "An", or "The"
    followed by a space.

    Does **not** fire on:
    * Descriptions where the article is part of a compound abbreviation
      like "A/B" (slash immediately follows, no space)
    * Descriptions starting with "Access", "Abort", "Annotate", etc.
      (different words — only bare articles match)
    * Checks 56–58 already flag 3rd-person verbs, "This tool…", and
      "Allows you to…"; this check catches the remaining article pattern.

    Examples::

        # flagged — noun-phrase start
        "A utility that searches for files by name."
        "An endpoint for creating new database records."
        "The current user's profile data."
        "A wrapper around the calendar API."

        # correct — imperative verb start
        "Search for files by name."
        "Create new database records."
        "Get the current user's profile."
        "Interact with the calendar API."
    """
    desc = _get_tool_description(obj, fmt)
    if not desc or not isinstance(desc, str):
        return None
    m = _ARTICLE_START_RE.match(desc)
    if not m:
        return None
    article = m.group(1)
    return Issue(
        tool=name,
        severity="warn",
        check="description_starts_with_article",
        message=(
            "tool description starts with article '{article}' — "
            "use an imperative verb instead "
            "(e.g. 'A tool that searches' → 'Search for records', "
            "'The list of users' → 'List users')."
        ).format(article=article),
    )


_RANGE_IN_DESC_RE = re.compile(
    r'(?<!\d)(\d+)\s*(?:[-–—]|to)\s*(\d+)(?!\d)',
    re.IGNORECASE,
)


def _check_range_described_not_constrained(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 51: range_described_not_constrained — numeric param description mentions a range but schema lacks min/max.

    When a developer writes ``"limit": {"type": "integer", "description": "Number of results (1-100)"}``
    they are documenting a constraint in prose.  But prose constraints are only
    enforced at the API level — models see the description and *may* respect the
    range, but JSON Schema validation does not prevent them from passing 0 or
    9999.  Adding ``"minimum": 1, "maximum": 100`` to the schema makes the
    constraint machine-readable, lets validators enforce it, and removes
    ambiguity for the model.

    Fires when:

    * Param type is ``integer`` or ``number``, AND
    * Description contains a range pattern like ``1-100``, ``0-5``, ``1 to 20``, AND
    * The range bounds are plausible (low < high, difference ≤ 10000, high ≤ 1000000), AND
    * Schema has no ``minimum`` or ``maximum`` (or exclusive variants)

    Does **not** fire for params that already have min/max constraints, even
    partial ones (e.g. only ``minimum`` is set).

    Examples::

        # flagged — range in English but not in schema
        "per_page": {"type": "integer", "description": "Results per page (1-100)"}
        "limit":    {"type": "integer", "description": "Max results, 1 to 100"}
        "fps":      {"type": "integer", "description": "Frames per second, 1-30"}

        # correct — schema enforces what description says
        "per_page": {"type": "integer", "description": "Results per page (1-100)",
                     "minimum": 1, "maximum": 100}
        "temperature": {"type": "number", "description": "Sampling temp 0.0-1.0",
                        "minimum": 0.0, "maximum": 1.0}
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        ptype = param_schema.get("type", "")
        if ptype not in ("integer", "number"):
            continue
        # Skip if already has any min/max constraint
        if any(k in param_schema for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")):
            continue
        desc = param_schema.get("description", "")
        if not desc or not isinstance(desc, str):
            continue
        m = _RANGE_IN_DESC_RE.search(desc)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        # Plausibility filter: lo < hi, range ≤ 10000, hi ≤ 1_000_000
        if lo >= hi or hi - lo > 10_000 or hi > 1_000_000:
            continue
        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="range_described_not_constrained",
            message=(
                "param '{param}' description says '{lo}–{hi}' but schema has no "
                "minimum/maximum — models can pass out-of-range values; add "
                "\"minimum\": {lo}, \"maximum\": {hi} to enforce the constraint"
            ).format(param=param_name, lo=lo, hi=hi),
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 52: number_should_be_integer
# ---------------------------------------------------------------------------

# Param names whose semantics are unambiguously integer (no fractional meaning).
_INTEGER_PARAM_RE = re.compile(
    r'^(page|limit|count|per_page|page_size|offset|'
    r'num_|number_of_|n_results|max_results|top_k|top_n|'
    r'batch_size|concurrency|workers|retries|retry_count|retry|'
    r'port|depth|level|rank|priority|row|col|column|line|index|chunk|'
    r'max_tokens|max_length|max_size|start_index|end_index|'
    r'page_number|start_page|end_page|skip|take)'
    r'|(_count|_limit|_size|_num|_page|_index|_max_tokens)$',
    re.IGNORECASE,
)


def _check_number_should_be_integer(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 52: number_should_be_integer — param declared as ``number`` but name implies integer semantics.

    ``number`` admits floats (1.5, 2.7) which makes no sense for pagination
    params like ``page``, ``limit``, ``offset``, ``count``, ``per_page``, or
    structural params like ``port``, ``depth``, ``index``.  Passing 1.5 as a
    page number causes API errors downstream yet the schema would not reject it.

    Using ``integer`` instead:

    * Prevents models from generating invalid values (e.g. page 2.5)
    * Communicates the intent unambiguously
    * Enables downstream validators to enforce whole-number semantics

    Fires when:

    * Param type is ``number`` (not ``integer``), AND
    * Param name matches an unambiguously-integer pattern (page, limit, offset,
      count, per_page, port, depth, index, etc.)

    Does **not** fire for ``timeout``, ``duration``, ``delay``, ``threshold``,
    ``temperature``, ``ratio``, or similar params where fractional values are
    legitimate.

    Examples::

        # flagged — should be integer
        "page":     {"type": "number", "description": "Page number"}
        "limit":    {"type": "number", "description": "Max results"}
        "per_page": {"type": "number", "description": "Results per page"}

        # correct
        "page":       {"type": "integer", "description": "Page number"}
        "temperature": {"type": "number",  "description": "Sampling temperature"}
        "ratio":       {"type": "number",  "description": "Aspect ratio"}
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
        if _INTEGER_PARAM_RE.search(param_name.lower()):
            issues.append(Issue(
                tool=tool_name,
                severity="warn",
                check="number_should_be_integer",
                message=(
                    "param '{param}' has type 'number' but the name implies integer semantics "
                    "— use type 'integer' to prevent models from passing fractional values "
                    "(e.g. page 1.5) that most APIs reject"
                ).format(param=param_name),
            ))

    return issues


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
    # Additional integer-implying names (merged from check 52 in v0.103.1)
    "port", "level", "rank", "priority", "row", "col", "column", "line",
    "concurrency", "workers", "n_results", "max_length", "max_size",
    "page_number", "start_page", "end_page", "start_index", "end_index",
    "retry_count",
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


# ---------------------------------------------------------------------------
# Check 43: string_comma_separated
# ---------------------------------------------------------------------------

_COMMA_SEP_RE = re.compile(
    r'comma[- ]separated|comma[- ]delimited|pipe[- ]separated'
    r'|newline[- ]separated|space[- ]separated',
    re.IGNORECASE,
)


def _check_string_comma_separated(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 43: string_comma_separated — param description says "comma-separated" but type is string.

    When a description says the param is "comma-separated", the value is
    actually a list — and JSON Schema has native support for lists via
    ``type: "array"``.  Using a string forces the model to manually construct
    a delimited string, which is error-prone and forfeits schema-level
    validation of individual items.

    This check fires when:

    * ``type`` is ``"string"``
    * No ``enum`` field (enum strings are intentional)
    * Description matches: comma-separated / comma-delimited / pipe-separated
      / newline-separated / space-separated

    The fix is to change the type to ``"array"`` and add an ``items`` schema.
    """
    issues = []
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        ptype = param_schema.get("type")
        if ptype != "string":
            continue
        if "enum" in param_schema:
            continue
        desc = param_schema.get("description") or ""
        if not isinstance(desc, str):
            continue
        if not _COMMA_SEP_RE.search(desc):
            continue

        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="string_comma_separated",
            message=(
                "param '{param}' description says values are delimited "
                "(e.g. 'comma-separated') but type is 'string'; "
                "use type: 'array' with an items schema so each element "
                "is validated individually"
            ).format(param=param_name),
        ))

    return issues


# ---------------------------------------------------------------------------
# Check 44: enum_single_const
# ---------------------------------------------------------------------------
# Check 45: required_array_no_minitems
# ---------------------------------------------------------------------------

def _check_required_array_no_minitems(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 45: required_array_no_minitems — required array param has no minItems constraint.

    A ``required`` array param with no ``minItems`` field allows the model to
    pass an empty list ``[]``.  For most tools that take a list of resources
    (files, IDs, topics, paths, queries) an empty array is either invalid or
    a no-op.  Adding ``minItems: 1`` makes the constraint explicit and prevents
    the model from accidentally sending empty arrays.

    This check fires when:

    * Param is listed in the schema's ``required`` array
    * Param type is ``"array"``
    * Param has no ``minItems`` field

    The fix is adding ``"minItems": 1`` (or a larger value if appropriate).
    """
    issues = []
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        return issues

    for param_name, param_schema in properties.items():
        if not isinstance(param_schema, dict):
            continue
        if param_schema.get("type") != "array":
            continue
        if param_name not in required:
            continue
        if "minItems" in param_schema:
            continue

        issues.append(Issue(
            tool=tool_name,
            severity="warn",
            check="required_array_no_minitems",
            message=(
                "required array param '{param}' has no 'minItems' constraint; "
                "the model can pass an empty list [] — add 'minItems: 1' "
                "if an empty array is not a valid input"
            ).format(param=param_name),
        ))

    return issues


# ---------------------------------------------------------------------------

def _check_enum_single_const(tool_name: str, schema: Dict[str, Any]) -> List[Issue]:
    """Check 44: enum_single_const — enum array with a single value; use const instead.

    When ``enum`` contains exactly one value, the intent is to express a
    constant — a param that can only ever hold one specific value.  JSON Schema
    provides ``const`` precisely for this case:

    * ``{"enum": ["graphite"]}`` — technically valid, misleads readers
    * ``{"const": "graphite"}`` — semantically clear, shorter, correct

    This check fires whenever a param's ``enum`` array has exactly one element
    at any schema nesting level (top-level params and nested object properties).
    """
    issues = []

    def _check_props(properties: Dict[str, Any], path: str = "") -> None:
        for param_name, param_schema in properties.items():
            if not isinstance(param_schema, dict):
                continue
            param_path = f"{path}.{param_name}" if path else param_name
            enum_val = param_schema.get("enum")
            if isinstance(enum_val, list) and len(enum_val) == 1:
                issues.append(Issue(
                    tool=tool_name,
                    severity="warn",
                    check="enum_single_const",
                    message=(
                        "param '{param}' has enum with a single value {val!r}; "
                        "use const: {val!r} instead — enum implies multiple options, "
                        "const expresses a fixed value"
                    ).format(param=param_path, val=enum_val[0]),
                ))
            # Recurse into nested object properties
            nested = param_schema.get("properties")
            if isinstance(nested, dict):
                _check_props(nested, param_path)

    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        _check_props(properties)

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

        # Check 14: name_snake_case
        issue = _check_name_snake_case(name)
        if issue is not None:
            issues.append(issue)

        # Check 78 (tool): name_uses_hyphen
        issue = _check_tool_name_uses_hyphen(name)
        if issue is not None:
            issues.append(issue)

        # Check 93: tool_name_contains_version
        issue = _check_tool_name_contains_version(name)
        if issue is not None:
            issues.append(issue)

        # Check 94: param_name_is_reserved_word
        issues.extend(_check_param_name_is_reserved_word(name, schema))

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

        # Check 95: description_has_version_info
        issue = _check_description_has_version_info(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 96: description_has_todo_marker
        issues.extend(_check_description_has_todo_marker(name, raw_obj, schema, fmt))

        # Check 97: array_max_items_zero
        issues.extend(_check_array_max_items_zero(name, schema))

        # Check 98: description_says_see_docs
        issues.extend(_check_description_says_see_docs(name, raw_obj, schema, fmt))

        # Check 99: description_has_internal_path
        issues.extend(_check_description_has_internal_path(name, raw_obj, schema, fmt))

        # Check 100: param_accepts_secret_no_format
        issues.extend(_check_param_accepts_secret_no_format(name, schema))

        # Check 101: param_uses_schema_ref
        issues.extend(_check_param_uses_schema_ref(name, schema))

        # Check 102: tool_name_too_generic
        issue = _check_tool_name_too_generic(name)
        if issue is not None:
            issues.append(issue)

        # Check 103: string_minlength_zero
        issues.extend(_check_string_minlength_zero(name, schema))

        # Check 104: enum_values_inconsistent_case
        issues.extend(_check_enum_values_inconsistent_case(name, schema))

        # Check 105: schema_has_definitions
        issue = _check_schema_has_definitions(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 106: description_ends_abruptly
        issues.extend(_check_description_ends_abruptly(name, raw_obj, schema, fmt))

        # Check 107: object_no_props_additional_false
        issues.extend(_check_object_no_props_additional_false(name, schema))

        # Check 108: array_items_empty_schema
        issues.extend(_check_array_items_empty_schema(name, schema))

        # Check 109: description_has_parenthetical_type
        issues.extend(_check_description_has_parenthetical_type(name, schema))

        # Check 110: param_name_starts_with_type
        issues.extend(_check_param_name_starts_with_type(name, schema))

        # Check 111: param_name_ends_with_type
        issues.extend(_check_param_name_ends_with_type(name, schema))

        # Check 112: enum_duplicate_values
        issues.extend(_check_enum_duplicate_values(name, schema))

        # Check 113: name_uses_camelcase
        issues.extend(_check_name_uses_camelcase(name, raw_obj, schema, fmt))

        # Check 114: name_starts_with_uppercase
        issues.extend(_check_name_starts_with_uppercase(name, schema))

        # Check 115: param_type_is_null
        issues.extend(_check_param_type_is_null(name, schema))

        # Check 116: enum_has_empty_value
        issues.extend(_check_enum_has_empty_value(name, schema))

        # Check 117: param_name_too_generic
        issues.extend(_check_param_name_too_generic(name, schema))

        # Check 118: description_uses_first_person
        issues.extend(_check_description_uses_first_person(name, raw_obj, schema, fmt))

        # Check 119: description_has_json_example
        issues.extend(_check_description_has_json_example(name, raw_obj, schema, fmt))

        # Check 120: required_not_array
        issues.extend(_check_required_not_array(name, schema))

        # Check 121: param_name_all_uppercase
        issues.extend(_check_param_name_all_uppercase(name, schema))

        # Check 122: required_param_not_in_properties
        issues.extend(_check_required_param_not_in_properties(name, schema))

        # Check 123: param_name_has_double_underscore
        issues.extend(_check_param_name_has_double_underscore(name, schema))

        # Check 124: param_name_starts_with_underscore
        issues.extend(_check_param_name_starts_with_underscore(name, schema))

        # Check 125: param_name_is_number
        issues.extend(_check_param_name_is_number(name, schema))

        # Check 126: name_ends_with_underscore
        issues.extend(_check_name_ends_with_underscore(name, schema))

        # Check 127: param_name_has_space
        issues.extend(_check_param_name_has_space(name, schema))

        # Check 128: schema_type_not_object
        issue = _check_schema_type_not_object(name, schema)
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

        # Check 43: string_comma_separated
        issues.extend(_check_string_comma_separated(name, schema))

        # Check 44: enum_single_const
        issues.extend(_check_enum_single_const(name, schema))

        # Check 45: required_array_no_minitems
        issues.extend(_check_required_array_no_minitems(name, schema))

        # Check 46: required_array_empty
        issue = _check_required_array_empty(name, schema)
        if issue is not None:
            issues.append(issue)

        # Check 47: description_markdown_formatting
        _tool_desc_47 = (raw_obj.get('description', '') or '') if isinstance(raw_obj, dict) else ''
        issues.extend(_check_description_markdown_formatting(name, _tool_desc_47, schema))

        # Check 48: description_model_instructions
        _tool_desc_48 = (raw_obj.get('description', '') or '') if isinstance(raw_obj, dict) else ''
        issue = _check_description_model_instructions(name, _tool_desc_48)
        if issue is not None:
            issues.append(issue)

        # Check 49: required_string_no_minlength
        issues.extend(_check_required_string_no_minlength(name, schema))

        # Check 50: param_description_says_optional
        issues.extend(_check_param_description_says_optional(name, schema))

        # Check 51: range_described_not_constrained
        issues.extend(_check_range_described_not_constrained(name, schema))

        # Check 54: optional_string_no_minlength
        issues.extend(_check_optional_string_no_minlength(name, schema))

        # Check 55: required_param_has_default
        issues.extend(_check_required_param_has_default(name, schema))

        # Check 56: tool_description_non_imperative
        issue = _check_tool_description_non_imperative(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 57: description_this_tool
        issue = _check_description_this_tool(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 58: description_allows_you_to
        issue = _check_description_allows_you_to(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 59: description_starts_with_article
        issue = _check_description_starts_with_article(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 60: description_starts_with_gerund
        issue = _check_description_starts_with_gerund(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 62: description_3p_action_verb
        issue = _check_description_3p_action_verb(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 63: description_has_note_label
        issue = _check_description_has_note_label(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 64: description_contains_url
        issue = _check_description_contains_url(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 65: description_says_deprecated
        issue = _check_description_says_deprecated(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 66: param_description_says_required
        issues.extend(_check_param_description_says_required(name, schema))

        # Check 67: enum_default_not_in_enum
        issues.extend(_check_enum_default_not_in_enum(name, schema))

        # Check 68: const_param_should_be_removed
        issues.extend(_check_const_param_should_be_removed(name, schema))

        # Check 69: contradictory_min_max
        issues.extend(_check_contradictory_min_max(name, schema))

        # Check 70: description_is_placeholder
        issues.extend(_check_description_is_placeholder(name, raw_obj, fmt))

        # Check 71: schema_has_title_field
        issues.extend(_check_schema_has_title_field(name, schema, fmt))

        # Check 72: tool_name_too_long
        issue = _check_tool_name_too_long(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 73: param_name_too_long
        issues.extend(_check_param_name_too_long(name, schema))

        # Check 74: description_word_repetition
        issue = _check_description_word_repetition(name, raw_obj, fmt)
        if issue is not None:
            issues.append(issue)

        # Check 75: default_type_mismatch
        issues.extend(_check_default_type_mismatch(name, schema))

        # Check 76: param_name_implies_boolean
        issues.extend(_check_param_name_implies_boolean(name, schema))

        # Check 77: anyof_null_should_be_optional
        issues.extend(_check_anyof_null_should_be_optional(name, schema))

        # Check 78 (param): name_uses_hyphen
        issues.extend(_check_param_name_uses_hyphen(name, schema))

        # Check 79: description_has_example
        issues.extend(_check_description_has_example(name, schema))

        # Check 80: description_lists_enum_values
        issues.extend(_check_description_lists_enum_values(name, schema))

        # Check 81: param_description_says_ignored
        issues.extend(_check_param_description_says_ignored(name, schema))

        # Check 82: enum_boolean_string
        issues.extend(_check_enum_boolean_string(name, schema))

        # Check 83: param_nullable_field
        issues.extend(_check_param_nullable_field(name, schema))

        # Check 84: schema_has_x_field
        issues.extend(_check_schema_has_x_field(name, schema))

        # Check 85: default_violates_minimum
        issues.extend(_check_default_violates_minimum(name, schema))

        # Check 86: param_name_single_char
        issues.extend(_check_param_name_single_char(name, schema))

        # Check 87: allof_single_schema
        issues.extend(_check_allof_single_schema(name, schema))

        # Check 88: enum_has_duplicates
        issues.extend(_check_enum_has_duplicates(name, schema))

        # Check 89: description_has_html
        issues.extend(_check_description_has_html(name, raw_obj, schema, fmt))

        # Check 90: description_starts_with_param_name
        issues.extend(_check_description_starts_with_param_name(name, schema))

        # Check 91: string_type_describes_json
        issues.extend(_check_string_type_describes_json(name, schema))

        # Check 92: object_param_no_properties
        issues.extend(_check_object_param_no_properties(name, schema))

        # Note: check 52 (number_should_be_integer) is subsumed by check 40
        # (number_type_for_integer) — merged into check 40 in v0.103.1.

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

    # Check 53: tool_name_redundant_prefix (cross-tool)
    issues.extend(_check_tool_name_redundant_prefix(names))

    # Check 61: description_duplicate (cross-tool)
    _tool_descs_61 = [
        (n, (_get_tool_description(raw, fmt) or "").strip())
        for n, fmt, raw, _schema in tool_data
    ]
    issues.extend(_check_description_duplicate(_tool_descs_61))

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
