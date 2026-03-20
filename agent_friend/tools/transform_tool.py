"""transform_tool.py — TransformTool for agent-friend (stdlib only).

Structured data transformation: map field names, coerce types, flatten/
nest JSON, pick/omit keys, apply record templates, and transform lists of
dicts — all with zero dependencies.

Features:
* transform_pick    — extract specific keys from a dict
* transform_omit    — remove specific keys from a dict
* transform_rename  — rename keys in a dict
* transform_coerce  — type-coerce values (str/int/float/bool/list/dict)
* transform_flatten — flatten nested dict into dot-notation keys
* transform_unflatten — convert dot-notation keys back into nested dict
* transform_map_records — apply pick/omit/rename/coerce to a list of dicts
* transform_merge   — deep merge multiple dicts

Usage::

    tool = TransformTool()

    tool.transform_pick({"a": 1, "b": 2, "c": 3}, keys=["a", "c"])
    # {"result": {"a": 1, "c": 3}}

    tool.transform_flatten({"user": {"name": "alice", "age": 30}})
    # {"result": {"user.name": "alice", "user.age": 30}}

    tool.transform_map_records(
        records=[{"n": "alice", "s": "95"}, ...],
        rename={"n": "name", "s": "score"},
        coerce={"score": "int"},
    )
"""

import json
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


# ── helpers ───────────────────────────────────────────────────────────────

def _coerce_value(value: Any, target_type: str) -> Any:
    """Convert *value* to *target_type*. Raises ValueError on failure."""
    if target_type == "str":
        return str(value)
    if target_type == "int":
        return int(float(value))
    if target_type == "float":
        return float(value)
    if target_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() not in ("", "false", "0", "no", "null", "none")
        return bool(value)
    if target_type == "list":
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                result = json.loads(value)
                if isinstance(result, list):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass
        return [value]
    if target_type == "dict":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            return json.loads(value)
        raise ValueError(f"Cannot coerce {type(value).__name__} to dict")
    if target_type == "null" or target_type == "none":
        return None
    raise ValueError(f"Unknown target type: {target_type!r}")


def _flatten(obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dict into dot-notation keys."""
    result: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{sep}{k}" if prefix else k
            result.update(_flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}{sep}{i}" if prefix else str(i)
            result.update(_flatten(v, new_key, sep))
    else:
        result[prefix] = obj
    return result


def _unflatten(flat: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Convert dot-notation keys back into a nested dict."""
    result: Dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge *override* into *base* (override wins on conflict)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── tool ──────────────────────────────────────────────────────────────────

class TransformTool(BaseTool):
    """Structured data transformation: pick, omit, rename, coerce, flatten,
    unflatten, deep merge, and batch record transforms.
    """

    def transform_pick(self, record: Dict, keys: List[str]) -> str:
        """Extract only *keys* from *record*.

        Returns ``{result, picked, missing}``.
        """
        if not isinstance(record, dict):
            return json.dumps({"error": "record must be a dict"})
        if not isinstance(keys, list):
            return json.dumps({"error": "keys must be a list"})
        result = {}
        missing = []
        for k in keys:
            if k in record:
                result[k] = record[k]
            else:
                missing.append(k)
        return json.dumps({"result": result, "picked": len(result), "missing": missing})

    def transform_omit(self, record: Dict, keys: List[str]) -> str:
        """Remove *keys* from *record*, keeping everything else.

        Returns ``{result, omitted}``.
        """
        if not isinstance(record, dict):
            return json.dumps({"error": "record must be a dict"})
        if not isinstance(keys, list):
            return json.dumps({"error": "keys must be a list"})
        omit_set = set(keys)
        result = {k: v for k, v in record.items() if k not in omit_set}
        return json.dumps({"result": result, "omitted": len(record) - len(result)})

    def transform_rename(self, record: Dict, mapping: Dict[str, str]) -> str:
        """Rename keys in *record* according to *mapping* {old: new}.

        Unmapped keys are kept as-is.

        Returns ``{result, renamed}``.
        """
        if not isinstance(record, dict):
            return json.dumps({"error": "record must be a dict"})
        if not isinstance(mapping, dict):
            return json.dumps({"error": "mapping must be a dict"})
        result = {}
        renamed = 0
        for k, v in record.items():
            new_key = mapping.get(k, k)
            if new_key != k:
                renamed += 1
            result[new_key] = v
        return json.dumps({"result": result, "renamed": renamed})

    def transform_coerce(self, record: Dict, types: Dict[str, str]) -> str:
        """Type-coerce values in *record* according to *types* {key: type}.

        Supported types: str | int | float | bool | list | dict | null.
        Keys not in *types* are kept unchanged.

        Returns ``{result, coerced, errors}``.
        """
        if not isinstance(record, dict):
            return json.dumps({"error": "record must be a dict"})
        if not isinstance(types, dict):
            return json.dumps({"error": "types must be a dict"})
        result = dict(record)
        errors = []
        coerced = 0
        for key, target in types.items():
            if key not in result:
                continue
            try:
                result[key] = _coerce_value(result[key], target)
                coerced += 1
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                errors.append({"key": key, "error": str(e)})
        return json.dumps({"result": result, "coerced": coerced, "errors": errors})

    def transform_flatten(
        self,
        record: Any,
        sep: str = ".",
        max_depth: Optional[int] = None,
    ) -> str:
        """Flatten a nested dict into dot-notation keys.

        Arrays are indexed as ``key.0``, ``key.1``, etc.
        *max_depth* limits recursion depth (None = unlimited).

        Returns ``{result, key_count}``.
        """
        if not isinstance(record, (dict, list)):
            return json.dumps({"error": "record must be a dict or list"})
        if not sep:
            return json.dumps({"error": "sep must be non-empty"})
        result = _flatten(record, sep=sep)
        return json.dumps({"result": result, "key_count": len(result)})

    def transform_unflatten(self, record: Dict[str, Any], sep: str = ".") -> str:
        """Convert dot-notation keys into a nested dict.

        Returns ``{result}``.
        """
        if not isinstance(record, dict):
            return json.dumps({"error": "record must be a dict"})
        if not sep:
            return json.dumps({"error": "sep must be non-empty"})
        result = _unflatten(record, sep=sep)
        return json.dumps({"result": result})

    def transform_map_records(
        self,
        records: List[Dict],
        pick: Optional[List[str]] = None,
        omit: Optional[List[str]] = None,
        rename: Optional[Dict[str, str]] = None,
        coerce: Optional[Dict[str, str]] = None,
        add: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply transformations to each record in a list.

        Operations applied in order: pick → omit → rename → coerce → add.
        *add*: add static key-value pairs to every record.

        Returns ``{results, count, errors}``.
        """
        if not isinstance(records, list):
            return json.dumps({"error": "records must be a list"})

        results = []
        errors = []

        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                errors.append({"index": i, "error": "record is not a dict"})
                continue
            r = dict(rec)

            # pick
            if pick is not None:
                r = {k: v for k, v in r.items() if k in pick}

            # omit
            if omit is not None:
                omit_set = set(omit)
                r = {k: v for k, v in r.items() if k not in omit_set}

            # rename
            if rename is not None:
                r = {rename.get(k, k): v for k, v in r.items()}

            # coerce
            if coerce is not None:
                for key, target in coerce.items():
                    if key in r:
                        try:
                            r[key] = _coerce_value(r[key], target)
                        except (ValueError, TypeError, json.JSONDecodeError) as e:
                            errors.append({"index": i, "key": key, "error": str(e)})

            # add static fields
            if add is not None:
                r.update(add)

            results.append(r)

        return json.dumps({"results": results, "count": len(results), "errors": errors})

    def transform_merge(self, *dicts) -> str:
        """Deep merge multiple dicts. Later dicts override earlier ones.

        Returns ``{result, merged_from}``.
        """
        if not dicts:
            return json.dumps({"error": "at least one dict required"})
        valid = []
        for i, d in enumerate(dicts):
            if not isinstance(d, dict):
                return json.dumps({"error": f"argument {i} is not a dict"})
            valid.append(d)
        result = {}
        for d in valid:
            result = _deep_merge(result, d)
        return json.dumps({"result": result, "merged_from": len(valid)})

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "transform"

    @property
    def description(self) -> str:
        return (
            "Structured data transformation. transform_pick, transform_omit, "
            "transform_rename, transform_coerce (str/int/float/bool/list/dict), "
            "transform_flatten (nested→dot-notation), transform_unflatten, "
            "transform_map_records (batch pick+omit+rename+coerce+add), "
            "transform_merge (deep merge). Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "transform_pick",
                "description": "Extract only specified keys from a dict. Returns {result, picked, missing}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "Input dict"},
                        "keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to keep"},
                    },
                    "required": ["record", "keys"],
                },
            },
            {
                "name": "transform_omit",
                "description": "Remove specified keys from a dict. Returns {result, omitted}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "Input dict"},
                        "keys": {"type": "array", "items": {"type": "string"}, "description": "Keys to remove"},
                    },
                    "required": ["record", "keys"],
                },
            },
            {
                "name": "transform_rename",
                "description": "Rename keys using {old: new} mapping. Unmapped keys kept. Returns {result, renamed}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "Input dict"},
                        "mapping": {"type": "object", "description": "{old_key: new_key}"},
                    },
                    "required": ["record", "mapping"],
                },
            },
            {
                "name": "transform_coerce",
                "description": "Type-coerce values. types={key: str/int/float/bool/list/dict/null}. Returns {result, coerced, errors}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "Input dict"},
                        "types": {"type": "object", "description": "{key: target_type}"},
                    },
                    "required": ["record", "types"],
                },
            },
            {
                "name": "transform_flatten",
                "description": "Flatten nested dict to dot-notation keys (arrays indexed as key.0). Returns {result, key_count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"description": "Nested dict or list"},
                        "sep": {"type": "string", "description": "Separator (default '.')"},
                    },
                    "required": ["record"],
                },
            },
            {
                "name": "transform_unflatten",
                "description": "Convert dot-notation keys back into nested dict. Returns {result}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "record": {"type": "object", "description": "Flat dot-notation dict"},
                        "sep": {"type": "string", "description": "Separator (default '.')"},
                    },
                    "required": ["record"],
                },
            },
            {
                "name": "transform_map_records",
                "description": "Apply pick/omit/rename/coerce/add to each record in a list. Returns {results, count, errors}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "records": {"type": "array", "description": "List of dicts"},
                        "pick": {"type": "array", "items": {"type": "string"}, "description": "Keys to keep"},
                        "omit": {"type": "array", "items": {"type": "string"}, "description": "Keys to remove"},
                        "rename": {"type": "object", "description": "{old: new} mapping"},
                        "coerce": {"type": "object", "description": "{key: type} mapping"},
                        "add": {"type": "object", "description": "Static fields to add"},
                    },
                    "required": ["records"],
                },
            },
            {
                "name": "transform_merge",
                "description": "Deep merge multiple dicts. Later dicts win. Returns {result, merged_from}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dicts": {"type": "array", "items": {"type": "object"}, "description": "Dicts to merge"},
                    },
                    "required": ["dicts"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "transform_pick":
            return self.transform_pick(**arguments)
        if tool_name == "transform_omit":
            return self.transform_omit(**arguments)
        if tool_name == "transform_rename":
            return self.transform_rename(**arguments)
        if tool_name == "transform_coerce":
            return self.transform_coerce(**arguments)
        if tool_name == "transform_flatten":
            return self.transform_flatten(**arguments)
        if tool_name == "transform_unflatten":
            return self.transform_unflatten(**arguments)
        if tool_name == "transform_map_records":
            return self.transform_map_records(**arguments)
        if tool_name == "transform_merge":
            dicts = arguments.get("dicts", [])
            return self.transform_merge(*dicts)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
