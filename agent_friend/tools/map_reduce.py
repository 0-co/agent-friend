"""map_reduce.py — MapReduceTool for agent-friend (stdlib only).

Map, filter, and reduce operations over JSON lists without needing
CodeTool or pandas.  Pairs well with JSONTool, TableTool, and HTTPTool
for data-pipeline style transformations.

Features:
* map  — extract a field or apply a built-in transform to every item
* filter — keep items matching a predicate (eq/ne/gt/lt/gte/lte/contains/startswith/exists)
* reduce — aggregate a list to a scalar (count, sum, avg, min, max, first, last, join, unique)
* sort   — sort a list by a field
* group  — group items by a field value → {key: [items]}
* flatten — flatten a list of lists
* zip_lists — zip two lists into a list of {left, right} pairs

All inputs and outputs are JSON strings.

Usage::

    tool = MapReduceTool()

    data = '[{"name": "Alice", "score": 90}, {"name": "Bob", "score": 75}]'

    # Extract all names
    tool.mr_map(data, "name")          # '["Alice", "Bob"]'

    # Keep scores >= 80
    tool.mr_filter(data, "score", "gte", 80)   # '[{"name": "Alice", "score": 90}]'

    # Average score
    tool.mr_reduce(data, "score", "avg")       # '82.5'
"""

import json
import math
from typing import Any, Dict, List, Optional

from .base import BaseTool

_NUMERIC_OPS = {"sum", "avg", "min", "max"}


def _get_field(item: Any, field: str) -> Any:
    """Traverse dot-notation path: 'user.name' → item['user']['name']."""
    if field == "." or field == "":
        return item
    parts = field.split(".")
    v = item
    for p in parts:
        if isinstance(v, dict):
            v = v.get(p)
        else:
            return None
    return v


def _match(value: Any, operator: str, operand: Any) -> bool:
    """Return True if *value* satisfies *operator* *operand*."""
    if operator == "eq":
        return value == operand
    if operator == "ne":
        return value != operand
    if operator == "exists":
        return value is not None
    if operator == "contains":
        return isinstance(value, str) and str(operand) in value
    if operator == "startswith":
        return isinstance(value, str) and value.startswith(str(operand))
    if operator == "endswith":
        return isinstance(value, str) and value.endswith(str(operand))
    # Numeric comparisons
    try:
        v = float(value)
        o = float(operand)
    except (TypeError, ValueError):
        return False
    if operator == "gt":
        return v > o
    if operator == "lt":
        return v < o
    if operator == "gte":
        return v >= o
    if operator == "lte":
        return v <= o
    return False


# Built-in field transforms for mr_map
_TRANSFORMS = {
    "upper": lambda v: v.upper() if isinstance(v, str) else v,
    "lower": lambda v: v.lower() if isinstance(v, str) else v,
    "strip": lambda v: v.strip() if isinstance(v, str) else v,
    "len": lambda v: len(v) if isinstance(v, (str, list, dict)) else None,
    "bool": lambda v: bool(v),
    "str": lambda v: str(v) if v is not None else None,
    "int": lambda v: int(v) if v is not None else None,
    "float": lambda v: float(v) if v is not None else None,
    "abs": lambda v: abs(v) if isinstance(v, (int, float)) else v,
    "not": lambda v: not v,
}


class MapReduceTool(BaseTool):
    """Map, filter, sort, group, and reduce JSON lists without CodeTool.

    All operations take JSON strings and return JSON strings, so they
    chain naturally with JSONTool, HTTPTool, and TableTool.

    Parameters
    ----------
    max_items:
        Maximum list size to process (default 10_000).
    """

    def __init__(self, max_items: int = 10_000) -> None:
        self.max_items = max_items

    # ── helpers ───────────────────────────────────────────────────────

    def _parse(self, data: str) -> List[Any]:
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc
        if not isinstance(parsed, list):
            raise ValueError("Input must be a JSON array")
        if len(parsed) > self.max_items:
            raise ValueError(
                f"List has {len(parsed)} items; max is {self.max_items}"
            )
        return parsed

    # ── public API ────────────────────────────────────────────────────

    def mr_map(self, data: str, field: str, transform: Optional[str] = None) -> str:
        """Extract *field* from every item (dot-notation OK).

        If *transform* is given, apply it to each extracted value.
        Transforms: upper, lower, strip, len, bool, str, int, float, abs, not.

        Returns a JSON array of extracted values.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        fn = _TRANSFORMS.get(transform) if transform else None
        if transform and fn is None:
            return json.dumps({"error": f"Unknown transform '{transform}'. Valid: {sorted(_TRANSFORMS)}"})

        result = []
        for item in items:
            v = _get_field(item, field)
            if fn is not None:
                try:
                    v = fn(v)
                except Exception:
                    v = None
            result.append(v)
        return json.dumps(result)

    def mr_filter(
        self,
        data: str,
        field: str,
        operator: str,
        value: Any = None,
    ) -> str:
        """Keep items where *field* satisfies *operator* against *value*.

        Operators: eq, ne, gt, lt, gte, lte, contains, startswith, endswith, exists.

        Returns a JSON array of matching items.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        valid_ops = {"eq", "ne", "gt", "lt", "gte", "lte", "contains", "startswith", "endswith", "exists"}
        if operator not in valid_ops:
            return json.dumps({"error": f"Unknown operator '{operator}'. Valid: {sorted(valid_ops)}"})

        operand = value
        if isinstance(operand, str):
            # Try to coerce numeric strings for comparison operators
            if operator in {"gt", "lt", "gte", "lte"}:
                try:
                    operand = float(operand)
                except ValueError:
                    pass

        result = [item for item in items if _match(_get_field(item, field), operator, operand)]
        return json.dumps(result)

    def mr_reduce(self, data: str, field: str, operation: str, separator: str = ", ") -> str:
        """Reduce a list to a scalar by aggregating *field* values.

        Operations: count, sum, avg, min, max, first, last, join, unique.

        *separator* is used by the ``join`` operation (default ", ").

        Returns a JSON scalar (number, string, or array for ``unique``).
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        valid_ops = {"count", "sum", "avg", "min", "max", "first", "last", "join", "unique"}
        if operation not in valid_ops:
            return json.dumps({"error": f"Unknown operation '{operation}'. Valid: {sorted(valid_ops)}"})

        if operation == "count":
            return json.dumps(len(items))
        if operation == "first":
            if not items:
                return json.dumps(None)
            return json.dumps(_get_field(items[0], field))
        if operation == "last":
            if not items:
                return json.dumps(None)
            return json.dumps(_get_field(items[-1], field))

        values = [_get_field(item, field) for item in items]

        if operation == "join":
            return json.dumps(separator.join(str(v) for v in values if v is not None))
        if operation == "unique":
            seen = []
            seen_set = set()
            for v in values:
                key = json.dumps(v, sort_keys=True)
                if key not in seen_set:
                    seen_set.add(key)
                    seen.append(v)
            return json.dumps(seen)

        # Numeric ops
        nums = []
        for v in values:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass

        if not nums:
            return json.dumps({"error": f"No numeric values found for field '{field}'"})

        if operation == "sum":
            return json.dumps(sum(nums))
        if operation == "min":
            return json.dumps(min(nums))
        if operation == "max":
            return json.dumps(max(nums))
        if operation == "avg":
            return json.dumps(sum(nums) / len(nums))

        return json.dumps({"error": f"Unhandled operation '{operation}'"})

    def mr_sort(self, data: str, field: str, reverse: bool = False) -> str:
        """Sort a list by *field* (ascending by default).

        Returns a sorted JSON array.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        def key_fn(item: Any):
            v = _get_field(item, field)
            if v is None:
                return (1, "")          # None sorts last
            if isinstance(v, (int, float)):
                return (0, v)
            return (0, str(v))

        try:
            sorted_items = sorted(items, key=key_fn, reverse=reverse)
        except Exception as exc:
            return json.dumps({"error": f"Sort failed: {exc}"})

        return json.dumps(sorted_items)

    def mr_group(self, data: str, field: str) -> str:
        """Group items by the value of *field*.

        Returns a JSON object: ``{group_value: [items], ...}``.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        groups: Dict[str, List[Any]] = {}
        for item in items:
            key = str(_get_field(item, field))
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return json.dumps(groups)

    def mr_flatten(self, data: str) -> str:
        """Flatten a list of lists into a single list.

        Each element that is itself a list is unpacked; others are kept as-is.

        Returns a flat JSON array.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        result = []
        for item in items:
            if isinstance(item, list):
                result.extend(item)
            else:
                result.append(item)
        return json.dumps(result)

    def mr_zip(self, left: str, right: str) -> str:
        """Zip two JSON arrays into ``[{\"left\": ..., \"right\": ...}, ...]``.

        The result length equals the shorter of the two arrays.

        Returns a JSON array of dicts.
        """
        try:
            l_items = self._parse(left)
            r_items = self._parse(right)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        result = [{"left": l, "right": r} for l, r in zip(l_items, r_items)]
        return json.dumps(result)

    def mr_pick(self, data: str, fields: List[str]) -> str:
        """Keep only the specified *fields* in each dict item.

        Returns a JSON array of dicts with only the requested keys.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        result = []
        for item in items:
            if isinstance(item, dict):
                result.append({k: item[k] for k in fields if k in item})
            else:
                result.append(item)
        return json.dumps(result)

    def mr_slice(self, data: str, start: int = 0, end: Optional[int] = None) -> str:
        """Return a slice of the list (Python-style: start inclusive, end exclusive).

        Returns a JSON array.
        """
        try:
            items = self._parse(data)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps(items[start:end])

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "map_reduce"

    @property
    def description(self) -> str:
        return (
            "Map, filter, sort, group, and reduce JSON arrays without CodeTool. "
            "Dot-notation field access. Chainable with JSONTool, TableTool, HTTPTool. "
            "Operations: mr_map, mr_filter, mr_reduce, mr_sort, mr_group, "
            "mr_flatten, mr_zip, mr_pick, mr_slice. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "mr_map",
                "description": (
                    "Extract a field from every item in a JSON array. "
                    "Optional transform: upper, lower, strip, len, bool, str, int, float, abs, not."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "JSON array"},
                        "field": {"type": "string", "description": "Field name (dot-notation OK; '.' for whole item)"},
                        "transform": {"type": "string", "description": "Optional transform (upper/lower/len/int/float/...)"},
                    },
                    "required": ["data", "field"],
                },
            },
            {
                "name": "mr_filter",
                "description": (
                    "Keep items where field satisfies operator against value. "
                    "Operators: eq, ne, gt, lt, gte, lte, contains, startswith, endswith, exists."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "field": {"type": "string"},
                        "operator": {"type": "string"},
                        "value": {"description": "Comparison value (omit for 'exists')"},
                    },
                    "required": ["data", "field", "operator"],
                },
            },
            {
                "name": "mr_reduce",
                "description": (
                    "Reduce a list to a scalar. "
                    "Operations: count, sum, avg, min, max, first, last, join, unique."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "field": {"type": "string"},
                        "operation": {"type": "string"},
                        "separator": {"type": "string", "description": "Separator for join (default ', ')"},
                    },
                    "required": ["data", "field", "operation"],
                },
            },
            {
                "name": "mr_sort",
                "description": "Sort a JSON array by a field (ascending by default).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "field": {"type": "string"},
                        "reverse": {"type": "boolean", "description": "Sort descending if true"},
                    },
                    "required": ["data", "field"],
                },
            },
            {
                "name": "mr_group",
                "description": "Group items by field value → {group_key: [items]}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "field": {"type": "string"},
                    },
                    "required": ["data", "field"],
                },
            },
            {
                "name": "mr_flatten",
                "description": "Flatten a list of lists into a single list.",
                "input_schema": {
                    "type": "object",
                    "properties": {"data": {"type": "string"}},
                    "required": ["data"],
                },
            },
            {
                "name": "mr_zip",
                "description": "Zip two JSON arrays into [{left, right}, ...] pairs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "string"},
                        "right": {"type": "string"},
                    },
                    "required": ["left", "right"],
                },
            },
            {
                "name": "mr_pick",
                "description": "Keep only specified fields in each dict item.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Field names to keep",
                        },
                    },
                    "required": ["data", "fields"],
                },
            },
            {
                "name": "mr_slice",
                "description": "Return a slice of the list (start inclusive, end exclusive).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "start": {"type": "integer", "description": "Start index (default 0)"},
                        "end": {"type": "integer", "description": "End index (default: end of list)"},
                    },
                    "required": ["data"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "mr_map":
            return self.mr_map(**arguments)
        if tool_name == "mr_filter":
            return self.mr_filter(**arguments)
        if tool_name == "mr_reduce":
            return self.mr_reduce(**arguments)
        if tool_name == "mr_sort":
            return self.mr_sort(**arguments)
        if tool_name == "mr_group":
            return self.mr_group(**arguments)
        if tool_name == "mr_flatten":
            return self.mr_flatten(**arguments)
        if tool_name == "mr_zip":
            return self.mr_zip(**arguments)
        if tool_name == "mr_pick":
            return self.mr_pick(**arguments)
        if tool_name == "mr_slice":
            return self.mr_slice(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
