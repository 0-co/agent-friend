"""batch_tool.py — BatchTool for agent-friend (stdlib only).

Process collections of items in batches: apply per-item transformations,
collect results, handle errors, and track progress — all in stdlib.

Features:
* batch_map       — apply a registered function to each item in a list
* batch_filter    — filter items using a registered predicate function
* batch_reduce    — fold items using an accumulator function
* batch_partition — split into passing/failing lists by predicate
* fn_define       — register a named Python function (def fn(item): ...)
* batch_chunk     — split a list into equal-sized chunks
* batch_zip       — zip multiple equal-length lists into list of dicts
* batch_stats     — summary stats for a previous batch run

Usage::

    tool = BatchTool()

    tool.fn_define("double", "def fn(item): return item * 2")
    tool.batch_map([1, 2, 3, 4, 5], fn="double")
    # {"results": [2, 4, 6, 8, 10], "ok": 5, "errors": 0, ...}

    tool.fn_define("is_even", "def fn(item): return item % 2 == 0")
    tool.batch_filter([1, 2, 3, 4, 5], fn="is_even")
    # {"results": [2, 4], "total": 5, "kept": 2}
"""

import json
import time
from typing import Any, Dict, List, Optional

from .base import BaseTool


# ── built-in functions ────────────────────────────────────────────────────

def _fn_identity(item: Any) -> Any:
    return item


def _fn_str(item: Any) -> Any:
    return str(item)


def _fn_int(item: Any) -> Any:
    return int(item)


def _fn_float(item: Any) -> Any:
    return float(item)


def _fn_upper(item: Any) -> Any:
    return str(item).upper()


def _fn_lower(item: Any) -> Any:
    return str(item).lower()


def _fn_strip(item: Any) -> Any:
    return str(item).strip() if isinstance(item, str) else item


def _fn_len(item: Any) -> Any:
    return len(item)


def _fn_bool(item: Any) -> Any:
    return bool(item)


def _fn_not(item: Any) -> Any:
    return not item


def _fn_is_none(item: Any) -> Any:
    return item is None


def _fn_is_truthy(item: Any) -> Any:
    return bool(item)


def _fn_negate(item: Any) -> Any:
    return -item


def _fn_abs(item: Any) -> Any:
    return abs(item)


def _fn_double(item: Any) -> Any:
    return item * 2


def _fn_square(item: Any) -> Any:
    return item * item


_BUILTIN_FNS: Dict[str, Any] = {
    "identity": _fn_identity,
    "str": _fn_str,
    "int": _fn_int,
    "float": _fn_float,
    "upper": _fn_upper,
    "lower": _fn_lower,
    "strip": _fn_strip,
    "len": _fn_len,
    "bool": _fn_bool,
    "not": _fn_not,
    "is_none": _fn_is_none,
    "is_truthy": _fn_is_truthy,
    "negate": _fn_negate,
    "abs": _fn_abs,
    "double": _fn_double,
    "square": _fn_square,
}

# ── built-in reducers ─────────────────────────────────────────────────────

def _red_sum(acc: Any, item: Any) -> Any:
    return acc + item


def _red_product(acc: Any, item: Any) -> Any:
    return acc * item


def _red_max(acc: Any, item: Any) -> Any:
    return max(acc, item)


def _red_min(acc: Any, item: Any) -> Any:
    return min(acc, item)


def _red_concat(acc: Any, item: Any) -> Any:
    return str(acc) + str(item)


_BUILTIN_REDUCERS: Dict[str, Any] = {
    "sum": _red_sum,
    "product": _red_product,
    "max": _red_max,
    "min": _red_min,
    "concat": _red_concat,
}


# ── tool ──────────────────────────────────────────────────────────────────

class BatchTool(BaseTool):
    """Process lists of items in batches with map/filter/reduce/partition.

    Register named functions (Python source or built-ins) and apply them
    to collections. Errors per item are captured, not raised.
    """

    MAX_ITEMS = 10_000
    MAX_FNS = 200

    def __init__(self):
        self._custom_fns: Dict[str, Any] = {}
        self._custom_reducers: Dict[str, Any] = {}
        self._last_run: Dict[str, Any] = {}

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_fn(self, name: str):
        if name in self._custom_fns:
            return self._custom_fns[name]
        return _BUILTIN_FNS.get(name)

    def _resolve_reducer(self, name: str):
        if name in self._custom_reducers:
            return self._custom_reducers[name]
        return _BUILTIN_REDUCERS.get(name)

    def _apply(self, fn, item: Any) -> tuple:
        """Returns (result, error_str). error_str is '' on success."""
        try:
            return fn(item), ""
        except Exception as e:
            return None, str(e)

    # ── public API ────────────────────────────────────────────────────

    def fn_define(self, name: str, source: str, is_reducer: bool = False) -> str:
        """Register a named function from Python source.

        The source must define exactly one function:
        - For map/filter/partition: ``def fn(item): ...``
        - For reduce: ``def fn(acc, item): ...`` (set is_reducer=True)

        Returns ``{name, registered, is_reducer}``.
        """
        if not name or not name.strip():
            return json.dumps({"error": "name must be non-empty"})
        if not source or not source.strip():
            return json.dumps({"error": "source must be non-empty"})
        if name in _BUILTIN_FNS or name in _BUILTIN_REDUCERS:
            return json.dumps({"error": f"{name!r} conflicts with a built-in name"})

        try:
            ns: Dict = {}
            exec(compile(source, "<fn>", "exec"), ns)
            fn = ns.get("fn")
            if fn is None or not callable(fn):
                return json.dumps({"error": "source must define a callable named 'fn'"})
            if is_reducer:
                self._custom_reducers[name] = fn
            else:
                self._custom_fns[name] = fn
            return json.dumps({"name": name, "registered": True, "is_reducer": is_reducer})
        except SyntaxError as e:
            return json.dumps({"error": f"syntax error: {e}"})
        except Exception as e:
            return json.dumps({"error": f"compilation error: {e}"})

    def batch_map(
        self,
        items: List[Any],
        fn: str,
        on_error: str = "null",
    ) -> str:
        """Apply *fn* to each item. Returns ``{results, ok, errors, elapsed_ms}``.

        *on_error*: what to put in results for failed items:
          ``"null"`` (None), ``"skip"`` (omit from results), ``"raise"`` (fail fast).
        """
        if not isinstance(items, list):
            return json.dumps({"error": "items must be a list"})
        if len(items) > self.MAX_ITEMS:
            return json.dumps({"error": f"too many items (max {self.MAX_ITEMS})"})
        if on_error not in ("null", "skip", "raise"):
            return json.dumps({"error": "on_error must be null/skip/raise"})

        func = self._resolve_fn(fn)
        if func is None:
            return json.dumps({"error": f"unknown function: {fn!r}"})

        t0 = time.monotonic()
        results = []
        ok_count = 0
        errors = []

        for i, item in enumerate(items):
            result, err = self._apply(func, item)
            if err:
                errors.append({"index": i, "item": item, "error": err})
                if on_error == "raise":
                    return json.dumps({
                        "error": f"item {i} failed: {err}",
                        "index": i,
                    })
                elif on_error == "null":
                    results.append(None)
                # skip: don't append
            else:
                results.append(result)
                ok_count += 1

        elapsed = round((time.monotonic() - t0) * 1000, 3)
        record = {
            "results": results,
            "ok": ok_count,
            "errors": len(errors),
            "error_details": errors,
            "total": len(items),
            "elapsed_ms": elapsed,
        }
        self._last_run = record
        return json.dumps(record)

    def batch_filter(
        self,
        items: List[Any],
        fn: str,
    ) -> str:
        """Keep items where *fn(item)* is truthy.

        Returns ``{results, total, kept, rejected, elapsed_ms}``.
        """
        if not isinstance(items, list):
            return json.dumps({"error": "items must be a list"})
        if len(items) > self.MAX_ITEMS:
            return json.dumps({"error": f"too many items (max {self.MAX_ITEMS})"})

        func = self._resolve_fn(fn)
        if func is None:
            return json.dumps({"error": f"unknown function: {fn!r}"})

        t0 = time.monotonic()
        kept = []
        errors = []

        for i, item in enumerate(items):
            result, err = self._apply(func, item)
            if err:
                errors.append({"index": i, "error": err})
            elif result:
                kept.append(item)

        elapsed = round((time.monotonic() - t0) * 1000, 3)
        return json.dumps({
            "results": kept,
            "total": len(items),
            "kept": len(kept),
            "rejected": len(items) - len(kept) - len(errors),
            "errors": len(errors),
            "elapsed_ms": elapsed,
        })

    def batch_reduce(
        self,
        items: List[Any],
        fn: str,
        initial: Any = None,
    ) -> str:
        """Fold items with *fn(acc, item)*.

        *initial*: starting accumulator (None uses the first item).

        Returns ``{result, total, elapsed_ms}``.
        """
        if not isinstance(items, list):
            return json.dumps({"error": "items must be a list"})
        if not items:
            return json.dumps({"error": "items must be non-empty"})

        func = self._resolve_reducer(fn)
        if func is None:
            return json.dumps({"error": f"unknown reducer: {fn!r}"})

        t0 = time.monotonic()
        try:
            if initial is None:
                acc = items[0]
                rest = items[1:]
            else:
                acc = initial
                rest = items

            for item in rest:
                acc = func(acc, item)

            elapsed = round((time.monotonic() - t0) * 1000, 3)
            return json.dumps({"result": acc, "total": len(items), "elapsed_ms": elapsed})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def batch_partition(
        self,
        items: List[Any],
        fn: str,
    ) -> str:
        """Split items into two lists: *passing* (truthy) and *failing* (falsy).

        Returns ``{passing, failing, total, pass_count, fail_count, elapsed_ms}``.
        """
        if not isinstance(items, list):
            return json.dumps({"error": "items must be a list"})

        func = self._resolve_fn(fn)
        if func is None:
            return json.dumps({"error": f"unknown function: {fn!r}"})

        t0 = time.monotonic()
        passing = []
        failing = []
        errors = []

        for i, item in enumerate(items):
            result, err = self._apply(func, item)
            if err:
                errors.append({"index": i, "error": err})
                failing.append(item)
            elif result:
                passing.append(item)
            else:
                failing.append(item)

        elapsed = round((time.monotonic() - t0) * 1000, 3)
        return json.dumps({
            "passing": passing,
            "failing": failing,
            "total": len(items),
            "pass_count": len(passing),
            "fail_count": len(failing),
            "errors": len(errors),
            "elapsed_ms": elapsed,
        })

    def batch_chunk(self, items: List[Any], size: int) -> str:
        """Split *items* into chunks of *size*.

        Returns ``{chunks, chunk_count, total}``.
        """
        if not isinstance(items, list):
            return json.dumps({"error": "items must be a list"})
        if size < 1:
            return json.dumps({"error": "size must be >= 1"})
        chunks = [items[i: i + size] for i in range(0, len(items), size)]
        return json.dumps({"chunks": chunks, "chunk_count": len(chunks), "total": len(items)})

    def batch_zip(self, keys: List[str], *lists) -> str:
        """Zip multiple lists into a list of dicts keyed by *keys*.

        Returns ``{results, count}``.
        """
        if not keys:
            return json.dumps({"error": "keys must be non-empty"})
        if not lists:
            return json.dumps({"error": "at least one list required"})
        if len(keys) != len(lists):
            return json.dumps({"error": f"keys ({len(keys)}) must match number of lists ({len(lists)})"})
        lengths = [len(lst) for lst in lists]
        if len(set(lengths)) != 1:
            return json.dumps({"error": f"all lists must have equal length: {lengths}"})
        results = [
            {k: v for k, v in zip(keys, row)}
            for row in zip(*lists)
        ]
        return json.dumps({"results": results, "count": len(results)})

    def batch_stats(self) -> str:
        """Return stats from the last batch_map run.

        Returns ``{ok, errors, total, elapsed_ms}`` or ``{error}`` if no run yet.
        """
        if not self._last_run:
            return json.dumps({"error": "no batch run recorded yet"})
        return json.dumps({
            "ok": self._last_run.get("ok", 0),
            "errors": self._last_run.get("errors", 0),
            "total": self._last_run.get("total", 0),
            "elapsed_ms": self._last_run.get("elapsed_ms", 0),
        })

    def builtin_fns(self) -> str:
        """List built-in map/filter function names and reducer names.

        Returns ``{fns, reducers}``.
        """
        return json.dumps({
            "fns": sorted(_BUILTIN_FNS.keys()),
            "reducers": sorted(_BUILTIN_REDUCERS.keys()),
        })

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "batch"

    @property
    def description(self) -> str:
        return (
            "Batch processing: map/filter/reduce/partition lists. Register named "
            "Python functions (fn_define). Built-in fns: identity/str/int/float/"
            "upper/lower/strip/len/bool/not/negate/abs/double/square. Built-in "
            "reducers: sum/product/max/min/concat. batch_chunk, batch_zip. Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "fn_define",
                "description": "Register a named function. `def fn(item): ...` for map/filter. `def fn(acc, item): ...` for reduce (is_reducer=True).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Function name"},
                        "source": {"type": "string", "description": "Python source code"},
                        "is_reducer": {"type": "boolean", "description": "True for reducers"},
                    },
                    "required": ["name", "source"],
                },
            },
            {
                "name": "batch_map",
                "description": "Apply fn to each item. on_error: null/skip/raise. Returns {results, ok, errors, total, elapsed_ms}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "Items to process"},
                        "fn": {"type": "string", "description": "Function name"},
                        "on_error": {"type": "string", "description": "null/skip/raise"},
                    },
                    "required": ["items", "fn"],
                },
            },
            {
                "name": "batch_filter",
                "description": "Keep items where fn(item) is truthy. Returns {results, total, kept, rejected}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "Items to filter"},
                        "fn": {"type": "string", "description": "Predicate function name"},
                    },
                    "required": ["items", "fn"],
                },
            },
            {
                "name": "batch_reduce",
                "description": "Fold items with fn(acc, item). initial=starting accumulator. Built-in reducers: sum/product/max/min/concat.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "Items to reduce"},
                        "fn": {"type": "string", "description": "Reducer function name"},
                        "initial": {"description": "Starting accumulator"},
                    },
                    "required": ["items", "fn"],
                },
            },
            {
                "name": "batch_partition",
                "description": "Split into passing (truthy) and failing (falsy) lists. Returns {passing, failing, pass_count, fail_count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "Items to partition"},
                        "fn": {"type": "string", "description": "Predicate function name"},
                    },
                    "required": ["items", "fn"],
                },
            },
            {
                "name": "batch_chunk",
                "description": "Split list into chunks of `size`. Returns {chunks, chunk_count, total}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "Items to chunk"},
                        "size": {"type": "integer", "description": "Chunk size"},
                    },
                    "required": ["items", "size"],
                },
            },
            {
                "name": "batch_zip",
                "description": "Zip multiple lists into list of dicts with given keys. Returns {results, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keys": {"type": "array", "items": {"type": "string"}, "description": "Dict key names"},
                        "lists": {"type": "array", "items": {"type": "array"}, "description": "Lists to zip"},
                    },
                    "required": ["keys"],
                },
            },
            {
                "name": "batch_stats",
                "description": "Stats from last batch_map run: ok, errors, total, elapsed_ms.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "builtin_fns",
                "description": "List built-in function names (fns + reducers).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "fn_define":
            return self.fn_define(**arguments)
        if tool_name == "batch_map":
            return self.batch_map(**arguments)
        if tool_name == "batch_filter":
            return self.batch_filter(**arguments)
        if tool_name == "batch_reduce":
            return self.batch_reduce(**arguments)
        if tool_name == "batch_partition":
            return self.batch_partition(**arguments)
        if tool_name == "batch_chunk":
            return self.batch_chunk(**arguments)
        if tool_name == "batch_zip":
            # batch_zip gets lists as a separate arg in execute
            keys = arguments.get("keys", [])
            lists = arguments.get("lists", [])
            return self.batch_zip(keys, *lists)
        if tool_name == "batch_stats":
            return self.batch_stats(**arguments)
        if tool_name == "builtin_fns":
            return self.builtin_fns(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
