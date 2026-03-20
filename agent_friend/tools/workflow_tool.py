"""workflow_tool.py — WorkflowTool for agent-friend (stdlib only).

Lightweight workflow / pipeline runner for agent orchestration.
Define steps, chain them, run with shared context, handle retries
and conditional branching — all in-process, zero dependencies.

Features:
* workflow_define — register a named workflow with ordered steps
* workflow_run — execute a workflow with input data, returns step results
* step_define — register a named step (Python callable stored in registry)
* workflow_list — list registered workflows
* workflow_get — inspect a workflow definition
* workflow_delete — remove a workflow
* workflow_status — get execution history summary

Usage::

    tool = WorkflowTool()

    # Register a workflow with three steps
    tool.workflow_define("etl", steps=[
        {"name": "extract", "fn": "upper"},
        {"name": "transform", "fn": "strip"},
        {"name": "load", "fn": "identity"},
    ])

    tool.workflow_run("etl", input="  hello world  ")
    # {"workflow": "etl", "output": "  HELLO WORLD  ", "steps": [...]}
"""

import json
import time
import traceback
from typing import Any, Dict, List, Optional

from .base import BaseTool


# ── built-in step functions ───────────────────────────────────────────────

def _fn_identity(x: Any, ctx: Dict) -> Any:
    return x


def _fn_upper(x: Any, ctx: Dict) -> Any:
    return str(x).upper() if isinstance(x, str) else x


def _fn_lower(x: Any, ctx: Dict) -> Any:
    return str(x).lower() if isinstance(x, str) else x


def _fn_strip(x: Any, ctx: Dict) -> Any:
    return str(x).strip() if isinstance(x, str) else x


def _fn_to_int(x: Any, ctx: Dict) -> Any:
    try:
        return int(x)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {x!r} to int")


def _fn_to_float(x: Any, ctx: Dict) -> Any:
    try:
        return float(x)
    except (ValueError, TypeError):
        raise ValueError(f"Cannot convert {x!r} to float")


def _fn_to_str(x: Any, ctx: Dict) -> Any:
    return str(x)


def _fn_to_list(x: Any, ctx: Dict) -> Any:
    if isinstance(x, list):
        return x
    if isinstance(x, (str, bytes)):
        return list(x)
    try:
        return list(x)
    except TypeError:
        return [x]


def _fn_reverse(x: Any, ctx: Dict) -> Any:
    if isinstance(x, str):
        return x[::-1]
    if isinstance(x, list):
        return list(reversed(x))
    return x


def _fn_length(x: Any, ctx: Dict) -> Any:
    try:
        return len(x)
    except TypeError:
        return 0


def _fn_sum_list(x: Any, ctx: Dict) -> Any:
    if isinstance(x, (list, tuple)):
        return sum(x)
    return x


def _fn_sort(x: Any, ctx: Dict) -> Any:
    if isinstance(x, list):
        return sorted(x)
    return x


def _fn_unique(x: Any, ctx: Dict) -> Any:
    if isinstance(x, list):
        seen = []
        for item in x:
            if item not in seen:
                seen.append(item)
        return seen
    return x


def _fn_flatten(x: Any, ctx: Dict) -> Any:
    if not isinstance(x, list):
        return x
    result = []
    for item in x:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def _fn_noop(x: Any, ctx: Dict) -> Any:
    """Does nothing, passes value through. Useful for conditional branches."""
    return x


_BUILTIN_FNS: Dict[str, Any] = {
    "identity": _fn_identity,
    "upper": _fn_upper,
    "lower": _fn_lower,
    "strip": _fn_strip,
    "to_int": _fn_to_int,
    "to_float": _fn_to_float,
    "to_str": _fn_to_str,
    "to_list": _fn_to_list,
    "reverse": _fn_reverse,
    "length": _fn_length,
    "sum_list": _fn_sum_list,
    "sort": _fn_sort,
    "unique": _fn_unique,
    "flatten": _fn_flatten,
    "noop": _fn_noop,
}


# ── workflow datastructures ────────────────────────────────────────────────

class _Step:
    """A single step in a workflow."""

    def __init__(
        self,
        name: str,
        fn: str,
        retries: int = 0,
        on_error: str = "fail",  # "fail" | "skip" | "default"
        default: Any = None,
        condition: Optional[str] = None,  # "truthy" | "falsy" | None (always)
    ):
        self.name = name
        self.fn = fn
        self.retries = max(0, retries)
        self.on_error = on_error
        self.default = default
        self.condition = condition

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "fn": self.fn,
            "retries": self.retries,
            "on_error": self.on_error,
            "default": self.default,
            "condition": self.condition,
        }


class _Workflow:
    """A named workflow composed of ordered steps."""

    def __init__(self, name: str, steps: List[_Step], description: str = ""):
        self.name = name
        self.steps = steps
        self.description = description
        self.created_at = time.time()
        self.run_count = 0
        self.last_run: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "step_count": len(self.steps),
            "run_count": self.run_count,
            "created_at": self.created_at,
            "last_run": self.last_run,
        }


# ── tool ──────────────────────────────────────────────────────────────────

class WorkflowTool(BaseTool):
    """Lightweight workflow / pipeline runner.

    Define named workflows as sequences of steps. Each step applies a
    built-in or user-registered transform function to the current value.
    Supports retries, error handling (fail/skip/default), and conditional
    execution based on value truthiness.
    """

    MAX_WORKFLOWS = 100
    MAX_STEPS = 50
    MAX_RETRIES = 5
    MAX_HISTORY = 200

    def __init__(self):
        self._workflows: Dict[str, _Workflow] = {}
        self._custom_fns: Dict[str, Any] = {}
        self._history: List[Dict] = []  # capped run log

    # ── helpers ──────────────────────────────────────────────────────────

    def _resolve_fn(self, fn_name: str):
        if fn_name in self._custom_fns:
            return self._custom_fns[fn_name]
        if fn_name in _BUILTIN_FNS:
            return _BUILTIN_FNS[fn_name]
        return None

    def _run_step(self, step: _Step, value: Any, ctx: Dict) -> tuple:
        """Returns (new_value, status, error_msg)."""
        fn = self._resolve_fn(step.fn)
        if fn is None:
            if step.on_error == "skip":
                return value, "skipped", f"unknown fn: {step.fn}"
            if step.on_error == "default":
                return step.default, "defaulted", f"unknown fn: {step.fn}"
            return value, "failed", f"Unknown function: {step.fn!r}"

        attempts = 0
        last_err = ""
        while attempts <= step.retries:
            try:
                result = fn(value, ctx)
                return result, "ok", ""
            except Exception as e:
                last_err = str(e)
                attempts += 1

        # exhausted retries
        if step.on_error == "skip":
            return value, "skipped", last_err
        if step.on_error == "default":
            return step.default, "defaulted", last_err
        return value, "failed", last_err

    # ── public API ───────────────────────────────────────────────────────

    def workflow_define(
        self,
        name: str,
        steps: List[Dict],
        description: str = "",
    ) -> str:
        """Define a named workflow with ordered steps.

        Each step dict: ``{name, fn, retries?, on_error?, default?, condition?}``

        *fn* can be a built-in name (identity, upper, lower, strip, to_int,
        to_float, to_str, to_list, reverse, length, sum_list, sort, unique,
        flatten, noop) or a custom function registered via ``step_define``.

        *on_error*: ``"fail"`` (default, stops workflow), ``"skip"`` (pass
        current value through), ``"default"`` (use *default* value).

        *condition*: ``"truthy"`` runs step only if value is truthy;
        ``"falsy"`` only if falsy; omit/null to always run.

        Returns ``{workflow, step_count}``.
        """
        if not name or not name.strip():
            return json.dumps({"error": "workflow name must be non-empty"})
        if not isinstance(steps, list) or not steps:
            return json.dumps({"error": "steps must be a non-empty list"})
        if len(steps) > self.MAX_STEPS:
            return json.dumps({"error": f"too many steps (max {self.MAX_STEPS})"})
        if len(self._workflows) >= self.MAX_WORKFLOWS and name not in self._workflows:
            return json.dumps({"error": f"max workflows ({self.MAX_WORKFLOWS}) reached"})

        parsed_steps = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                return json.dumps({"error": f"step {i} must be a dict"})
            step_name = s.get("name", f"step_{i}")
            fn = s.get("fn", "identity")
            if not isinstance(fn, str):
                return json.dumps({"error": f"step {i} fn must be a string"})
            retries = int(s.get("retries", 0))
            if retries > self.MAX_RETRIES:
                return json.dumps({"error": f"step {i} retries exceeds max ({self.MAX_RETRIES})"})
            on_error = s.get("on_error", "fail")
            if on_error not in ("fail", "skip", "default"):
                return json.dumps({"error": f"step {i} on_error must be fail/skip/default"})
            condition = s.get("condition", None)
            if condition is not None and condition not in ("truthy", "falsy"):
                return json.dumps({"error": f"step {i} condition must be truthy/falsy or null"})
            parsed_steps.append(_Step(
                name=step_name,
                fn=fn,
                retries=retries,
                on_error=on_error,
                default=s.get("default", None),
                condition=condition,
            ))

        self._workflows[name] = _Workflow(name, parsed_steps, description)
        return json.dumps({"workflow": name, "step_count": len(parsed_steps)})

    def workflow_run(
        self,
        name: str,
        input: Any = None,
        context: Optional[Dict] = None,
    ) -> str:
        """Execute a workflow on *input* data.

        Returns ``{workflow, output, steps: [{name, status, output, error_ms}],
        elapsed_ms, ok}``.

        Execution stops on the first failed step (unless on_error is skip/default).
        A shared *context* dict is passed to each step function.
        """
        if name not in self._workflows:
            return json.dumps({"error": f"workflow {name!r} not found"})

        wf = self._workflows[name]
        ctx = dict(context) if context else {}
        value = input
        step_results = []
        ok = True
        t0 = time.monotonic()

        for step in wf.steps:
            # conditional check
            if step.condition == "truthy" and not value:
                step_results.append({"name": step.name, "status": "skipped_condition", "output": value, "error": ""})
                continue
            if step.condition == "falsy" and value:
                step_results.append({"name": step.name, "status": "skipped_condition", "output": value, "error": ""})
                continue

            st = time.monotonic()
            new_value, status, err = self._run_step(step, value, ctx)
            elapsed_step = round((time.monotonic() - st) * 1000, 3)

            step_results.append({
                "name": step.name,
                "status": status,
                "output": new_value,
                "error": err,
                "elapsed_ms": elapsed_step,
            })

            if status == "failed":
                ok = False
                break

            value = new_value

        elapsed = round((time.monotonic() - t0) * 1000, 3)
        wf.run_count += 1
        wf.last_run = time.time()

        record = {
            "workflow": name,
            "output": value,
            "steps": step_results,
            "elapsed_ms": elapsed,
            "ok": ok,
        }

        self._history.append({**record, "timestamp": time.time()})
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[-self.MAX_HISTORY:]

        return json.dumps(record)

    def step_define(self, name: str, source: str) -> str:
        """Register a custom step function from Python *source* string.

        The source must define a function named exactly ``step(value, ctx)``
        where *value* is the current pipeline value and *ctx* is the shared
        context dict.

        Example: ``"def step(value, ctx): return value * 2"``

        Returns ``{name, registered}``.
        """
        if not name or not name.strip():
            return json.dumps({"error": "name must be non-empty"})
        if name in _BUILTIN_FNS:
            return json.dumps({"error": f"{name!r} conflicts with a built-in function name"})
        if not source or not source.strip():
            return json.dumps({"error": "source must be non-empty"})

        try:
            ns: Dict = {}
            exec(compile(source, "<step>", "exec"), ns)  # noqa: S102
            fn = ns.get("step")
            if fn is None or not callable(fn):
                return json.dumps({"error": "source must define a callable named 'step'"})
            self._custom_fns[name] = fn
            return json.dumps({"name": name, "registered": True})
        except SyntaxError as e:
            return json.dumps({"error": f"syntax error: {e}"})
        except Exception as e:
            return json.dumps({"error": f"compilation error: {e}"})

    def workflow_list(self) -> str:
        """List all registered workflows.

        Returns ``{workflows: [{name, step_count, run_count, description}], count}``.
        """
        items = [
            {
                "name": wf.name,
                "step_count": len(wf.steps),
                "run_count": wf.run_count,
                "description": wf.description,
                "last_run": wf.last_run,
            }
            for wf in self._workflows.values()
        ]
        return json.dumps({"workflows": items, "count": len(items)})

    def workflow_get(self, name: str) -> str:
        """Get full definition of a workflow.

        Returns ``{name, description, steps, step_count, run_count}``.
        """
        if name not in self._workflows:
            return json.dumps({"error": f"workflow {name!r} not found"})
        return json.dumps(self._workflows[name].to_dict())

    def workflow_delete(self, name: str) -> str:
        """Delete a workflow by name.

        Returns ``{deleted, name}``.
        """
        if name not in self._workflows:
            return json.dumps({"error": f"workflow {name!r} not found"})
        del self._workflows[name]
        return json.dumps({"deleted": True, "name": name})

    def workflow_status(self) -> str:
        """Return execution history summary.

        Returns ``{total_runs, ok_runs, failed_runs, workflows, recent: [...]}``.
        """
        total = len(self._history)
        ok_runs = sum(1 for h in self._history if h.get("ok"))
        recent = self._history[-10:]
        return json.dumps({
            "total_runs": total,
            "ok_runs": ok_runs,
            "failed_runs": total - ok_runs,
            "workflows": len(self._workflows),
            "recent": [
                {
                    "workflow": h["workflow"],
                    "ok": h["ok"],
                    "elapsed_ms": h["elapsed_ms"],
                    "timestamp": h["timestamp"],
                }
                for h in recent
            ],
        })

    def builtin_fns(self) -> str:
        """List all built-in step function names.

        Returns ``{functions: [...], count}``.
        """
        names = sorted(_BUILTIN_FNS.keys())
        return json.dumps({"functions": names, "count": len(names)})

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "workflow"

    @property
    def description(self) -> str:
        return (
            "Lightweight workflow / pipeline runner. Define named workflows as "
            "step sequences, run them on data with shared context. Supports "
            "retries, on_error (fail/skip/default), conditional steps, custom "
            "step functions, and execution history. Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "workflow_define",
                "description": "Define a named workflow. Steps: {name, fn, retries?, on_error?, default?, condition?}. Use builtin_fns to list available fns.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Workflow name"},
                        "steps": {"type": "array", "items": {"type": "object"}, "description": "Ordered step definitions"},
                        "description": {"type": "string", "description": "Workflow description"},
                    },
                    "required": ["name", "steps"],
                },
            },
            {
                "name": "workflow_run",
                "description": "Run a workflow on input data. Returns {output, steps, elapsed_ms, ok}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Workflow name"},
                        "input": {"description": "Input data"},
                        "context": {"type": "object", "description": "Shared context dict"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "step_define",
                "description": "Register custom step function from Python source. Must define `def step(value, ctx): ...`.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Step function name"},
                        "source": {"type": "string", "description": "Python source code"},
                    },
                    "required": ["name", "source"],
                },
            },
            {
                "name": "workflow_list",
                "description": "List all registered workflows. Returns {workflows, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "workflow_get",
                "description": "Get full definition of a workflow by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Workflow name"}},
                    "required": ["name"],
                },
            },
            {
                "name": "workflow_delete",
                "description": "Delete a workflow by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Workflow name"}},
                    "required": ["name"],
                },
            },
            {
                "name": "workflow_status",
                "description": "Return execution history summary: total_runs, ok_runs, failed_runs, recent runs.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "builtin_fns",
                "description": "List all built-in step function names.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "workflow_define":
            return self.workflow_define(**arguments)
        if tool_name == "workflow_run":
            return self.workflow_run(**arguments)
        if tool_name == "step_define":
            return self.step_define(**arguments)
        if tool_name == "workflow_list":
            return self.workflow_list(**arguments)
        if tool_name == "workflow_get":
            return self.workflow_get(**arguments)
        if tool_name == "workflow_delete":
            return self.workflow_delete(**arguments)
        if tool_name == "workflow_status":
            return self.workflow_status(**arguments)
        if tool_name == "builtin_fns":
            return self.builtin_fns(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
