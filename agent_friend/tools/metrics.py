"""metrics.py — MetricsTool for agent-friend (stdlib only).

Agents run tasks silently, accumulating tool calls, errors, and token spend
without any record of what happened. MetricsTool gives them observability:
counters that count, gauges that hold a value, and timers that measure duration.

All metrics are in-memory and session-scoped. Export to JSON or Prometheus
text format when you need them outside the agent.

Usage::

    tool = MetricsTool()
    tool.metric_increment("api_calls")
    tool.metric_increment("api_calls", 3)
    tool.metric_gauge("queue_depth", 42)
    timer_id = tool.metric_timer_start("search_duration")
    # ... do work ...
    tool.metric_timer_stop(timer_id)
    tool.metric_summary()
    # {"api_calls": {"type": "counter", "count": 4, "total": 4, ...}, ...}
    tool.metric_export("prometheus")
    # # TYPE api_calls counter\\napi_calls_total 4\\n...
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseTool


class MetricsTool(BaseTool):
    """Session-scoped metrics: counters, gauges, and timers.

    All operations are stdlib-only — zero dependencies.
    Metrics are in-memory and reset when the tool instance is recreated.
    """

    def __init__(self) -> None:
        # name -> metric dict
        self._metrics: Dict[str, Dict[str, Any]] = {}
        # timer_id -> {name, start_time}
        self._active_timers: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "metrics"

    @property
    def description(self) -> str:
        return (
            "Track custom metrics for your agent session: increment counters, "
            "set gauges, measure time with timers, and export everything as JSON "
            "or Prometheus text format."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "metric_increment",
                "description": (
                    "Increment a counter metric by a value (default 1). "
                    "Creates the counter if it doesn't exist. "
                    "Tracks: count (number of increments), total, min, max, last. "
                    "Returns the updated counter state."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Metric name (e.g. 'api_calls', 'errors')."},
                        "value": {"type": "number", "description": "Amount to increment by (default 1.0)."},
                        "tags": {
                            "type": "object",
                            "description": "Optional key-value tags (e.g. {tool: 'search', status: 'ok'}).",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "metric_gauge",
                "description": (
                    "Set a gauge metric to a specific value. "
                    "Gauges hold the last value set — useful for queue depth, "
                    "memory usage, active connections. "
                    "Returns the updated gauge state."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Metric name (e.g. 'queue_depth', 'active_sessions')."},
                        "value": {"type": "number", "description": "Current value to set."},
                        "tags": {
                            "type": "object",
                            "description": "Optional key-value tags.",
                        },
                    },
                    "required": ["name", "value"],
                },
            },
            {
                "name": "metric_timer_start",
                "description": (
                    "Start a timer for measuring elapsed time. "
                    "Returns a timer_id string — pass it to metric_timer_stop. "
                    "Multiple timers with the same name can run concurrently."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Timer name (e.g. 'search_duration', 'llm_call')."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "metric_timer_stop",
                "description": (
                    "Stop a running timer and record its duration in milliseconds. "
                    "Accumulates: count, total_ms, min_ms, max_ms, avg_ms, last_ms. "
                    "Returns the recorded duration and updated timer state."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timer_id": {"type": "string", "description": "Timer ID returned by metric_timer_start."},
                    },
                    "required": ["timer_id"],
                },
            },
            {
                "name": "metric_get",
                "description": (
                    "Get the current state of a specific metric. "
                    "Returns the full metric dict, or an error if not found."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Metric name."},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "metric_list",
                "description": (
                    "List all metric names and their types. "
                    "Returns [{name, type}, ...] sorted by name."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "metric_summary",
                "description": (
                    "Get a summary of all metrics as a single dict. "
                    "Keys are metric names, values are metric state dicts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "metric_reset",
                "description": (
                    "Reset a metric to its initial state (zero/empty). "
                    "If name is not provided, resets ALL metrics. "
                    "Returns the count of metrics reset."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Metric name to reset. Omit to reset all."},
                    },
                },
            },
            {
                "name": "metric_export",
                "description": (
                    "Export all metrics in a structured format. "
                    "format='json' (default): returns a JSON string. "
                    "format='prometheus': returns Prometheus text exposition format. "
                    "Useful for saving metrics to a file or sending to a monitoring system."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "Export format: 'json' (default) or 'prometheus'.",
                        },
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    def metric_increment(
        self,
        name: str,
        value: float = 1.0,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if name not in self._metrics:
            self._metrics[name] = {
                "type": "counter",
                "count": 0,
                "total": 0.0,
                "min": None,
                "max": None,
                "last": None,
                "tags": tags or {},
            }
        m = self._metrics[name]
        if m["type"] != "counter":
            return {"error": f"Metric '{name}' exists as type '{m['type']}', not counter"}

        m["count"] += 1
        m["total"] += value
        m["last"] = value
        m["min"] = value if m["min"] is None else min(m["min"], value)
        m["max"] = value if m["max"] is None else max(m["max"], value)
        if tags:
            m["tags"].update(tags)

        return {"name": name, **m}

    # ------------------------------------------------------------------
    # Gauges
    # ------------------------------------------------------------------

    def metric_gauge(
        self,
        name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if name in self._metrics and self._metrics[name]["type"] != "gauge":
            return {"error": f"Metric '{name}' exists as type '{self._metrics[name]['type']}', not gauge"}

        self._metrics[name] = {
            "type": "gauge",
            "value": value,
            "tags": tags or {},
        }
        return {"name": name, **self._metrics[name]}

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def metric_timer_start(self, name: str) -> Dict[str, Any]:
        timer_id = str(uuid.uuid4())
        self._active_timers[timer_id] = {
            "name": name,
            "start_time": time.monotonic(),
        }
        return {"timer_id": timer_id, "name": name, "status": "running"}

    def metric_timer_stop(self, timer_id: str) -> Dict[str, Any]:
        if timer_id not in self._active_timers:
            return {"error": f"Timer '{timer_id}' not found or already stopped"}

        entry = self._active_timers.pop(timer_id)
        elapsed_ms = (time.monotonic() - entry["start_time"]) * 1000.0
        name = entry["name"]

        if name not in self._metrics:
            self._metrics[name] = {
                "type": "timer",
                "count": 0,
                "total_ms": 0.0,
                "min_ms": None,
                "max_ms": None,
                "last_ms": None,
            }
        m = self._metrics[name]
        if m["type"] != "timer":
            return {"error": f"Metric '{name}' exists as type '{m['type']}', not timer"}

        m["count"] += 1
        m["total_ms"] += elapsed_ms
        m["last_ms"] = elapsed_ms
        m["min_ms"] = elapsed_ms if m["min_ms"] is None else min(m["min_ms"], elapsed_ms)
        m["max_ms"] = elapsed_ms if m["max_ms"] is None else max(m["max_ms"], elapsed_ms)
        m["avg_ms"] = m["total_ms"] / m["count"]

        return {"name": name, "elapsed_ms": round(elapsed_ms, 3), **m}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def metric_get(self, name: str) -> Dict[str, Any]:
        if name not in self._metrics:
            return {"error": f"Metric '{name}' not found"}
        return {"name": name, **self._metrics[name]}

    def metric_list(self) -> List[Dict[str, str]]:
        return sorted(
            [{"name": n, "type": m["type"]} for n, m in self._metrics.items()],
            key=lambda x: x["name"],
        )

    def metric_summary(self) -> Dict[str, Any]:
        return {name: dict(m) for name, m in self._metrics.items()}

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def metric_reset(self, name: Optional[str] = None) -> Dict[str, Any]:
        if name is not None:
            if name not in self._metrics:
                return {"error": f"Metric '{name}' not found", "reset_count": 0}
            del self._metrics[name]
            return {"reset_count": 1, "reset": [name]}
        else:
            count = len(self._metrics)
            names = list(self._metrics.keys())
            self._metrics.clear()
            self._active_timers.clear()
            return {"reset_count": count, "reset": names}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def metric_export(self, format: str = "json") -> str:
        fmt = format.lower().strip()
        if fmt == "prometheus":
            lines = []
            for name, m in sorted(self._metrics.items()):
                safe_name = name.replace("-", "_").replace(".", "_").replace(" ", "_")
                mtype = m["type"]
                if mtype == "counter":
                    lines.append(f"# TYPE {safe_name} counter")
                    lines.append(f"{safe_name}_total {m['total']}")
                    lines.append(f"{safe_name}_count {m['count']}")
                elif mtype == "gauge":
                    lines.append(f"# TYPE {safe_name} gauge")
                    lines.append(f"{safe_name} {m['value']}")
                elif mtype == "timer":
                    lines.append(f"# TYPE {safe_name}_ms summary")
                    lines.append(f"{safe_name}_ms_count {m['count']}")
                    lines.append(f"{safe_name}_ms_sum {round(m['total_ms'], 3)}")
                    if m["min_ms"] is not None:
                        lines.append(f"{safe_name}_ms_min {round(m['min_ms'], 3)}")
                        lines.append(f"{safe_name}_ms_max {round(m['max_ms'], 3)}")
                        lines.append(f"{safe_name}_ms_avg {round(m['avg_ms'], 3)}")
            return "\n".join(lines)
        else:
            return json.dumps(self.metric_summary(), indent=2)

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "metric_increment":
                return json.dumps(
                    self.metric_increment(
                        arguments["name"],
                        float(arguments.get("value", 1.0)),
                        arguments.get("tags"),
                    )
                )

            elif tool_name == "metric_gauge":
                return json.dumps(
                    self.metric_gauge(
                        arguments["name"],
                        float(arguments["value"]),
                        arguments.get("tags"),
                    )
                )

            elif tool_name == "metric_timer_start":
                return json.dumps(self.metric_timer_start(arguments["name"]))

            elif tool_name == "metric_timer_stop":
                return json.dumps(self.metric_timer_stop(arguments["timer_id"]))

            elif tool_name == "metric_get":
                return json.dumps(self.metric_get(arguments["name"]))

            elif tool_name == "metric_list":
                return json.dumps(self.metric_list())

            elif tool_name == "metric_summary":
                return json.dumps(self.metric_summary())

            elif tool_name == "metric_reset":
                return json.dumps(self.metric_reset(arguments.get("name")))

            elif tool_name == "metric_export":
                result = self.metric_export(arguments.get("format", "json"))
                return json.dumps({"output": result})

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc)})
