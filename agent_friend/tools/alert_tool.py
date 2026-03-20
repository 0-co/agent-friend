"""alert_tool.py — AlertTool for agent-friend (stdlib only).

Threshold-based alerting: define named alert rules, evaluate them against
incoming values, and maintain an alert history with severity levels.

Features:
* alert_define   — register a named alert rule with condition + threshold
* alert_evaluate — check a value against a rule; records firing events
* alert_list     — list all defined rules
* alert_get      — get full rule definition
* alert_delete   — remove a rule
* alert_history  — retrieve recent fired alert events
* alert_clear    — clear history for a rule (or all)
* alert_stats    — aggregate stats per rule (count, last_fired)

Supported conditions: gt, gte, lt, lte, eq, ne, between, outside,
                      contains, not_contains, is_empty, is_truthy

Usage::

    tool = AlertTool()

    tool.alert_define("high_cpu", metric="cpu_pct", condition="gt",
                      threshold=90.0, severity="critical")

    tool.alert_evaluate("high_cpu", 95.0)
    # {"rule": "high_cpu", "fired": True, "value": 95.0, "severity": "critical"}

    tool.alert_history()
    # {"events": [...], "count": 1}
"""

import json
import time
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


_VALID_CONDITIONS = frozenset([
    "gt", "gte", "lt", "lte", "eq", "ne",
    "between", "outside", "contains", "not_contains",
    "is_empty", "is_truthy",
])
_VALID_SEVERITIES = frozenset(["info", "warning", "error", "critical"])
_MAX_RULES = 200
_MAX_HISTORY = 500


def _evaluate_condition(
    condition: str,
    value: Any,
    threshold: Any,
    threshold_high: Any,
) -> bool:
    """Return True if *value* satisfies the condition."""
    try:
        if condition == "gt":
            return float(value) > float(threshold)
        if condition == "gte":
            return float(value) >= float(threshold)
        if condition == "lt":
            return float(value) < float(threshold)
        if condition == "lte":
            return float(value) <= float(threshold)
        if condition == "eq":
            return value == threshold
        if condition == "ne":
            return value != threshold
        if condition == "between":
            return float(threshold) <= float(value) <= float(threshold_high)
        if condition == "outside":
            return not (float(threshold) <= float(value) <= float(threshold_high))
        if condition == "contains":
            return threshold in value
        if condition == "not_contains":
            return threshold not in value
        if condition == "is_empty":
            return not value
        if condition == "is_truthy":
            return bool(value)
    except (TypeError, ValueError):
        return False
    return False


class _AlertRule:
    def __init__(
        self,
        name: str,
        condition: str,
        threshold: Any,
        threshold_high: Any,
        severity: str,
        message: str,
        metric: str,
        cooldown_s: float,
    ):
        self.name = name
        self.condition = condition
        self.threshold = threshold
        self.threshold_high = threshold_high
        self.severity = severity
        self.message = message
        self.metric = metric
        self.cooldown_s = cooldown_s
        self.created_at = time.time()
        self.fire_count = 0
        self.last_fired: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "condition": self.condition,
            "threshold": self.threshold,
            "threshold_high": self.threshold_high,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
            "cooldown_s": self.cooldown_s,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
            "created_at": self.created_at,
        }


class AlertTool(BaseTool):
    """Threshold-based alerting for agent workflows.

    Define named alert rules, evaluate incoming values, and maintain
    a history of fired events with severity levels.
    """

    def __init__(self):
        self._rules: Dict[str, _AlertRule] = {}
        self._history: List[Dict] = []

    # ── public API ────────────────────────────────────────────────────

    def alert_define(
        self,
        name: str,
        condition: str,
        threshold: Any = None,
        threshold_high: Any = None,
        severity: str = "warning",
        message: str = "",
        metric: str = "",
        cooldown_s: float = 0.0,
    ) -> str:
        """Define a named alert rule.

        *condition*: gt | gte | lt | lte | eq | ne | between | outside |
                     contains | not_contains | is_empty | is_truthy

        *threshold*: primary threshold value (required for most conditions).
        *threshold_high*: upper bound for 'between' and 'outside'.
        *severity*: info | warning | error | critical.
        *cooldown_s*: minimum seconds between repeated firings (0 = no limit).

        Returns ``{name, condition, severity}``.
        """
        if not name or not name.strip():
            return json.dumps({"error": "name must be non-empty"})
        if condition not in _VALID_CONDITIONS:
            return json.dumps({"error": f"condition must be one of {sorted(_VALID_CONDITIONS)}"})
        if severity not in _VALID_SEVERITIES:
            return json.dumps({"error": f"severity must be one of {sorted(_VALID_SEVERITIES)}"})
        if condition in ("between", "outside") and threshold_high is None:
            return json.dumps({"error": f"condition '{condition}' requires threshold_high"})
        if len(self._rules) >= _MAX_RULES and name not in self._rules:
            return json.dumps({"error": f"max rules ({_MAX_RULES}) reached"})

        self._rules[name] = _AlertRule(
            name=name,
            condition=condition,
            threshold=threshold,
            threshold_high=threshold_high,
            severity=severity,
            message=message,
            metric=metric,
            cooldown_s=float(cooldown_s),
        )
        return json.dumps({"name": name, "condition": condition, "severity": severity})

    def alert_evaluate(
        self,
        name: str,
        value: Any,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Evaluate *value* against rule *name*.

        Returns ``{rule, fired, value, severity, message, threshold, timestamp}``.

        If cooldown is active, *fired* is False even if condition matches.
        """
        if name not in self._rules:
            return json.dumps({"error": f"rule {name!r} not found"})

        rule = self._rules[name]
        fired = _evaluate_condition(rule.condition, value, rule.threshold, rule.threshold_high)

        # cooldown check
        now = time.time()
        if fired and rule.cooldown_s > 0 and rule.last_fired is not None:
            if (now - rule.last_fired) < rule.cooldown_s:
                fired = False  # in cooldown

        event: Dict[str, Any] = {
            "rule": name,
            "fired": fired,
            "value": value,
            "severity": rule.severity if fired else None,
            "message": rule.message,
            "metric": rule.metric,
            "threshold": rule.threshold,
            "threshold_high": rule.threshold_high,
            "condition": rule.condition,
            "timestamp": now,
        }
        if metadata:
            event["metadata"] = metadata

        if fired:
            rule.fire_count += 1
            rule.last_fired = now
            self._history.append(event)
            if len(self._history) > _MAX_HISTORY:
                self._history = self._history[-_MAX_HISTORY:]

        return json.dumps(event)

    def alert_list(self) -> str:
        """List all defined alert rules.

        Returns ``{rules: [{name, condition, severity, fire_count, last_fired}], count}``.
        """
        items = [
            {
                "name": r.name,
                "condition": r.condition,
                "severity": r.severity,
                "metric": r.metric,
                "fire_count": r.fire_count,
                "last_fired": r.last_fired,
            }
            for r in self._rules.values()
        ]
        return json.dumps({"rules": items, "count": len(items)})

    def alert_get(self, name: str) -> str:
        """Get full definition of a rule.

        Returns full rule dict including threshold, cooldown, fire_count.
        """
        if name not in self._rules:
            return json.dumps({"error": f"rule {name!r} not found"})
        return json.dumps(self._rules[name].to_dict())

    def alert_delete(self, name: str) -> str:
        """Delete an alert rule by name.

        Returns ``{deleted, name}``.
        """
        if name not in self._rules:
            return json.dumps({"error": f"rule {name!r} not found"})
        del self._rules[name]
        return json.dumps({"deleted": True, "name": name})

    def alert_history(
        self,
        rule: Optional[str] = None,
        limit: int = 50,
        severity: Optional[str] = None,
    ) -> str:
        """Retrieve fired alert history.

        *rule*: filter to a specific rule name.
        *severity*: filter to a severity level.
        *limit*: max events to return (default 50).

        Returns ``{events: [...], count, total}``.
        """
        events = self._history
        if rule:
            events = [e for e in events if e.get("rule") == rule]
        if severity:
            events = [e for e in events if e.get("severity") == severity]
        total = len(events)
        events = events[-limit:] if limit > 0 else events
        return json.dumps({"events": list(reversed(events)), "count": len(events), "total": total})

    def alert_clear(self, rule: Optional[str] = None) -> str:
        """Clear alert history for *rule* (or all if omitted).

        Returns ``{cleared, rule}``.
        """
        if rule:
            before = len(self._history)
            self._history = [e for e in self._history if e.get("rule") != rule]
            cleared = before - len(self._history)
        else:
            cleared = len(self._history)
            self._history = []
        return json.dumps({"cleared": cleared, "rule": rule})

    def alert_stats(self) -> str:
        """Return aggregate stats per rule.

        Returns ``{stats: [{name, fire_count, last_fired, severity}], total_fires}``.
        """
        stats = [
            {
                "name": r.name,
                "fire_count": r.fire_count,
                "last_fired": r.last_fired,
                "severity": r.severity,
                "condition": r.condition,
            }
            for r in self._rules.values()
        ]
        total = sum(r.fire_count for r in self._rules.values())
        return json.dumps({"stats": stats, "total_fires": total, "rules": len(self._rules)})

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "alert"

    @property
    def description(self) -> str:
        return (
            "Threshold-based alerting. alert_define, alert_evaluate, "
            "alert_list/get/delete, alert_history, alert_clear, alert_stats. "
            "Conditions: gt/gte/lt/lte/eq/ne/between/outside/contains/"
            "not_contains/is_empty/is_truthy. Severity: info/warning/error/"
            "critical. Cooldown support. Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "alert_define",
                "description": "Define a named alert rule. Conditions: gt/gte/lt/lte/eq/ne/between/outside/contains/not_contains/is_empty/is_truthy. Severity: info/warning/error/critical.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "condition": {"type": "string"},
                        "threshold": {},
                        "threshold_high": {},
                        "severity": {"type": "string"},
                        "message": {"type": "string"},
                        "metric": {"type": "string"},
                        "cooldown_s": {"type": "number"},
                    },
                    "required": ["name", "condition"],
                },
            },
            {
                "name": "alert_evaluate",
                "description": "Evaluate a value against a named rule. Returns {fired, severity, value, timestamp}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {},
                        "metadata": {"type": "object"},
                    },
                    "required": ["name", "value"],
                },
            },
            {
                "name": "alert_list",
                "description": "List all alert rules. Returns {rules, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "alert_get",
                "description": "Get full definition of an alert rule by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "alert_delete",
                "description": "Delete an alert rule by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "alert_history",
                "description": "Retrieve fired alert history. Filter by rule name, severity, limit. Returns {events, count, total}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "rule": {"type": "string"},
                        "limit": {"type": "integer"},
                        "severity": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "alert_clear",
                "description": "Clear alert history for a rule (or all if rule omitted). Returns {cleared, rule}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"rule": {"type": "string"}},
                    "required": [],
                },
            },
            {
                "name": "alert_stats",
                "description": "Aggregate stats per rule: fire_count, last_fired. Returns {stats, total_fires}.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "alert_define":
            return self.alert_define(**arguments)
        if tool_name == "alert_evaluate":
            return self.alert_evaluate(**arguments)
        if tool_name == "alert_list":
            return self.alert_list(**arguments)
        if tool_name == "alert_get":
            return self.alert_get(**arguments)
        if tool_name == "alert_delete":
            return self.alert_delete(**arguments)
        if tool_name == "alert_history":
            return self.alert_history(**arguments)
        if tool_name == "alert_clear":
            return self.alert_clear(**arguments)
        if tool_name == "alert_stats":
            return self.alert_stats(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
