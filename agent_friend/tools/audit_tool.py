"""audit_tool.py — AuditTool for agent-friend (stdlib only).

Structured event audit log for agent observability and tracing.
Record events with type, actor, resource, metadata; search and filter
the log; enforce retention limits; export to JSON lines.

Features:
* audit_log      — record a structured audit event
* audit_search   — filter events by type/actor/resource/time/text
* audit_get      — retrieve a single event by ID
* audit_stats    — aggregate counts by type, actor, and resource
* audit_export   — export events as JSON lines string
* audit_clear    — clear all events (with optional before-timestamp filter)
* audit_types    — list distinct event types in the log
* audit_timeline — events bucketed by hour/day for trend analysis

Usage::

    tool = AuditTool()

    tool.audit_log("user.login", actor="alice", resource="auth",
                   metadata={"ip": "1.2.3.4"})
    # {"id": "...", "type": "user.login", "actor": "alice", ...}

    tool.audit_search(actor="alice", limit=10)
    # {"events": [...], "count": 5, "total": 5}
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseTool


_MAX_EVENTS = 10_000
_VALID_RETENTION_UNITS = ("hours", "days", "weeks")


class _Event:
    def __init__(
        self,
        event_type: str,
        actor: str,
        resource: str,
        metadata: Dict,
        severity: str,
        outcome: str,
    ):
        self.id = str(uuid.uuid4())
        self.type = event_type
        self.actor = actor
        self.resource = resource
        self.metadata = metadata
        self.severity = severity
        self.outcome = outcome
        self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "actor": self.actor,
            "resource": self.resource,
            "metadata": self.metadata,
            "severity": self.severity,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
        }

    def matches(self, **filters) -> bool:
        """Return True if this event matches all provided filters."""
        if "type" in filters and filters["type"] and self.type != filters["type"]:
            return False
        if "actor" in filters and filters["actor"] and self.actor != filters["actor"]:
            return False
        if "resource" in filters and filters["resource"] and self.resource != filters["resource"]:
            return False
        if "severity" in filters and filters["severity"] and self.severity != filters["severity"]:
            return False
        if "outcome" in filters and filters["outcome"] and self.outcome != filters["outcome"]:
            return False
        if "after" in filters and filters["after"] and self.timestamp < filters["after"]:
            return False
        if "before" in filters and filters["before"] and self.timestamp > filters["before"]:
            return False
        if "text" in filters and filters["text"]:
            needle = filters["text"].lower()
            haystack = (
                self.type + " " + self.actor + " " + self.resource +
                " " + json.dumps(self.metadata)
            ).lower()
            if needle not in haystack:
                return False
        return True


class AuditTool(BaseTool):
    """Structured audit log for agent observability.

    Record events with type, actor, resource, metadata, severity, and
    outcome. Search, filter, aggregate, and export the log.
    """

    def __init__(self, max_events: int = _MAX_EVENTS):
        self._events: List[_Event] = []
        self._max_events = max_events
        self._total_logged = 0

    # ── public API ────────────────────────────────────────────────────

    def audit_log(
        self,
        event_type: str,
        actor: str = "",
        resource: str = "",
        metadata: Optional[Dict] = None,
        severity: str = "info",
        outcome: str = "success",
    ) -> str:
        """Record a structured audit event.

        *event_type*: dot-notation string, e.g. "user.login", "file.delete".
        *actor*: who performed the action (user, agent, service name).
        *resource*: what was acted upon (file path, DB table, API endpoint).
        *metadata*: arbitrary dict of additional context.
        *severity*: info | warning | error | critical.
        *outcome*: success | failure | denied | unknown.

        Returns ``{id, type, actor, resource, timestamp}``.
        """
        if not event_type or not event_type.strip():
            return json.dumps({"error": "event_type must be non-empty"})
        valid_severities = {"info", "warning", "error", "critical"}
        if severity not in valid_severities:
            return json.dumps({"error": f"severity must be one of {sorted(valid_severities)}"})
        valid_outcomes = {"success", "failure", "denied", "unknown"}
        if outcome not in valid_outcomes:
            return json.dumps({"error": f"outcome must be one of {sorted(valid_outcomes)}"})

        event = _Event(
            event_type=event_type.strip(),
            actor=actor or "",
            resource=resource or "",
            metadata=dict(metadata) if metadata else {},
            severity=severity,
            outcome=outcome,
        )
        self._events.append(event)
        self._total_logged += 1

        # enforce retention cap (drop oldest)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        return json.dumps({
            "id": event.id,
            "type": event.type,
            "actor": event.actor,
            "resource": event.resource,
            "timestamp": event.timestamp,
        })

    def audit_search(
        self,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        severity: Optional[str] = None,
        outcome: Optional[str] = None,
        after: Optional[float] = None,
        before: Optional[float] = None,
        text: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> str:
        """Search and filter the audit log.

        All filters are ANDed. *text* performs a substring search across
        type, actor, resource, and JSON-serialized metadata.

        Returns ``{events, count, total, offset}``.
        """
        filters = {
            "type": event_type,
            "actor": actor,
            "resource": resource,
            "severity": severity,
            "outcome": outcome,
            "after": after,
            "before": before,
            "text": text,
        }
        matched = [e for e in self._events if e.matches(**filters)]
        total = len(matched)
        paged = matched[offset: offset + limit] if limit > 0 else matched[offset:]
        return json.dumps({
            "events": [e.to_dict() for e in reversed(paged)],  # newest first
            "count": len(paged),
            "total": total,
            "offset": offset,
        })

    def audit_get(self, event_id: str) -> str:
        """Retrieve a single event by ID.

        Returns full event dict or ``{error}``.
        """
        for event in self._events:
            if event.id == event_id:
                return json.dumps(event.to_dict())
        return json.dumps({"error": f"event {event_id!r} not found"})

    def audit_stats(
        self,
        after: Optional[float] = None,
        before: Optional[float] = None,
    ) -> str:
        """Aggregate stats: counts by type, actor, resource, severity, outcome.

        Returns ``{total, by_type, by_actor, by_resource, by_severity, by_outcome}``.
        """
        events = self._events
        if after:
            events = [e for e in events if e.timestamp >= after]
        if before:
            events = [e for e in events if e.timestamp <= before]

        by_type: Dict[str, int] = {}
        by_actor: Dict[str, int] = {}
        by_resource: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_outcome: Dict[str, int] = {}

        for e in events:
            by_type[e.type] = by_type.get(e.type, 0) + 1
            if e.actor:
                by_actor[e.actor] = by_actor.get(e.actor, 0) + 1
            if e.resource:
                by_resource[e.resource] = by_resource.get(e.resource, 0) + 1
            by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
            by_outcome[e.outcome] = by_outcome.get(e.outcome, 0) + 1

        return json.dumps({
            "total": len(events),
            "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
            "by_actor": dict(sorted(by_actor.items(), key=lambda x: -x[1])),
            "by_resource": dict(sorted(by_resource.items(), key=lambda x: -x[1])),
            "by_severity": by_severity,
            "by_outcome": by_outcome,
        })

    def audit_export(
        self,
        after: Optional[float] = None,
        before: Optional[float] = None,
        event_type: Optional[str] = None,
    ) -> str:
        """Export matching events as a JSON lines string.

        Returns ``{lines, count}`` where *lines* is newline-separated JSON.
        """
        filters = {"type": event_type, "after": after, "before": before}
        matched = [e for e in self._events if e.matches(**filters)]
        lines = "\n".join(json.dumps(e.to_dict()) for e in matched)
        return json.dumps({"lines": lines, "count": len(matched)})

    def audit_clear(
        self,
        before: Optional[float] = None,
    ) -> str:
        """Clear events from the log.

        *before*: if provided, only clear events older than this timestamp.
        Otherwise clears all events.

        Returns ``{cleared, remaining}``.
        """
        if before is not None:
            original = len(self._events)
            self._events = [e for e in self._events if e.timestamp >= before]
            cleared = original - len(self._events)
        else:
            cleared = len(self._events)
            self._events = []
        return json.dumps({"cleared": cleared, "remaining": len(self._events)})

    def audit_types(self) -> str:
        """List distinct event types currently in the log.

        Returns ``{types: [...], count}``.
        """
        types = sorted({e.type for e in self._events})
        return json.dumps({"types": types, "count": len(types)})

    def audit_timeline(
        self,
        bucket: str = "hour",
        after: Optional[float] = None,
        before: Optional[float] = None,
    ) -> str:
        """Bucket events by time interval for trend analysis.

        *bucket*: "hour" (3600s) or "day" (86400s).

        Returns ``{buckets: [{bucket, count, timestamp_start}], total}``.
        """
        if bucket not in ("hour", "day"):
            return json.dumps({"error": "bucket must be 'hour' or 'day'"})

        step = 3600 if bucket == "hour" else 86400
        events = self._events
        if after:
            events = [e for e in events if e.timestamp >= after]
        if before:
            events = [e for e in events if e.timestamp <= before]

        counts: Dict[int, int] = {}
        for e in events:
            bucket_ts = int(e.timestamp // step) * step
            counts[bucket_ts] = counts.get(bucket_ts, 0) + 1

        buckets = [
            {"bucket": k, "count": v, "timestamp_start": k}
            for k, v in sorted(counts.items())
        ]
        return json.dumps({"buckets": buckets, "total": len(events), "bucket_size": bucket})

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "audit"

    @property
    def description(self) -> str:
        return (
            "Structured audit log for agent observability. audit_log (event_type, "
            "actor, resource, metadata, severity, outcome), audit_search (filter by "
            "type/actor/resource/time/text), audit_get, audit_stats, audit_export "
            "(JSON lines), audit_clear, audit_types, audit_timeline. Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "audit_log",
                "description": "Record a structured audit event. Returns {id, type, actor, resource, timestamp}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_type": {"type": "string"},
                        "actor": {"type": "string"},
                        "resource": {"type": "string"},
                        "metadata": {"type": "object"},
                        "severity": {"type": "string"},
                        "outcome": {"type": "string"},
                    },
                    "required": ["event_type"],
                },
            },
            {
                "name": "audit_search",
                "description": "Filter the audit log. Filters: event_type, actor, resource, severity, outcome, after, before, text. Returns {events, count, total}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "event_type": {"type": "string"},
                        "actor": {"type": "string"},
                        "resource": {"type": "string"},
                        "severity": {"type": "string"},
                        "outcome": {"type": "string"},
                        "after": {"type": "number"},
                        "before": {"type": "number"},
                        "text": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": [],
                },
            },
            {
                "name": "audit_get",
                "description": "Retrieve a single event by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {"event_id": {"type": "string"}},
                    "required": ["event_id"],
                },
            },
            {
                "name": "audit_stats",
                "description": "Aggregate stats: by_type, by_actor, by_resource, by_severity, by_outcome.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "after": {"type": "number"},
                        "before": {"type": "number"},
                    },
                    "required": [],
                },
            },
            {
                "name": "audit_export",
                "description": "Export events as JSON lines. Filter by event_type, after, before. Returns {lines, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "after": {"type": "number"},
                        "before": {"type": "number"},
                        "event_type": {"type": "string"},
                    },
                    "required": [],
                },
            },
            {
                "name": "audit_clear",
                "description": "Clear events. before=timestamp to clear only older events. Returns {cleared, remaining}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"before": {"type": "number"}},
                    "required": [],
                },
            },
            {
                "name": "audit_types",
                "description": "List distinct event types in the log. Returns {types, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "audit_timeline",
                "description": "Bucket events by hour or day for trend analysis. Returns {buckets, total}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "bucket": {"type": "string", "enum": ["hour", "day"]},
                        "after": {"type": "number"},
                        "before": {"type": "number"},
                    },
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "audit_log":
            return self.audit_log(**arguments)
        if tool_name == "audit_search":
            return self.audit_search(**arguments)
        if tool_name == "audit_get":
            return self.audit_get(**arguments)
        if tool_name == "audit_stats":
            return self.audit_stats(**arguments)
        if tool_name == "audit_export":
            return self.audit_export(**arguments)
        if tool_name == "audit_clear":
            return self.audit_clear(**arguments)
        if tool_name == "audit_types":
            return self.audit_types(**arguments)
        if tool_name == "audit_timeline":
            return self.audit_timeline(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
