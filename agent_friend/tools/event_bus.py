"""event_bus.py — EventBusTool for agent-friend (stdlib only).

A pub/sub event bus lets agent components communicate without direct coupling.
One agent publishes events; others subscribe by topic.

Unlike a real message broker, this is an *in-process* event bus — ideal for
coordinating tools within a single agent or for testing multi-agent patterns
without external infrastructure.

Features:
- Multiple topics, each with an independent subscriber list
- Event history per topic (configurable depth)
- Wildcard topic matching (``"*"`` subscribes to every topic)
- Subscriber call counts tracked for observability

Usage::

    tool = EventBusTool()

    tool.bus_subscribe("new_url", "scraper")
    tool.bus_subscribe("new_url", "logger")

    tool.bus_publish("new_url", {"url": "https://example.com"})
    # → scraper and logger are both notified

    tool.bus_history("new_url", n=5)
    # [{"topic": "new_url", "data": {...}, "event_id": 1, "timestamp": ...}]
"""

import json
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from .base import BaseTool


_WILDCARD = "*"


class EventBusTool(BaseTool):
    """In-process pub/sub event bus for agent coordination.

    Agents publish events to named topics; subscribers are notified in order
    of subscription.  A ``"*"`` subscriber receives events from all topics.

    Parameters
    ----------
    max_history:
        Number of recent events to keep per topic (default 100).
    max_topics:
        Maximum number of distinct topics (default 200).
    max_subscribers_per_topic:
        Maximum subscribers per topic (default 50).
    """

    def __init__(
        self,
        max_history: int = 100,
        max_topics: int = 200,
        max_subscribers_per_topic: int = 50,
    ) -> None:
        self.max_history = max_history
        self.max_topics = max_topics
        self.max_subscribers_per_topic = max_subscribers_per_topic

        # topic → ordered list of subscriber names
        self._subscribers: Dict[str, List[str]] = defaultdict(list)
        # topic → deque of event dicts
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_history))
        # subscriber_name → call_count
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._event_seq = 0

    # ── public API ────────────────────────────────────────────────────────

    def bus_subscribe(self, topic: str, subscriber: str) -> str:
        """Subscribe *subscriber* to *topic*.

        Use ``topic="*"`` to receive all events regardless of topic.

        Returns ``{"subscribed": true, "topic": "...", "subscriber": "..."}``.
        """
        if topic not in self._subscribers and len(self._subscribers) >= self.max_topics:
            return json.dumps({"error": f"Max topics ({self.max_topics}) reached."})
        subs = self._subscribers[topic]
        if subscriber in subs:
            return json.dumps({"subscribed": False, "reason": "already subscribed"})
        if len(subs) >= self.max_subscribers_per_topic:
            return json.dumps({"error": f"Max subscribers per topic ({self.max_subscribers_per_topic}) reached."})
        subs.append(subscriber)
        return json.dumps({"subscribed": True, "topic": topic, "subscriber": subscriber})

    def bus_unsubscribe(self, topic: str, subscriber: str) -> str:
        """Remove *subscriber* from *topic*.

        Returns ``{"unsubscribed": true/false}``.
        """
        subs = self._subscribers.get(topic)
        if subs is None or subscriber not in subs:
            return json.dumps({"unsubscribed": False, "reason": "not subscribed"})
        subs.remove(subscriber)
        return json.dumps({"unsubscribed": True, "topic": topic, "subscriber": subscriber})

    def bus_publish(self, topic: str, data: Any = None) -> str:
        """Publish an event to *topic*.

        All subscribers to *topic* **and** all wildcard subscribers (``"*"``)
        are notified and their call counts incremented.

        Returns ``{"published": true, "event_id": N, "notified": [subscribers]}``.
        """
        self._event_seq += 1
        event = {
            "event_id": self._event_seq,
            "topic": topic,
            "data": data,
            "timestamp": time.time(),
        }

        # Store in topic history
        self._history[topic].append(event)

        # Determine notified subscribers
        topic_subs = list(self._subscribers.get(topic, []))
        wildcard_subs = [s for s in self._subscribers.get(_WILDCARD, []) if s not in topic_subs]
        all_notified = topic_subs + wildcard_subs

        for sub in all_notified:
            self._call_counts[sub] += 1

        return json.dumps({
            "published": True,
            "event_id": self._event_seq,
            "topic": topic,
            "notified": all_notified,
        })

    def bus_history(self, topic: str, n: int = 10) -> str:
        """Return the *n* most recent events for *topic*.

        Returns a list of ``{event_id, topic, data, timestamp}`` dicts,
        oldest first.
        """
        hist = self._history.get(topic)
        if hist is None:
            return json.dumps([])
        events = list(hist)[-n:]
        return json.dumps(events)

    def bus_topics(self) -> str:
        """List all topics that have at least one subscriber or history entry.

        Returns list of ``{topic, subscribers, event_count}`` dicts.
        """
        all_topics = set(self._subscribers.keys()) | set(self._history.keys())
        result = []
        for topic in sorted(all_topics):
            result.append({
                "topic": topic,
                "subscribers": list(self._subscribers.get(topic, [])),
                "event_count": len(self._history.get(topic, [])),
            })
        return json.dumps(result)

    def bus_subscribers(self, topic: str) -> str:
        """List all subscribers to *topic*."""
        subs = self._subscribers.get(topic, [])
        return json.dumps({"topic": topic, "subscribers": list(subs)})

    def bus_stats(self) -> str:
        """Return call counts per subscriber and total events published."""
        return json.dumps({
            "total_events": self._event_seq,
            "topic_count": len(set(self._subscribers.keys()) | set(self._history.keys())),
            "subscriber_counts": dict(self._call_counts),
        })

    def bus_clear(self, topic: Optional[str] = None) -> str:
        """Clear history (and optionally subscribers) for *topic*, or all topics.

        If *topic* is ``None``, clears everything.
        Returns ``{"cleared": true, "topic": topic or "all"}``.
        """
        if topic is None:
            self._history.clear()
            self._subscribers.clear()
            self._call_counts.clear()
            self._event_seq = 0
            return json.dumps({"cleared": True, "topic": "all"})
        if topic in self._history:
            self._history[topic].clear()
        if topic in self._subscribers:
            del self._subscribers[topic]
        return json.dumps({"cleared": True, "topic": topic})

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "event_bus"

    @property
    def description(self) -> str:
        return (
            "In-process pub/sub event bus for agent coordination. Publish events "
            "to named topics; subscribed agents are notified in order. Supports "
            "wildcard subscriptions ('*'), event history per topic, and subscriber "
            "call-count observability. All in-memory, stdlib only."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "bus_subscribe",
                "description": (
                    "Subscribe to a topic. Use topic='*' to receive all events. "
                    "subscriber is a name string identifying who receives the events."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic name or '*' for wildcard"},
                        "subscriber": {"type": "string", "description": "Subscriber name/ID"},
                    },
                    "required": ["topic", "subscriber"],
                },
            },
            {
                "name": "bus_unsubscribe",
                "description": "Unsubscribe from a topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "subscriber": {"type": "string"},
                    },
                    "required": ["topic", "subscriber"],
                },
            },
            {
                "name": "bus_publish",
                "description": (
                    "Publish an event to a topic. All subscribers (and wildcard subscribers) "
                    "are notified. Returns {published, event_id, notified: [subscriber names]}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "data": {"description": "Event payload (any JSON value)"},
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "bus_history",
                "description": "Return the n most recent events for a topic (oldest first).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "n": {"type": "integer", "description": "Number of events to return (default 10)"},
                    },
                    "required": ["topic"],
                },
            },
            {
                "name": "bus_topics",
                "description": "List all topics with their subscriber counts and event counts.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "bus_subscribers",
                "description": "List all subscribers to a specific topic.",
                "input_schema": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
            {
                "name": "bus_stats",
                "description": "Return total events published and per-subscriber call counts.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "bus_clear",
                "description": "Clear history and subscribers for a topic, or all topics if topic is omitted.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic to clear (omit to clear all)"},
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "bus_subscribe":
            return self.bus_subscribe(**arguments)
        if tool_name == "bus_unsubscribe":
            return self.bus_unsubscribe(**arguments)
        if tool_name == "bus_publish":
            return self.bus_publish(**arguments)
        if tool_name == "bus_history":
            return self.bus_history(**arguments)
        if tool_name == "bus_topics":
            return self.bus_topics()
        if tool_name == "bus_subscribers":
            return self.bus_subscribers(**arguments)
        if tool_name == "bus_stats":
            return self.bus_stats()
        if tool_name == "bus_clear":
            return self.bus_clear(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
