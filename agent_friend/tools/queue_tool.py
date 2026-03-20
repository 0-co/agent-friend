"""queue_tool.py — QueueTool for agent-friend (stdlib only).

Work queues let agents process tasks in order, prioritize critical work,
and implement producer-consumer patterns — all without external message
brokers.

Three queue types are provided:

* **FIFO** — first-in, first-out (standard queue).
* **LIFO** — last-in, first-out (stack).
* **Priority** — items with lower priority numbers are dequeued first
  (like a min-heap).  Useful for scheduling urgent tasks ahead of routine ones.

Usage::

    tool = QueueTool()

    # FIFO work queue
    tool.queue_create("tasks")
    tool.queue_push("tasks", {"url": "https://example.com", "action": "scrape"})
    tool.queue_push("tasks", {"url": "https://other.com", "action": "scrape"})

    item = json.loads(tool.queue_pop("tasks"))   # {"url": "https://example.com", ...}
    size = json.loads(tool.queue_size("tasks"))  # {"size": 1}

    # Priority queue — process high-priority tasks first
    tool.queue_create("alerts", kind="priority")
    tool.queue_push("alerts", "disk full", priority=1)        # high priority
    tool.queue_push("alerts", "CPU usage high", priority=5)   # low priority
    item = json.loads(tool.queue_pop("alerts"))               # "disk full"
"""

import heapq
import json
from collections import deque
from typing import Any, Dict, List, Optional

from .base import BaseTool


# ---------------------------------------------------------------------------
# Queue implementations
# ---------------------------------------------------------------------------

class _FIFOQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self._q: deque = deque()

    def push(self, item: Any) -> bool:
        if self.maxsize and len(self._q) >= self.maxsize:
            return False
        self._q.append(item)
        return True

    def pop(self) -> Optional[Any]:
        if not self._q:
            return None
        return self._q.popleft()

    def peek(self) -> Optional[Any]:
        if not self._q:
            return None
        return self._q[0]

    def size(self) -> int:
        return len(self._q)

    def is_empty(self) -> bool:
        return len(self._q) == 0

    def clear(self) -> None:
        self._q.clear()

    def kind(self) -> str:
        return "fifo"


class _LIFOQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self._q: List[Any] = []

    def push(self, item: Any) -> bool:
        if self.maxsize and len(self._q) >= self.maxsize:
            return False
        self._q.append(item)
        return True

    def pop(self) -> Optional[Any]:
        if not self._q:
            return None
        return self._q.pop()

    def peek(self) -> Optional[Any]:
        if not self._q:
            return None
        return self._q[-1]

    def size(self) -> int:
        return len(self._q)

    def is_empty(self) -> bool:
        return len(self._q) == 0

    def clear(self) -> None:
        self._q.clear()

    def kind(self) -> str:
        return "lifo"


class _PriorityQueue:
    """Min-heap — lower priority number = higher urgency (dequeued first)."""

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = maxsize
        self._heap: List = []  # (priority, sequence, item)
        self._seq = 0

    def push(self, item: Any, priority: float = 0.0) -> bool:
        if self.maxsize and len(self._heap) >= self.maxsize:
            return False
        heapq.heappush(self._heap, (priority, self._seq, item))
        self._seq += 1
        return True

    def pop(self) -> Optional[Any]:
        if not self._heap:
            return None
        _pri, _seq, item = heapq.heappop(self._heap)
        return item

    def peek(self) -> Optional[Any]:
        if not self._heap:
            return None
        return self._heap[0][2]

    def size(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def clear(self) -> None:
        self._heap.clear()
        self._seq = 0

    def kind(self) -> str:
        return "priority"


# ---------------------------------------------------------------------------
# QueueTool
# ---------------------------------------------------------------------------

class QueueTool(BaseTool):
    """Named work queues: FIFO, LIFO (stack), and priority queue.

    Create named queues and use them to coordinate agent work — batch
    processing, producer-consumer patterns, and priority scheduling.

    Parameters
    ----------
    max_queues:
        Maximum number of named queues (default 50).
    default_maxsize:
        Default maximum items per queue (0 = unlimited, default 0).
    """

    def __init__(self, max_queues: int = 50, default_maxsize: int = 0) -> None:
        self.max_queues = max_queues
        self.default_maxsize = default_maxsize
        self._queues: Dict[str, Any] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _get(self, name: str) -> Any:
        q = self._queues.get(name)
        if q is None:
            raise KeyError(f"No queue named '{name}'")
        return q

    # ── public API ────────────────────────────────────────────────────────

    def queue_create(
        self,
        name: str,
        kind: str = "fifo",
        maxsize: int = 0,
    ) -> str:
        """Create a named queue.

        Parameters
        ----------
        name:
            Unique name for this queue.
        kind:
            ``"fifo"`` (default), ``"lifo"`` (stack), or ``"priority"``.
        maxsize:
            Maximum number of items (0 = unlimited).

        Returns ``{"created": true, "name": "...", "kind": "..."}``
        """
        if name in self._queues:
            return json.dumps({"error": f"Queue '{name}' already exists."})
        if len(self._queues) >= self.max_queues:
            return json.dumps({"error": f"Max queues ({self.max_queues}) reached."})

        ms = maxsize if maxsize >= 0 else self.default_maxsize
        k = kind.lower()
        if k == "fifo":
            self._queues[name] = _FIFOQueue(maxsize=ms)
        elif k == "lifo":
            self._queues[name] = _LIFOQueue(maxsize=ms)
        elif k in ("priority", "pq"):
            self._queues[name] = _PriorityQueue(maxsize=ms)
        else:
            return json.dumps({"error": f"Unknown kind '{kind}'. Use: fifo, lifo, priority"})

        return json.dumps({"created": True, "name": name, "kind": k})

    def queue_push(
        self,
        name: str,
        item: Any,
        priority: float = 0.0,
    ) -> str:
        """Add an item to the queue.

        For priority queues, lower ``priority`` numbers are dequeued first.
        The ``priority`` parameter is ignored for FIFO/LIFO queues.

        Returns ``{"pushed": true/false}`` — false if the queue is full.
        """
        try:
            q = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if isinstance(q, _PriorityQueue):
            ok = q.push(item, priority)
        else:
            ok = q.push(item)
        return json.dumps({"pushed": ok, "size": q.size()})

    def queue_pop(self, name: str) -> str:
        """Remove and return the next item from the queue.

        Returns ``{"item": ..., "size": N}`` or ``{"item": null, "empty": true}``
        if the queue is empty.
        """
        try:
            q = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        item = q.pop()
        if item is None and q.is_empty():
            return json.dumps({"item": None, "empty": True, "size": 0})
        return json.dumps({"item": item, "size": q.size()})

    def queue_peek(self, name: str) -> str:
        """Return the next item **without** removing it.

        Returns ``{"item": ..., "size": N}`` or ``{"item": null, "empty": true}``.
        """
        try:
            q = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        item = q.peek()
        if item is None and q.is_empty():
            return json.dumps({"item": None, "empty": True, "size": 0})
        return json.dumps({"item": item, "size": q.size()})

    def queue_size(self, name: str) -> str:
        """Return the current number of items in the queue."""
        try:
            q = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"size": q.size(), "empty": q.is_empty()})

    def queue_clear(self, name: str) -> str:
        """Remove all items from the queue."""
        try:
            q = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        q.clear()
        return json.dumps({"cleared": True, "name": name})

    def queue_delete(self, name: str) -> str:
        """Delete a queue entirely."""
        if name not in self._queues:
            return json.dumps({"error": f"No queue named '{name}'"})
        del self._queues[name]
        return json.dumps({"deleted": True, "name": name})

    def queue_list(self) -> str:
        """List all queues with their sizes and types."""
        result = []
        for name, q in self._queues.items():
            result.append({
                "name": name,
                "kind": q.kind(),
                "size": q.size(),
                "empty": q.is_empty(),
            })
        return json.dumps(result)

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "queue"

    @property
    def description(self) -> str:
        return (
            "Named work queues for agent task coordination: FIFO, LIFO (stack), "
            "and priority queue. Create named queues and push/pop items to "
            "implement batch processing, producer-consumer patterns, and "
            "priority task scheduling. All in-memory, stdlib only."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "queue_create",
                "description": (
                    "Create a named queue. kind: 'fifo' (default), 'lifo' (stack), "
                    "'priority' (min-heap — lower priority number = higher urgency). "
                    "maxsize=0 means unlimited."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique queue name"},
                        "kind": {
                            "type": "string",
                            "enum": ["fifo", "lifo", "priority"],
                            "description": "Queue type (default: fifo)",
                        },
                        "maxsize": {"type": "integer", "description": "Max items (0=unlimited)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "queue_push",
                "description": (
                    "Add an item to the queue. item can be any JSON value. "
                    "priority (default 0) is used for priority queues — lower = more urgent."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "item": {"description": "The item to enqueue (any JSON value)"},
                        "priority": {"type": "number", "description": "Priority (priority queues only, lower = more urgent)"},
                    },
                    "required": ["name", "item"],
                },
            },
            {
                "name": "queue_pop",
                "description": "Remove and return the next item. Returns {item, size} or {item: null, empty: true}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "queue_peek",
                "description": "Return the next item WITHOUT removing it.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "queue_size",
                "description": "Return the number of items currently in the queue.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "queue_clear",
                "description": "Remove all items from a queue.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "queue_delete",
                "description": "Delete a queue entirely.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "queue_list",
                "description": "List all active queues with their sizes and types.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "queue_create":
            return self.queue_create(**arguments)
        if tool_name == "queue_push":
            return self.queue_push(**arguments)
        if tool_name == "queue_pop":
            return self.queue_pop(**arguments)
        if tool_name == "queue_peek":
            return self.queue_peek(**arguments)
        if tool_name == "queue_size":
            return self.queue_size(**arguments)
        if tool_name == "queue_clear":
            return self.queue_clear(**arguments)
        if tool_name == "queue_delete":
            return self.queue_delete(**arguments)
        if tool_name == "queue_list":
            return self.queue_list()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
