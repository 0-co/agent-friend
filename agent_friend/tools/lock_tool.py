"""lock_tool.py — LockTool for agent-friend (stdlib only).

Named mutex-style locking for agent workflows. Prevents concurrent
operations on shared resources within a single process.

Features:
* lock_acquire   — acquire a named lock (optionally with timeout and auto-expire)
* lock_release   — release a held lock
* lock_try       — non-blocking attempt to acquire a lock
* lock_status    — check whether a lock is held
* lock_list      — list all locks (held + available)
* lock_release_all — release all locks held by an owner
* lock_expire    — manually expire/force-release a lock
* lock_stats     — aggregate lock stats

Usage::

    tool = LockTool()

    tool.lock_acquire("db_write", owner="worker-1", ttl_s=30)
    # {"acquired": True, "lock": "db_write", "owner": "worker-1"}

    tool.lock_status("db_write")
    # {"held": True, "owner": "worker-1", "remaining_s": 28.4}

    tool.lock_release("db_write", owner="worker-1")
    # {"released": True, "lock": "db_write"}
"""

import json
import time
from typing import Any, Dict, List, Optional

from .base import BaseTool


class _Lock:
    def __init__(self, name: str, owner: str, ttl_s: Optional[float]):
        self.name = name
        self.owner = owner
        self.acquired_at = time.monotonic()
        self.ttl_s = ttl_s  # None = never expires
        self.acquire_count = 0

    def is_expired(self) -> bool:
        if self.ttl_s is None:
            return False
        return (time.monotonic() - self.acquired_at) >= self.ttl_s

    def remaining_s(self) -> Optional[float]:
        if self.ttl_s is None:
            return None
        remaining = self.ttl_s - (time.monotonic() - self.acquired_at)
        return max(0.0, round(remaining, 3))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "owner": self.owner,
            "acquired_at": self.acquired_at,
            "ttl_s": self.ttl_s,
            "remaining_s": self.remaining_s(),
            "expired": self.is_expired(),
        }


class LockTool(BaseTool):
    """Named mutex-style locking for agent workflows.

    All locks are in-memory and scoped to a single LockTool instance.
    Expired locks are automatically cleaned up on access.
    """

    MAX_LOCKS = 500
    MAX_WAIT_S = 60.0
    POLL_INTERVAL_S = 0.01

    def __init__(self):
        self._locks: Dict[str, _Lock] = {}
        self._total_acquisitions = 0
        self._total_contentions = 0

    # ── helpers ──────────────────────────────────────────────────────────

    def _cleanup_expired(self) -> None:
        expired = [n for n, lk in self._locks.items() if lk.is_expired()]
        for name in expired:
            del self._locks[name]

    def _is_held(self, name: str) -> bool:
        lk = self._locks.get(name)
        if lk is None:
            return False
        if lk.is_expired():
            del self._locks[name]
            return False
        return True

    # ── public API ───────────────────────────────────────────────────────

    def lock_acquire(
        self,
        name: str,
        owner: str = "default",
        ttl_s: Optional[float] = None,
        wait_s: float = 0.0,
    ) -> str:
        """Acquire a named lock.

        *owner*: identifier of the caller (used for release authorization).
        *ttl_s*: time-to-live in seconds; None = never auto-expires.
        *wait_s*: max seconds to block waiting for the lock (0 = don't wait).

        Returns ``{acquired, lock, owner, ttl_s, waited_ms}``.
        """
        if not name or not name.strip():
            return json.dumps({"error": "lock name must be non-empty"})
        if not owner or not owner.strip():
            return json.dumps({"error": "owner must be non-empty"})
        if ttl_s is not None and ttl_s <= 0:
            return json.dumps({"error": "ttl_s must be positive"})
        if wait_s < 0:
            return json.dumps({"error": "wait_s must be >= 0"})
        wait_s = min(wait_s, self.MAX_WAIT_S)

        t0 = time.monotonic()
        contended = False

        while True:
            self._cleanup_expired()

            if not self._is_held(name):
                if len(self._locks) >= self.MAX_LOCKS:
                    return json.dumps({"error": f"max locks ({self.MAX_LOCKS}) reached"})
                self._locks[name] = _Lock(name, owner, ttl_s)
                self._locks[name].acquire_count += 1
                self._total_acquisitions += 1
                if contended:
                    self._total_contentions += 1
                waited_ms = round((time.monotonic() - t0) * 1000, 3)
                return json.dumps({
                    "acquired": True,
                    "lock": name,
                    "owner": owner,
                    "ttl_s": ttl_s,
                    "waited_ms": waited_ms,
                })

            # Lock is held by someone else
            elapsed = time.monotonic() - t0
            if elapsed >= wait_s:
                held_by = self._locks[name].owner
                waited_ms = round(elapsed * 1000, 3)
                return json.dumps({
                    "acquired": False,
                    "lock": name,
                    "owner": owner,
                    "held_by": held_by,
                    "waited_ms": waited_ms,
                })

            contended = True
            time.sleep(self.POLL_INTERVAL_S)

    def lock_release(self, name: str, owner: str = "default") -> str:
        """Release a named lock.

        Only the owner that acquired the lock may release it (unless force=True
        is not implemented — use lock_expire for force release).

        Returns ``{released, lock, owner}``.
        """
        if not self._is_held(name):
            return json.dumps({"error": f"lock {name!r} is not held"})

        lk = self._locks[name]
        if lk.owner != owner:
            return json.dumps({
                "error": f"lock {name!r} is held by {lk.owner!r}, not {owner!r}"
            })

        del self._locks[name]
        return json.dumps({"released": True, "lock": name, "owner": owner})

    def lock_try(
        self,
        name: str,
        owner: str = "default",
        ttl_s: Optional[float] = None,
    ) -> str:
        """Non-blocking attempt to acquire a lock.

        Returns immediately with ``{acquired, lock, held_by?}``.
        """
        return self.lock_acquire(name, owner=owner, ttl_s=ttl_s, wait_s=0.0)

    def lock_status(self, name: str) -> str:
        """Check the current status of a lock.

        Returns ``{held, lock, owner?, remaining_s?, ttl_s?}``.
        """
        if not self._is_held(name):
            return json.dumps({"held": False, "lock": name})
        lk = self._locks[name]
        return json.dumps({
            "held": True,
            "lock": name,
            "owner": lk.owner,
            "ttl_s": lk.ttl_s,
            "remaining_s": lk.remaining_s(),
        })

    def lock_list(self) -> str:
        """List all currently held locks (after cleaning up expired).

        Returns ``{locks: [{name, owner, ttl_s, remaining_s}], count}``.
        """
        self._cleanup_expired()
        items = [lk.to_dict() for lk in self._locks.values()]
        return json.dumps({"locks": items, "count": len(items)})

    def lock_release_all(self, owner: str) -> str:
        """Release all locks held by *owner*.

        Returns ``{released_count, locks: [names], owner}``.
        """
        if not owner or not owner.strip():
            return json.dumps({"error": "owner must be non-empty"})
        self._cleanup_expired()
        to_release = [n for n, lk in self._locks.items() if lk.owner == owner]
        for name in to_release:
            del self._locks[name]
        return json.dumps({
            "released_count": len(to_release),
            "locks": to_release,
            "owner": owner,
        })

    def lock_expire(self, name: str) -> str:
        """Force-release a lock regardless of owner (admin operation).

        Returns ``{expired, lock, was_owner}``.
        """
        self._cleanup_expired()
        if name not in self._locks:
            return json.dumps({"error": f"lock {name!r} is not held"})
        was_owner = self._locks[name].owner
        del self._locks[name]
        return json.dumps({"expired": True, "lock": name, "was_owner": was_owner})

    def lock_stats(self) -> str:
        """Return aggregate lock statistics.

        Returns ``{total_acquisitions, total_contentions, currently_held, locks}``.
        """
        self._cleanup_expired()
        return json.dumps({
            "total_acquisitions": self._total_acquisitions,
            "total_contentions": self._total_contentions,
            "currently_held": len(self._locks),
            "locks": len(self._locks),
        })

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "lock"

    @property
    def description(self) -> str:
        return (
            "Named mutex-style locking for agent workflows. lock_acquire "
            "(with optional wait_s + ttl_s auto-expire), lock_release, lock_try "
            "(non-blocking), lock_status, lock_list, lock_release_all (by owner), "
            "lock_expire (force), lock_stats. Zero deps."
        )

    def definitions(self) -> List[Dict]:
        return [
            {
                "name": "lock_acquire",
                "description": "Acquire a named lock. owner identifies caller. ttl_s=auto-expire. wait_s=blocking wait. Returns {acquired, lock, owner, waited_ms}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "owner": {"type": "string"},
                        "ttl_s": {"type": "number"},
                        "wait_s": {"type": "number"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "lock_release",
                "description": "Release a lock held by owner. Only owner may release. Returns {released, lock}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "owner": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "lock_try",
                "description": "Non-blocking lock attempt. Returns immediately with {acquired, held_by?}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "owner": {"type": "string"},
                        "ttl_s": {"type": "number"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "lock_status",
                "description": "Check if lock is held. Returns {held, owner?, remaining_s?}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "lock_list",
                "description": "List all currently held locks. Returns {locks, count}.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "lock_release_all",
                "description": "Release all locks held by owner. Returns {released_count, locks}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"owner": {"type": "string"}},
                    "required": ["owner"],
                },
            },
            {
                "name": "lock_expire",
                "description": "Force-release a lock regardless of owner. Returns {expired, was_owner}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "lock_stats",
                "description": "Aggregate lock stats: total_acquisitions, total_contentions, currently_held.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "lock_acquire":
            return self.lock_acquire(**arguments)
        if tool_name == "lock_release":
            return self.lock_release(**arguments)
        if tool_name == "lock_try":
            return self.lock_try(**arguments)
        if tool_name == "lock_status":
            return self.lock_status(**arguments)
        if tool_name == "lock_list":
            return self.lock_list(**arguments)
        if tool_name == "lock_release_all":
            return self.lock_release_all(**arguments)
        if tool_name == "lock_expire":
            return self.lock_expire(**arguments)
        if tool_name == "lock_stats":
            return self.lock_stats(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
