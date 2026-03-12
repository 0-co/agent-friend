"""cache.py — CacheTool for agent-friend (stdlib only).

Agents can store and retrieve values with optional TTL (time-to-live)
expiry.  Entries are persisted to a JSON file so the cache survives
process restarts.

Usage::

    tool = CacheTool()
    tool.cache_set("weather_nyc", '{"temp": 72, "sky": "clear"}', ttl_seconds=3600)
    result = tool.cache_get("weather_nyc")   # returns the value within 1 hour
    tool.cache_stats()                        # {"entries": 1, "hits": 1, "misses": 0}
    tool.cache_delete("weather_nyc")
    tool.cache_clear()
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


_DEFAULT_CACHE_PATH = "~/.agent_friend/cache.json"


class CacheTool(BaseTool):
    """Store and retrieve values with optional TTL expiry.

    Useful for avoiding redundant API calls, memoizing expensive
    computations, or sharing state across agent runs.  The cache is
    backed by a local JSON file — no database required.

    Parameters
    ----------
    cache_path:
        Path to the JSON file used for persistent storage.
        Defaults to ``~/.agent_friend/cache.json``.
    """

    def __init__(self, cache_path: str = _DEFAULT_CACHE_PATH) -> None:
        self.cache_path = str(Path(cache_path).expanduser())
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._hits = 0
        self._misses = 0

    # ── internal helpers ──────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        """Load the cache from disk, returning an empty dict if missing."""
        if not os.path.exists(self.cache_path):
            return {}
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        expires_at = entry.get("expires_at")
        if expires_at is None:
            return False
        return time.time() > expires_at

    # ── public API ────────────────────────────────────────────────────────

    def cache_get(self, key: str) -> Optional[str]:
        """Return the cached value for *key*, or ``None`` if missing/expired."""
        data = self._load()
        entry = data.get(key)
        if entry is None or self._is_expired(entry):
            self._misses += 1
            # Remove expired entry
            if entry is not None:
                del data[key]
                self._save(data)
            return None
        self._hits += 1
        return entry["value"]

    def cache_set(
        self,
        key: str,
        value: str,
        ttl_seconds: Optional[int] = 3600,
    ) -> str:
        """Store *value* under *key* with optional TTL.

        Parameters
        ----------
        key:         Cache key (string).
        value:       Value to store (string; serialise dicts/lists to JSON first).
        ttl_seconds: Seconds until expiry.  Pass ``None`` for no expiry.

        Returns
        -------
        Confirmation string.
        """
        data = self._load()
        now = time.time()
        entry: Dict[str, Any] = {
            "value": value,
            "created_at": now,
            "expires_at": (now + ttl_seconds) if ttl_seconds is not None else None,
        }
        data[key] = entry
        self._save(data)
        if ttl_seconds is not None:
            return f"cached '{key}' (expires in {ttl_seconds}s)"
        return f"cached '{key}' (no expiry)"

    def cache_delete(self, key: str) -> str:
        """Remove *key* from the cache."""
        data = self._load()
        if key in data:
            del data[key]
            self._save(data)
            return f"deleted '{key}'"
        return f"'{key}' not found"

    def cache_clear(self) -> str:
        """Remove all entries from the cache."""
        data = self._load()
        count = len(data)
        self._save({})
        return f"cleared {count} entries"

    def cache_stats(self) -> str:
        """Return cache statistics as a JSON string."""
        data = self._load()
        now = time.time()
        active = sum(
            1 for e in data.values()
            if e.get("expires_at") is None or e["expires_at"] > now
        )
        expired = len(data) - active
        stats = {
            "entries": active,
            "expired_entries": expired,
            "total_entries_on_disk": len(data),
            "session_hits": self._hits,
            "session_misses": self._misses,
        }
        return json.dumps(stats)

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call from the LLM."""
        if tool_name == "cache_get":
            result = self.cache_get(arguments["key"])
            return result if result is not None else "null"
        if tool_name == "cache_set":
            return self.cache_set(
                arguments["key"],
                arguments["value"],
                arguments.get("ttl_seconds", 3600),
            )
        if tool_name == "cache_delete":
            return self.cache_delete(arguments["key"])
        if tool_name == "cache_clear":
            return self.cache_clear()
        if tool_name == "cache_stats":
            return self.cache_stats()
        return f"Unknown cache tool: {tool_name}"

    # ── BaseTool protocol ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "cache"

    @property
    def description(self) -> str:
        return (
            "Key-value cache with optional TTL expiry. "
            "Store results of expensive operations and retrieve them later "
            "without repeating the computation or API call."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "cache_get",
                "description": (
                    "Retrieve a previously cached value by key. "
                    "Returns the value string, or null if the key does not exist or has expired."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key to look up.",
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "cache_set",
                "description": (
                    "Store a value in the cache under the given key. "
                    "Use ttl_seconds to control how long the value is valid. "
                    "Pass ttl_seconds=null for a value that never expires."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key.",
                        },
                        "value": {
                            "type": "string",
                            "description": "Value to cache. Serialise dicts/lists to JSON first.",
                        },
                        "ttl_seconds": {
                            "type": ["integer", "null"],
                            "description": (
                                "Seconds until expiry. Default 3600 (1 hour). "
                                "Pass null for no expiry."
                            ),
                            "default": 3600,
                        },
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "cache_delete",
                "description": "Remove a single key from the cache.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Cache key to remove.",
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "cache_clear",
                "description": "Remove ALL entries from the cache.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "cache_stats",
                "description": (
                    "Return cache statistics: number of active entries, expired entries, "
                    "and session hit/miss counts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        ]
