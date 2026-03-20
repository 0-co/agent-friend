"""rate_limit.py — RateLimitTool for agent-friend (stdlib only).

Rate limiting is the complement to RetryTool: instead of retrying when a
service rejects you, RateLimitTool prevents you from exceeding limits in the
first place.  Three algorithms are provided:

* **Fixed window** — simple counter reset each window. Fast, slight burst
  at boundary.
* **Sliding window** — precise per-request timestamp log. No boundary burst.
* **Token bucket** — smooth continuous rate with configurable burst capacity.

Usage::

    tool = RateLimitTool()

    # Create a fixed-window limiter: 10 requests per 60 seconds
    tool.limiter_create("openai", max_calls=10, window_seconds=60,
                        algorithm="fixed")

    # Check before calling
    result = json.loads(tool.limiter_check("openai"))
    if result["allowed"]:
        # make the API call
        tool.limiter_consume("openai")

    # Or check-and-consume atomically
    result = json.loads(tool.limiter_acquire("openai"))
    # {"allowed": true, "remaining": 9, "reset_in_seconds": 45.2}
"""

import json
import time
from collections import deque
from typing import Any, Dict, List, Optional

from .base import BaseTool


# ---------------------------------------------------------------------------
# Limiter implementations
# ---------------------------------------------------------------------------

class _FixedWindow:
    """Simple fixed-window counter."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._count = 0
        self._window_start = time.monotonic()

    def _maybe_reset(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= self.window_seconds:
            elapsed_windows = int((now - self._window_start) / self.window_seconds)
            self._window_start += elapsed_windows * self.window_seconds
            self._count = 0

    def check(self) -> Dict[str, Any]:
        self._maybe_reset()
        now = time.monotonic()
        remaining = max(0, self.max_calls - self._count)
        reset_in = max(0.0, self.window_seconds - (now - self._window_start))
        return {
            "allowed": self._count < self.max_calls,
            "remaining": remaining,
            "reset_in_seconds": round(reset_in, 3),
            "count": self._count,
            "limit": self.max_calls,
        }

    def consume(self) -> bool:
        self._maybe_reset()
        if self._count < self.max_calls:
            self._count += 1
            return True
        return False

    def reset(self) -> None:
        self._count = 0
        self._window_start = time.monotonic()

    def status(self) -> Dict[str, Any]:
        info = self.check()
        info["algorithm"] = "fixed"
        info["window_seconds"] = self.window_seconds
        return info


class _SlidingWindow:
    """Sliding window log — precise, no boundary burst."""

    def __init__(self, max_calls: int, window_seconds: float) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._log: deque = deque()  # deque of monotonic timestamps

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.window_seconds
        while self._log and self._log[0] <= cutoff:
            self._log.popleft()

    def check(self) -> Dict[str, Any]:
        self._prune()
        count = len(self._log)
        remaining = max(0, self.max_calls - count)
        # earliest timestamp tells us when a slot opens
        if self._log and count >= self.max_calls:
            reset_in = max(0.0, self._log[0] + self.window_seconds - time.monotonic())
        else:
            reset_in = 0.0
        return {
            "allowed": count < self.max_calls,
            "remaining": remaining,
            "reset_in_seconds": round(reset_in, 3),
            "count": count,
            "limit": self.max_calls,
        }

    def consume(self) -> bool:
        self._prune()
        if len(self._log) < self.max_calls:
            self._log.append(time.monotonic())
            return True
        return False

    def reset(self) -> None:
        self._log.clear()

    def status(self) -> Dict[str, Any]:
        info = self.check()
        info["algorithm"] = "sliding"
        info["window_seconds"] = self.window_seconds
        return info


class _TokenBucket:
    """Token bucket — smooth rate with configurable burst capacity."""

    def __init__(
        self,
        rate: float,        # tokens per second
        capacity: float,    # burst capacity
    ) -> None:
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def check(self, cost: float = 1.0) -> Dict[str, Any]:
        self._refill()
        allowed = self._tokens >= cost
        if allowed:
            wait = 0.0
        else:
            wait = (cost - self._tokens) / self.rate
        return {
            "allowed": allowed,
            "tokens": round(self._tokens, 4),
            "capacity": self.capacity,
            "rate_per_second": self.rate,
            "wait_seconds": round(wait, 3),
        }

    def consume(self, cost: float = 1.0) -> bool:
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    def reset(self) -> None:
        self._tokens = self.capacity
        self._last_refill = time.monotonic()

    def status(self) -> Dict[str, Any]:
        info = self.check()
        info["algorithm"] = "token_bucket"
        return info


# ---------------------------------------------------------------------------
# RateLimitTool
# ---------------------------------------------------------------------------

class RateLimitTool(BaseTool):
    """Rate limiting for AI agents: fixed window, sliding window, token bucket.

    Create named limiters and check/consume before making API calls to avoid
    rate limit errors without relying on RetryTool's backoff.

    Parameters
    ----------
    max_limiters:
        Maximum number of named limiters that can be created (default 100).
    """

    def __init__(self, max_limiters: int = 100) -> None:
        self.max_limiters = max_limiters
        self._limiters: Dict[str, Any] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _get(self, name: str) -> Any:
        limiter = self._limiters.get(name)
        if limiter is None:
            raise KeyError(f"No limiter named '{name}'")
        return limiter

    # ── public API ────────────────────────────────────────────────────────

    def limiter_create(
        self,
        name: str,
        max_calls: int = 10,
        window_seconds: float = 60.0,
        algorithm: str = "fixed",
        rate_per_second: Optional[float] = None,
        burst_capacity: Optional[float] = None,
    ) -> str:
        """Create a named rate limiter.

        Parameters
        ----------
        name:
            Unique name for this limiter (e.g. ``"openai"``, ``"github"``).
        max_calls:
            Maximum calls allowed per window (fixed/sliding) or token bucket
            capacity (token_bucket).
        window_seconds:
            Window duration in seconds (fixed/sliding algorithms).
        algorithm:
            One of ``"fixed"``, ``"sliding"``, ``"token_bucket"``.
        rate_per_second:
            Token bucket only — refill rate (tokens/second). Defaults to
            ``max_calls / window_seconds``.
        burst_capacity:
            Token bucket only — max tokens. Defaults to ``max_calls``.

        Returns ``{"created": true, "name": "...", "algorithm": "..."}``
        """
        if name in self._limiters:
            return json.dumps({"error": f"Limiter '{name}' already exists. Use limiter_reset to clear it."})
        if len(self._limiters) >= self.max_limiters:
            return json.dumps({"error": f"Max limiters ({self.max_limiters}) reached."})

        alg = algorithm.lower()
        if alg == "fixed":
            self._limiters[name] = _FixedWindow(max_calls, window_seconds)
        elif alg == "sliding":
            self._limiters[name] = _SlidingWindow(max_calls, window_seconds)
        elif alg in ("token_bucket", "bucket"):
            rate = rate_per_second if rate_per_second is not None else max_calls / window_seconds
            cap = burst_capacity if burst_capacity is not None else float(max_calls)
            self._limiters[name] = _TokenBucket(rate=rate, capacity=cap)
        else:
            return json.dumps({"error": f"Unknown algorithm '{algorithm}'. Use: fixed, sliding, token_bucket"})

        return json.dumps({"created": True, "name": name, "algorithm": alg})

    def limiter_check(self, name: str, cost: float = 1.0) -> str:
        """Check if a request would be allowed **without consuming** a token.

        Returns ``{allowed, remaining, reset_in_seconds, count, limit}``
        (or ``{allowed, tokens, ...}`` for token bucket).
        """
        try:
            limiter = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if isinstance(limiter, _TokenBucket):
            return json.dumps(limiter.check(cost))
        return json.dumps(limiter.check())

    def limiter_consume(self, name: str, cost: float = 1.0) -> str:
        """Consume one token from the limiter (marks a request as made).

        Returns ``{"consumed": true/false}``.  False means over limit.
        """
        try:
            limiter = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if isinstance(limiter, _TokenBucket):
            ok = limiter.consume(cost)
        else:
            ok = limiter.consume()
        return json.dumps({"consumed": ok})

    def limiter_acquire(self, name: str, cost: float = 1.0) -> str:
        """Atomically check **and** consume — the most common operation.

        Returns the same dict as ``limiter_check`` with ``allowed`` set.
        If allowed, the token is consumed; if not allowed, nothing changes.
        """
        try:
            limiter = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if isinstance(limiter, _TokenBucket):
            status = limiter.check(cost)
            if status["allowed"]:
                limiter.consume(cost)
        else:
            status = limiter.check()
            if status["allowed"]:
                limiter.consume()
        return json.dumps(status)

    def limiter_status(self, name: str) -> str:
        """Return current state of a limiter including its algorithm."""
        try:
            limiter = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps(limiter.status())

    def limiter_reset(self, name: str) -> str:
        """Reset a limiter's counters/tokens to initial state."""
        try:
            limiter = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        limiter.reset()
        return json.dumps({"reset": True, "name": name})

    def limiter_delete(self, name: str) -> str:
        """Remove a limiter entirely."""
        if name not in self._limiters:
            return json.dumps({"error": f"No limiter named '{name}'"})
        del self._limiters[name]
        return json.dumps({"deleted": True, "name": name})

    def limiter_list(self) -> str:
        """List all active limiters with their names and algorithms."""
        result = []
        for name, limiter in self._limiters.items():
            info = limiter.status()
            info["name"] = name
            result.append(info)
        return json.dumps(result)

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "rate_limit"

    @property
    def description(self) -> str:
        return (
            "Rate limiting for agent API calls: fixed window, sliding window, "
            "and token bucket algorithms. Create named limiters and check/acquire "
            "before calling external services to stay within API limits. "
            "All stdlib — no external dependencies."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "limiter_create",
                "description": (
                    "Create a named rate limiter. "
                    "algorithm: fixed (window counter), sliding (no burst), "
                    "or token_bucket (smooth rate with burst)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique limiter name"},
                        "max_calls": {"type": "integer", "description": "Max calls per window (default 10)"},
                        "window_seconds": {"type": "number", "description": "Window size in seconds (default 60)"},
                        "algorithm": {
                            "type": "string",
                            "enum": ["fixed", "sliding", "token_bucket"],
                            "description": "Algorithm to use (default: fixed)",
                        },
                        "rate_per_second": {"type": "number", "description": "Token bucket: refill rate (tokens/sec)"},
                        "burst_capacity": {"type": "number", "description": "Token bucket: max burst size"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_check",
                "description": (
                    "Check if a request is allowed WITHOUT consuming a token. "
                    "Returns {allowed, remaining, reset_in_seconds} or {allowed, tokens, wait_seconds} for token bucket."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost": {"type": "number", "description": "Token cost (default 1.0, token bucket only)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_consume",
                "description": "Record that a request was made (consumes one token). Returns {consumed: true/false}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost": {"type": "number", "description": "Token cost (default 1.0)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_acquire",
                "description": (
                    "Atomically check AND consume — the most common operation. "
                    "If allowed, consumes a token and returns {allowed: true, ...}. "
                    "If denied, returns {allowed: false, ...} without consuming."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "cost": {"type": "number", "description": "Token cost (default 1.0)"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_status",
                "description": "Get the current state of a limiter: count, remaining, algorithm, window size.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_reset",
                "description": "Reset a limiter's counters to their initial state.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_delete",
                "description": "Remove a named limiter entirely.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "limiter_list",
                "description": "List all active limiters with their current state.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "limiter_create":
            return self.limiter_create(**arguments)
        if tool_name == "limiter_check":
            return self.limiter_check(**arguments)
        if tool_name == "limiter_consume":
            return self.limiter_consume(**arguments)
        if tool_name == "limiter_acquire":
            return self.limiter_acquire(**arguments)
        if tool_name == "limiter_status":
            return self.limiter_status(**arguments)
        if tool_name == "limiter_reset":
            return self.limiter_reset(**arguments)
        if tool_name == "limiter_delete":
            return self.limiter_delete(**arguments)
        if tool_name == "limiter_list":
            return self.limiter_list()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
