"""retry.py — RetryTool for agent-friend (stdlib only).

Agents calling external APIs need retry logic.  ``RetryTool`` wraps HTTP
requests and shell commands with configurable exponential back-off and
optional jitter, plus a simple circuit-breaker that stops hammering a
service when it is clearly down.

Usage (programmatic)::

    tool = RetryTool()

    # Retry an HTTP GET up to 3 times with exponential back-off
    result = tool.retry_http("GET", "https://api.example.com/data",
                             max_attempts=3, delay_seconds=1.0, backoff_factor=2.0)

    # Retry a shell command
    result = tool.retry_shell("curl -sf https://api.example.com/health",
                              max_attempts=5, delay_seconds=0.5)

    # Circuit breaker — stop after 3 failures within 60 seconds
    tool.circuit_create("payments", max_failures=3, reset_timeout_seconds=60)
    result = tool.circuit_call("payments", "POST", "https://pay.example.com/charge",
                               body='{"amount": 100}')
    status = tool.circuit_status("payments")   # {"state": "open", "failures": 3, ...}
    tool.circuit_reset("payments")
"""

import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import BaseTool


_USER_AGENT = "agent-friend/retry (https://github.com/0-co/agent-friend)"

_CIRCUIT_STATES = ("closed", "open", "half-open")


class RetryTool(BaseTool):
    """Retry HTTP requests and shell commands with exponential back-off.

    Also provides a lightweight circuit-breaker so agents don't keep
    hammering a failing service.

    Parameters
    ----------
    default_max_attempts:
        Default number of attempts before giving up (default 3).
    default_delay_seconds:
        Initial delay between attempts in seconds (default 1.0).
    default_backoff_factor:
        Multiplier applied to delay after each failure (default 2.0).
    default_jitter:
        When True, add ±25 % random jitter to delay to avoid thundering herd.
    timeout_seconds:
        HTTP request timeout in seconds (default 15).
    """

    def __init__(
        self,
        default_max_attempts: int = 3,
        default_delay_seconds: float = 1.0,
        default_backoff_factor: float = 2.0,
        default_jitter: bool = True,
        timeout_seconds: int = 15,
    ) -> None:
        self.default_max_attempts = default_max_attempts
        self.default_delay_seconds = default_delay_seconds
        self.default_backoff_factor = default_backoff_factor
        self.default_jitter = default_jitter
        self.timeout_seconds = timeout_seconds
        # Stats
        self._total_calls = 0
        self._total_retries = 0
        self._total_successes = 0
        self._total_failures = 0
        # Circuit breakers: name → {max_failures, reset_timeout, failures, state, opened_at}
        self._circuits: Dict[str, Dict[str, Any]] = {}

    # ── helpers ───────────────────────────────────────────────────────────

    def _sleep(self, seconds: float) -> None:
        """Sleep *seconds*.  Overridable in tests."""
        time.sleep(seconds)

    def _compute_delay(
        self,
        attempt: int,
        delay: float,
        backoff: float,
        jitter: bool,
    ) -> float:
        """Return the sleep duration for *attempt* (0-indexed)."""
        d = delay * (backoff ** attempt)
        if jitter:
            d = d * (0.75 + random.random() * 0.5)
        return d

    def _http_request(
        self,
        method: str,
        url: str,
        body: Optional[str],
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        """Execute a single HTTP request and return a result dict."""
        data: Optional[bytes] = None
        if body:
            data = body.encode("utf-8")

        req = urllib.request.Request(url, data=data, method=method.upper())
        req.add_header("User-Agent", _USER_AGENT)
        for key, val in headers.items():
            req.add_header(key, val)
        if data and "Content-Type" not in headers:
            req.add_header("Content-Type", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body_bytes = resp.read(65_536)
                body_text = body_bytes.decode("utf-8", errors="replace")
                return {
                    "ok": True,
                    "status": resp.status,
                    "body": body_text,
                }
        except urllib.error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read(4096).decode("utf-8", errors="replace")
            except Exception:
                pass
            return {
                "ok": False,
                "status": exc.code,
                "body": body_text,
                "error": str(exc),
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": 0,
                "body": "",
                "error": str(exc),
            }

    def _should_retry_http(self, result: Dict[str, Any]) -> bool:
        """Return True if the HTTP result warrants a retry."""
        if not result["ok"] and result["status"] == 0:
            # Network error — always retry
            return True
        status = result.get("status", 0)
        # Retry on 429 (rate limit) and 5xx (server errors)
        return status == 429 or (500 <= status < 600)

    # ── public API ────────────────────────────────────────────────────────

    def retry_http(
        self,
        method: str = "GET",
        url: str = "",
        body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        max_attempts: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        jitter: Optional[bool] = None,
        retry_on_status: Optional[List[int]] = None,
    ) -> str:
        """Make an HTTP request with automatic retry on failure.

        Returns a JSON string with keys: ok, status, body, attempts.
        """
        method = method.upper()
        headers = headers or {}
        ma = max_attempts if max_attempts is not None else self.default_max_attempts
        d = delay_seconds if delay_seconds is not None else self.default_delay_seconds
        bf = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        jit = jitter if jitter is not None else self.default_jitter
        extra_statuses: set = set(retry_on_status or [])

        self._total_calls += 1
        last_result: Dict[str, Any] = {"ok": False, "status": 0, "body": "", "error": "no attempts"}

        for attempt in range(ma):
            last_result = self._http_request(method, url, body, headers)
            status = last_result.get("status", 0)

            if last_result["ok"] and status not in extra_statuses:
                self._total_successes += 1
                last_result["attempts"] = attempt + 1
                return json.dumps(last_result)

            should_retry = self._should_retry_http(last_result) or status in extra_statuses
            if not should_retry or attempt == ma - 1:
                break

            self._total_retries += 1
            sleep_for = self._compute_delay(attempt, d, bf, jit)
            self._sleep(sleep_for)

        self._total_failures += 1
        last_result["attempts"] = ma
        return json.dumps(last_result)

    def retry_shell(
        self,
        command: str = "",
        max_attempts: Optional[int] = None,
        delay_seconds: Optional[float] = None,
        backoff_factor: Optional[float] = None,
        jitter: Optional[bool] = None,
    ) -> str:
        """Run a shell command with retry on non-zero exit code.

        Returns JSON with: ok, returncode, stdout, stderr, attempts.
        """
        ma = max_attempts if max_attempts is not None else self.default_max_attempts
        d = delay_seconds if delay_seconds is not None else self.default_delay_seconds
        bf = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        jit = jitter if jitter is not None else self.default_jitter

        self._total_calls += 1
        last: Dict[str, Any] = {}

        for attempt in range(ma):
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                last = {
                    "ok": proc.returncode == 0,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout[:4096],
                    "stderr": proc.stderr[:1024],
                    "attempts": attempt + 1,
                }
            except subprocess.TimeoutExpired:
                last = {
                    "ok": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "timeout",
                    "attempts": attempt + 1,
                }
            except Exception as exc:
                last = {
                    "ok": False,
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(exc),
                    "attempts": attempt + 1,
                }

            if last["ok"]:
                self._total_successes += 1
                return json.dumps(last)

            if attempt < ma - 1:
                self._total_retries += 1
                self._sleep(self._compute_delay(attempt, d, bf, jit))

        self._total_failures += 1
        last["attempts"] = ma
        return json.dumps(last)

    def retry_status(self) -> str:
        """Return retry statistics as JSON."""
        return json.dumps({
            "total_calls": self._total_calls,
            "total_retries": self._total_retries,
            "total_successes": self._total_successes,
            "total_failures": self._total_failures,
        })

    # ── circuit breaker ───────────────────────────────────────────────────

    def circuit_create(
        self,
        name: str,
        max_failures: int = 5,
        reset_timeout_seconds: float = 60.0,
    ) -> str:
        """Create or reset a named circuit breaker.

        States:
          - closed: calls pass through normally
          - open: calls are rejected immediately (circuit is tripped)
          - half-open: one trial call allowed to test if service recovered
        """
        self._circuits[name] = {
            "max_failures": max_failures,
            "reset_timeout": reset_timeout_seconds,
            "failures": 0,
            "state": "closed",
            "opened_at": None,
            "last_failure": None,
        }
        return json.dumps({"ok": True, "name": name, "state": "closed"})

    def _get_circuit(self, name: str) -> Dict[str, Any]:
        if name not in self._circuits:
            raise KeyError(f"Circuit breaker '{name}' not found. Call circuit_create first.")
        return self._circuits[name]

    def _circuit_check(self, circuit: Dict[str, Any]) -> bool:
        """Return True if the call should proceed, False if rejected."""
        state = circuit["state"]
        if state == "closed":
            return True
        if state == "open":
            # Check if reset timeout has elapsed → move to half-open
            opened_at = circuit.get("opened_at") or 0.0
            if time.time() - opened_at >= circuit["reset_timeout"]:
                circuit["state"] = "half-open"
                return True
            return False
        # half-open: allow one trial
        return True

    def _circuit_record_success(self, circuit: Dict[str, Any]) -> None:
        circuit["failures"] = 0
        circuit["state"] = "closed"
        circuit["opened_at"] = None

    def _circuit_record_failure(self, circuit: Dict[str, Any]) -> None:
        circuit["failures"] += 1
        circuit["last_failure"] = time.time()
        if circuit["failures"] >= circuit["max_failures"] or circuit["state"] == "half-open":
            circuit["state"] = "open"
            circuit["opened_at"] = time.time()

    def circuit_call(
        self,
        name: str,
        method: str = "GET",
        url: str = "",
        body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Make an HTTP call through a named circuit breaker.

        Returns JSON with: ok, status, body, circuit_state.
        If the circuit is open, returns immediately with ok=False.
        """
        try:
            circuit = self._get_circuit(name)
        except KeyError as exc:
            return json.dumps({"ok": False, "error": str(exc)})

        if not self._circuit_check(circuit):
            return json.dumps({
                "ok": False,
                "status": 0,
                "body": "",
                "error": f"Circuit '{name}' is open — service unavailable",
                "circuit_state": "open",
            })

        result = self._http_request(method.upper(), url, body, headers or {})

        if result["ok"] and not self._should_retry_http(result):
            self._circuit_record_success(circuit)
        else:
            self._circuit_record_failure(circuit)

        result["circuit_state"] = circuit["state"]
        return json.dumps(result)

    def circuit_status(self, name: str) -> str:
        """Return the current state of a named circuit breaker."""
        try:
            c = self._get_circuit(name)
        except KeyError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        return json.dumps({
            "ok": True,
            "name": name,
            "state": c["state"],
            "failures": c["failures"],
            "max_failures": c["max_failures"],
            "reset_timeout_seconds": c["reset_timeout"],
            "opened_at": c["opened_at"],
            "last_failure": c["last_failure"],
        })

    def circuit_reset(self, name: str) -> str:
        """Manually reset a circuit breaker to closed state."""
        try:
            c = self._get_circuit(name)
        except KeyError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        c["failures"] = 0
        c["state"] = "closed"
        c["opened_at"] = None
        return json.dumps({"ok": True, "name": name, "state": "closed"})

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "retry"

    @property
    def description(self) -> str:
        return (
            "Retry HTTP requests and shell commands with exponential back-off "
            "and optional circuit breaker.  Handles transient failures, rate limits, "
            "and 5xx errors automatically."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "retry_http",
                "description": (
                    "Make an HTTP request with automatic retry on failure. "
                    "Retries on network errors, 429 rate limits, and 5xx server errors "
                    "using exponential back-off with optional jitter."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method: GET, POST, PUT, PATCH, DELETE",
                            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        },
                        "url": {"type": "string", "description": "Full URL to request"},
                        "body": {"type": "string", "description": "Request body (JSON string)"},
                        "headers": {
                            "type": "object",
                            "description": "Optional HTTP headers",
                        },
                        "max_attempts": {
                            "type": "integer",
                            "description": "Maximum number of attempts (default 3)",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "Initial delay between retries in seconds (default 1.0)",
                        },
                        "backoff_factor": {
                            "type": "number",
                            "description": "Multiplier for delay after each failure (default 2.0)",
                        },
                        "jitter": {
                            "type": "boolean",
                            "description": "Add random jitter to avoid thundering herd (default true)",
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "retry_shell",
                "description": (
                    "Run a shell command with automatic retry on non-zero exit code. "
                    "Uses exponential back-off between attempts."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "max_attempts": {
                            "type": "integer",
                            "description": "Maximum number of attempts (default 3)",
                        },
                        "delay_seconds": {
                            "type": "number",
                            "description": "Initial delay between retries in seconds (default 1.0)",
                        },
                        "backoff_factor": {
                            "type": "number",
                            "description": "Multiplier for delay after each failure (default 2.0)",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "retry_status",
                "description": "Return retry statistics: total calls, retries, successes, failures.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "circuit_create",
                "description": (
                    "Create a named circuit breaker. The circuit opens (trips) after "
                    "max_failures consecutive failures and stays open until reset_timeout_seconds "
                    "have elapsed, then allows a single trial call."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Unique circuit breaker name"},
                        "max_failures": {
                            "type": "integer",
                            "description": "Failures before circuit opens (default 5)",
                        },
                        "reset_timeout_seconds": {
                            "type": "number",
                            "description": "Seconds before circuit moves to half-open (default 60)",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "circuit_call",
                "description": (
                    "Make an HTTP call through a named circuit breaker. "
                    "If the circuit is open, returns immediately without making the request."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Circuit breaker name"},
                        "method": {
                            "type": "string",
                            "description": "HTTP method",
                            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        },
                        "url": {"type": "string", "description": "URL to request"},
                        "body": {"type": "string", "description": "Request body"},
                        "headers": {"type": "object", "description": "HTTP headers"},
                    },
                    "required": ["name", "url"],
                },
            },
            {
                "name": "circuit_status",
                "description": "Get the current state of a named circuit breaker.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Circuit breaker name"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "circuit_reset",
                "description": "Manually reset a circuit breaker to closed (normal) state.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Circuit breaker name"},
                    },
                    "required": ["name"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "retry_http":
            return self.retry_http(**arguments)
        if tool_name == "retry_shell":
            return self.retry_shell(**arguments)
        if tool_name == "retry_status":
            return self.retry_status()
        if tool_name == "circuit_create":
            return self.circuit_create(**arguments)
        if tool_name == "circuit_call":
            return self.circuit_call(**arguments)
        if tool_name == "circuit_status":
            return self.circuit_status(**arguments)
        if tool_name == "circuit_reset":
            return self.circuit_reset(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
