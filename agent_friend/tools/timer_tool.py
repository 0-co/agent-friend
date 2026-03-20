"""timer_tool.py — TimerTool for agent-friend (stdlib only).

Stopwatch and countdown timers for agent workflow timing and benchmarking.

Features:
* timer_start / timer_stop — named stopwatch timers
* timer_elapsed — get elapsed time without stopping
* timer_lap — record a lap split
* timer_reset — reset a timer
* timer_list — list all active and stopped timers
* countdown_start / countdown_remaining — countdown with deadline
* timer_benchmark — time a shell command (N runs, returns avg/min/max)

Usage::

    tool = TimerTool()

    tool.timer_start("search")
    # ... do work ...
    result = tool.timer_stop("search")
    # {"name": "search", "elapsed_ms": 142.3, "laps": []}
"""

import json
import subprocess
import time
from typing import Any, Dict, List, Optional

from .base import BaseTool


class _Timer:
    """A single named stopwatch."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.start_time: float = time.monotonic()
        self.stop_time: Optional[float] = None
        self.laps: List[float] = []  # elapsed_ms at each lap

    def elapsed_ms(self) -> float:
        end = self.stop_time if self.stop_time is not None else time.monotonic()
        return (end - self.start_time) * 1000.0

    def lap(self) -> float:
        t = self.elapsed_ms()
        self.laps.append(t)
        return t

    def stop(self) -> float:
        if self.stop_time is None:
            self.stop_time = time.monotonic()
        return self.elapsed_ms()

    def is_running(self) -> bool:
        return self.stop_time is None

    def to_dict(self) -> Dict[str, Any]:
        elapsed = self.elapsed_ms()
        return {
            "name": self.name,
            "elapsed_ms": round(elapsed, 3),
            "elapsed_s": round(elapsed / 1000.0, 3),
            "running": self.is_running(),
            "laps": [round(l, 3) for l in self.laps],
        }


class _Countdown:
    """A single named countdown timer."""

    def __init__(self, name: str, duration_ms: float) -> None:
        self.name = name
        self.start_time = time.monotonic()
        self.duration_ms = duration_ms

    def remaining_ms(self) -> float:
        elapsed = (time.monotonic() - self.start_time) * 1000.0
        return max(0.0, self.duration_ms - elapsed)

    def expired(self) -> bool:
        return self.remaining_ms() == 0.0

    def to_dict(self) -> Dict[str, Any]:
        rem = self.remaining_ms()
        return {
            "name": self.name,
            "remaining_ms": round(rem, 3),
            "remaining_s": round(rem / 1000.0, 3),
            "expired": rem == 0.0,
            "duration_ms": self.duration_ms,
        }


class TimerTool(BaseTool):
    """Stopwatch and countdown timers for agent workflow benchmarking.

    Named stopwatches with lap support, countdown timers, and shell
    command benchmarking.  All times in milliseconds or seconds.
    """

    def __init__(self) -> None:
        self._timers: Dict[str, _Timer] = {}
        self._countdowns: Dict[str, _Countdown] = {}

    # ── stopwatch API ─────────────────────────────────────────────────

    def timer_start(self, name: str) -> str:
        """Start (or restart) a named stopwatch.

        Returns ``{name, started: true}``.
        """
        self._timers[name] = _Timer(name)
        return json.dumps({"name": name, "started": True})

    def timer_stop(self, name: str) -> str:
        """Stop a named stopwatch.

        Returns ``{name, elapsed_ms, elapsed_s, laps}``.
        """
        t = self._timers.get(name)
        if t is None:
            return json.dumps({"error": f"No timer named '{name}'"})
        t.stop()
        return json.dumps(t.to_dict())

    def timer_elapsed(self, name: str) -> str:
        """Get elapsed time without stopping the timer.

        Returns ``{name, elapsed_ms, elapsed_s, running}``.
        """
        t = self._timers.get(name)
        if t is None:
            return json.dumps({"error": f"No timer named '{name}'"})
        return json.dumps(t.to_dict())

    def timer_lap(self, name: str) -> str:
        """Record a lap split on a running timer.

        Returns ``{name, lap_ms, lap_number, total_elapsed_ms}``.
        """
        t = self._timers.get(name)
        if t is None:
            return json.dumps({"error": f"No timer named '{name}'"})
        if not t.is_running():
            return json.dumps({"error": f"Timer '{name}' is stopped"})
        lap_ms = t.lap()
        return json.dumps({
            "name": name,
            "lap_ms": round(lap_ms, 3),
            "lap_number": len(t.laps),
            "total_elapsed_ms": round(t.elapsed_ms(), 3),
        })

    def timer_reset(self, name: str) -> str:
        """Reset a timer to zero (restarts it running).

        Returns ``{name, reset: true}``.
        """
        if name not in self._timers:
            return json.dumps({"error": f"No timer named '{name}'"})
        self._timers[name] = _Timer(name)
        return json.dumps({"name": name, "reset": True})

    def timer_delete(self, name: str) -> str:
        """Delete a timer.

        Returns ``{deleted: bool}``.
        """
        existed = name in self._timers
        if existed:
            del self._timers[name]
        return json.dumps({"deleted": existed, "name": name})

    def timer_list(self) -> str:
        """List all timers with their current state.

        Returns a JSON array of timer objects.
        """
        return json.dumps([t.to_dict() for t in self._timers.values()])

    # ── countdown API ─────────────────────────────────────────────────

    def countdown_start(self, name: str, seconds: float) -> str:
        """Start a countdown timer for *seconds* seconds.

        Returns ``{name, duration_ms, started: true}``.
        """
        if seconds <= 0:
            return json.dumps({"error": "seconds must be > 0"})
        self._countdowns[name] = _Countdown(name, seconds * 1000.0)
        return json.dumps({"name": name, "duration_ms": seconds * 1000.0, "started": True})

    def countdown_remaining(self, name: str) -> str:
        """Get remaining time on a countdown.

        Returns ``{name, remaining_ms, remaining_s, expired}``.
        """
        c = self._countdowns.get(name)
        if c is None:
            return json.dumps({"error": f"No countdown named '{name}'"})
        return json.dumps(c.to_dict())

    def countdown_list(self) -> str:
        """List all countdowns with remaining time."""
        return json.dumps([c.to_dict() for c in self._countdowns.values()])

    # ── benchmark API ─────────────────────────────────────────────────

    def timer_benchmark(
        self,
        command: str,
        runs: int = 3,
        timeout_s: float = 10.0,
    ) -> str:
        """Time a shell command over *runs* executions.

        Returns ``{command, runs, avg_ms, min_ms, max_ms, total_ms, results}``.
        *results* is a list of per-run elapsed times.
        """
        if runs < 1 or runs > 50:
            return json.dumps({"error": "runs must be between 1 and 50"})
        if timeout_s <= 0:
            return json.dumps({"error": "timeout_s must be > 0"})

        times: List[float] = []
        for _ in range(runs):
            start = time.monotonic()
            try:
                subprocess.run(
                    command,
                    shell=True,
                    timeout=timeout_s,
                    capture_output=True,
                )
            except subprocess.TimeoutExpired:
                return json.dumps({"error": f"Command timed out after {timeout_s}s"})
            elapsed = (time.monotonic() - start) * 1000.0
            times.append(round(elapsed, 3))

        total = sum(times)
        return json.dumps({
            "command": command,
            "runs": runs,
            "avg_ms": round(total / runs, 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "total_ms": round(total, 3),
            "results": times,
        })

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "timer"

    @property
    def description(self) -> str:
        return (
            "Named stopwatch timers with lap support, countdown timers, and shell "
            "command benchmarking. All times in ms/s. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "timer_start",
                "description": "Start (or restart) a named stopwatch. Returns {name, started}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_stop",
                "description": "Stop a timer. Returns {name, elapsed_ms, elapsed_s, laps}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_elapsed",
                "description": "Get elapsed time without stopping. Returns {name, elapsed_ms, elapsed_s, running}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_lap",
                "description": "Record a lap split. Returns {name, lap_ms, lap_number, total_elapsed_ms}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_reset",
                "description": "Reset and restart a timer.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_delete",
                "description": "Delete a timer. Returns {deleted: bool}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "timer_list",
                "description": "List all timers with elapsed times and lap splits.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "countdown_start",
                "description": "Start a countdown for N seconds. Returns {name, duration_ms, started}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "seconds": {"type": "number"},
                    },
                    "required": ["name", "seconds"],
                },
            },
            {
                "name": "countdown_remaining",
                "description": "Get remaining countdown time. Returns {remaining_ms, remaining_s, expired}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "countdown_list",
                "description": "List all countdowns with remaining times.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "timer_benchmark",
                "description": "Benchmark a shell command (N runs). Returns {avg_ms, min_ms, max_ms, total_ms}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "runs": {"type": "integer", "description": "Number of runs (1-50)"},
                        "timeout_s": {"type": "number"},
                    },
                    "required": ["command"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "timer_start":
            return self.timer_start(**arguments)
        if tool_name == "timer_stop":
            return self.timer_stop(**arguments)
        if tool_name == "timer_elapsed":
            return self.timer_elapsed(**arguments)
        if tool_name == "timer_lap":
            return self.timer_lap(**arguments)
        if tool_name == "timer_reset":
            return self.timer_reset(**arguments)
        if tool_name == "timer_delete":
            return self.timer_delete(**arguments)
        if tool_name == "timer_list":
            return self.timer_list()
        if tool_name == "countdown_start":
            return self.countdown_start(**arguments)
        if tool_name == "countdown_remaining":
            return self.countdown_remaining(**arguments)
        if tool_name == "countdown_list":
            return self.countdown_list()
        if tool_name == "timer_benchmark":
            return self.timer_benchmark(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
