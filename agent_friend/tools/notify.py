"""notify.py — NotifyTool for agent-friend (stdlib only).

Agents can send notifications to the user when tasks complete or
important events occur — desktop popups, terminal bell, or file-based
alerts.  All methods are no-op safe: if the channel is unavailable
(no display, no terminal), they return a message rather than raising.

Usage::

    tool = NotifyTool()
    tool.notify("Report ready", "Your daily news summary is complete.")
    tool.notify_file("task_done", "Report complete", path="~/.agent_friend/notifications.log")
    tool.bell()
"""

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


_DEFAULT_LOG_PATH = "~/.agent_friend/notifications.log"


class NotifyTool(BaseTool):
    """Send notifications when tasks complete or important events occur.

    Supports three channels:
    - **desktop**: system notification popup (notify-send on Linux, osascript on macOS)
    - **bell**: terminal bell character (works in any terminal)
    - **file**: append notification to a log file

    All channels degrade gracefully — if a channel is unavailable,
    a descriptive string is returned instead of raising.

    Parameters
    ----------
    log_path:
        Default path for file-based notifications.
        Defaults to ``~/.agent_friend/notifications.log``.
    """

    def __init__(self, log_path: str = _DEFAULT_LOG_PATH) -> None:
        self.log_path = str(Path(log_path).expanduser())
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # ── channels ──────────────────────────────────────────────────────────

    def notify(self, title: str, message: str) -> str:
        """Send a notification using the best available channel.

        Tries desktop notification first; falls back to file log.

        Returns
        -------
        String describing what happened.
        """
        desktop_result = self._try_desktop(title, message)
        if desktop_result is not None:
            self._append_log(title, message)
            return f"desktop notification sent: {title!r}"
        # Fall back to file
        self._append_log(title, message)
        return f"notification logged to file: {title!r} — {message!r}"

    def notify_desktop(self, title: str, message: str) -> str:
        """Send a desktop notification popup.

        Uses ``notify-send`` on Linux (requires libnotify), ``osascript``
        on macOS.  Returns an error string if unavailable rather than raising.
        """
        result = self._try_desktop(title, message)
        if result is not None:
            return result
        return "desktop notification unavailable (no display / unsupported OS)"

    def notify_file(self, title: str, message: str, path: Optional[str] = None) -> str:
        """Append a notification entry to a log file.

        Parameters
        ----------
        title:   Short notification title.
        message: Notification body.
        path:    Path to log file. Defaults to ``~/.agent_friend/notifications.log``.
        """
        log_path = str(Path(path).expanduser()) if path else self.log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._append_log(title, message, log_path)
        return f"notification written to {log_path}"

    def bell(self) -> str:
        """Ring the terminal bell."""
        try:
            print("\a", end="", flush=True)
            return "terminal bell rung"
        except Exception:
            return "terminal bell unavailable"

    def read_notifications(self, n: int = 10, path: Optional[str] = None) -> str:
        """Return the last *n* notifications from the log file as JSON."""
        log_path = str(Path(path).expanduser()) if path else self.log_path
        if not os.path.exists(log_path):
            return "[]"
        entries = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except OSError:
            return "[]"
        return json.dumps(entries[-n:])

    # ── internal helpers ──────────────────────────────────────────────────

    def _try_desktop(self, title: str, message: str) -> Optional[str]:
        """Attempt desktop notification. Returns result string or None on failure."""
        system = platform.system()
        try:
            if system == "Linux":
                r = subprocess.run(
                    ["notify-send", title, message],
                    capture_output=True,
                    timeout=5,
                )
                if r.returncode == 0:
                    return f"desktop notification: {title!r}"
            elif system == "Darwin":
                script = f'display notification "{message}" with title "{title}"'
                r = subprocess.run(
                    ["osascript", "-e", script],
                    capture_output=True,
                    timeout=5,
                )
                if r.returncode == 0:
                    return f"desktop notification: {title!r}"
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    def _append_log(
        self,
        title: str,
        message: str,
        path: Optional[str] = None,
    ) -> None:
        log_path = path or self.log_path
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "title": title,
            "message": message,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call from the LLM."""
        if tool_name == "notify":
            return self.notify(arguments["title"], arguments["message"])
        if tool_name == "notify_desktop":
            return self.notify_desktop(arguments["title"], arguments["message"])
        if tool_name == "notify_file":
            return self.notify_file(
                arguments["title"],
                arguments["message"],
                arguments.get("path"),
            )
        if tool_name == "bell":
            return self.bell()
        if tool_name == "read_notifications":
            return self.read_notifications(
                arguments.get("n", 10),
                arguments.get("path"),
            )
        return f"Unknown notify tool: {tool_name}"

    # ── BaseTool protocol ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "notify"

    @property
    def description(self) -> str:
        return (
            "Send notifications when tasks complete or important events occur. "
            "Desktop popups, terminal bell, or file-based log."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "notify",
                "description": (
                    "Send a notification using the best available channel. "
                    "Tries desktop notification first, falls back to file log. "
                    "Use when a task completes or an important event occurs."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short notification title (e.g. 'Task complete', 'Error').",
                        },
                        "message": {
                            "type": "string",
                            "description": "Notification body with details.",
                        },
                    },
                    "required": ["title", "message"],
                },
            },
            {
                "name": "notify_desktop",
                "description": (
                    "Send a desktop notification popup. "
                    "Uses notify-send (Linux) or osascript (macOS). "
                    "Returns an error string if unavailable — does not raise."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Notification title."},
                        "message": {"type": "string", "description": "Notification body."},
                    },
                    "required": ["title", "message"],
                },
            },
            {
                "name": "notify_file",
                "description": (
                    "Append a notification entry to a log file (JSONL format). "
                    "Reliable across all environments — no display required."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Notification title."},
                        "message": {"type": "string", "description": "Notification body."},
                        "path": {
                            "type": "string",
                            "description": (
                                "Optional path to log file. "
                                "Defaults to ~/.agent_friend/notifications.log."
                            ),
                        },
                    },
                    "required": ["title", "message"],
                },
            },
            {
                "name": "bell",
                "description": "Ring the terminal bell to alert the user.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "read_notifications",
                "description": "Return the last N notifications from the log file as a JSON array.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "n": {
                            "type": "integer",
                            "description": "Number of recent notifications to return. Default 10.",
                            "default": 10,
                        },
                        "path": {
                            "type": "string",
                            "description": "Optional path to log file.",
                        },
                    },
                    "required": [],
                },
            },
        ]
