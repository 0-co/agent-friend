"""browser.py — Browser tool that delegates to agent-browser subprocess."""

import subprocess
import shutil
from typing import Any, Dict, List

from .base import BaseTool


class BrowserTool(BaseTool):
    """Browser automation via the agent-browser CLI (optional dependency).

    Calls `agent-browser open {url}` then `agent-browser snapshot --json`
    to retrieve accessible page text. Returns an error message if
    agent-browser is not installed rather than raising an exception.
    """

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return "Open a URL in a browser and return the page's text content."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "browse",
                "description": "Open a URL and return the page text content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The URL to open.",
                        },
                    },
                    "required": ["url"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "browse":
            return f"Unknown browser tool: {tool_name}"
        return self._browse(arguments["url"])

    def _browse(self, url: str) -> str:
        """Open URL and return page text via agent-browser subprocess."""
        if not self._agent_browser_available():
            return (
                "agent-browser is not installed. "
                "Install it to enable browser support: "
                "https://github.com/0-co/agent-browser"
            )

        try:
            open_result = subprocess.run(
                ["agent-browser", "open", url],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if open_result.returncode != 0:
                return f"agent-browser open failed: {open_result.stderr.strip()}"

            snapshot_result = subprocess.run(
                ["agent-browser", "snapshot", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if snapshot_result.returncode != 0:
                return f"agent-browser snapshot failed: {snapshot_result.stderr.strip()}"

            return self._extract_text(snapshot_result.stdout)

        except subprocess.TimeoutExpired:
            return "Browser operation timed out after 30s."
        except Exception as error:
            return f"Browser error: {error}"

    def _agent_browser_available(self) -> bool:
        """Check whether agent-browser is on PATH."""
        return shutil.which("agent-browser") is not None

    def _extract_text(self, json_output: str) -> str:
        """Extract accessible text from agent-browser JSON snapshot."""
        import json
        try:
            data = json.loads(json_output)
            # agent-browser snapshot format: {"text": "...", "url": "...", ...}
            if isinstance(data, dict):
                return str(data.get("text") or data.get("content") or json_output)
            return json_output
        except (json.JSONDecodeError, TypeError):
            return json_output if json_output else "(empty page)"
