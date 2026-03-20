"""file.py — File read/write/list tool for agent-friend (no dependencies)."""

import os
import pathlib
from typing import Any, Dict, List, Optional

from .base import BaseTool


class FileTool(BaseTool):
    """Read, write, append, and list local files.

    Parameters
    ----------
    base_dir:       Root directory the agent is allowed to access (default: cwd).
                    Paths outside this directory are rejected to prevent accidental
                    access to system files.
    max_read_bytes: Maximum bytes to read from a single file (default: 32 KB).
                    Larger files are truncated with a notice.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        max_read_bytes: int = 32 * 1024,
    ) -> None:
        self.base_dir = pathlib.Path(base_dir or os.getcwd()).resolve()
        self.max_read_bytes = max_read_bytes

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Read, write, append, and list local files."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "file_read",
                "description": (
                    "Read a file and return its contents. "
                    f"Files larger than {self.max_read_bytes // 1024} KB are truncated."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read.",
                        }
                    },
                    "required": ["path"],
                },
            },
            {
                "name": "file_write",
                "description": "Write content to a file, overwriting any existing content.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "file_append",
                "description": "Append content to an existing file (creates it if missing).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to append to.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append.",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
            {
                "name": "file_list",
                "description": "List files and directories at a given path.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list (default: base_dir).",
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern to filter results (e.g. '*.py').",
                        },
                    },
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "file_read":
            return self._read(arguments["path"])
        if tool_name == "file_write":
            return self._write(arguments["path"], arguments["content"])
        if tool_name == "file_append":
            return self._append(arguments["path"], arguments["content"])
        if tool_name == "file_list":
            return self._list(arguments.get("path", "."), arguments.get("pattern"))
        return f"Unknown file tool: {tool_name}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> pathlib.Path:
        """Resolve path and check it's inside base_dir."""
        resolved = (self.base_dir / path).resolve()
        try:
            resolved.relative_to(self.base_dir)
        except ValueError:
            raise PermissionError(
                f"Access denied: '{path}' is outside the allowed directory "
                f"({self.base_dir}). Set base_dir to allow broader access."
            )
        return resolved

    def _read(self, path: str) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as err:
            return str(err)

        if not target.exists():
            return f"Error: file not found: {path}"
        if not target.is_file():
            return f"Error: '{path}' is a directory, not a file."

        try:
            size = target.stat().st_size
            with open(target, "rb") as fh:
                raw = fh.read(self.max_read_bytes)
            text = raw.decode("utf-8", errors="replace")
            if size > self.max_read_bytes:
                truncated_kb = self.max_read_bytes // 1024
                total_kb = size // 1024
                text += f"\n\n[... truncated — showing first {truncated_kb} KB of {total_kb} KB]"
            return text
        except Exception as err:
            return f"Error reading '{path}': {err}"

    def _write(self, path: str, content: str) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as err:
            return str(err)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Written {len(content)} characters to '{path}'."
        except Exception as err:
            return f"Error writing '{path}': {err}"

    def _append(self, path: str, content: str) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as err:
            return str(err)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(content)
            return f"Appended {len(content)} characters to '{path}'."
        except Exception as err:
            return f"Error appending to '{path}': {err}"

    def _list(self, path: str, pattern: Optional[str]) -> str:
        try:
            target = self._resolve(path)
        except PermissionError as err:
            return str(err)

        if not target.exists():
            return f"Error: path not found: {path}"
        if not target.is_dir():
            return f"Error: '{path}' is a file, not a directory."

        try:
            if pattern:
                entries = sorted(target.glob(pattern))
            else:
                entries = sorted(target.iterdir())

            lines = []
            for entry in entries:
                rel = entry.relative_to(self.base_dir)
                kind = "/" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
                lines.append(f"{rel}{kind}")

            if not lines:
                return f"Directory '{path}' is empty."
            return "\n".join(lines)
        except Exception as err:
            return f"Error listing '{path}': {err}"
