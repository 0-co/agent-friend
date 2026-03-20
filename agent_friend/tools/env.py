"""env.py — EnvTool for agent-friend (stdlib only).

Agents can read, set, and verify environment variables, and load ``.env``
files.  Useful for checking that required API keys are present before
attempting to use them.

Usage::

    tool = EnvTool()
    tool.env_get("HOME")                   # "/home/user"
    tool.env_check(["OPENAI_API_KEY"])     # {"missing": ["OPENAI_API_KEY"]}
    tool.env_load(".env")                  # loads key=value pairs into os.environ
    tool.env_list("AWS_")                  # lists vars starting with AWS_
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


class EnvTool(BaseTool):
    """Read, set, and verify environment variables for the running process.

    Useful for agents that need to check their own configuration —
    API keys, database URLs, feature flags — before executing tasks.

    Parameters
    ----------
    safe_prefixes:
        Whitelist of variable name prefixes the tool is allowed to return.
        Empty list (default) means no restriction.  Useful to prevent
        accidental leakage of sensitive variables when the tool is used in
        a public-facing context.
    """

    # Variables never returned regardless of safe_prefixes
    _BLOCKED = re.compile(
        r"(PASSWORD|SECRET|TOKEN|KEY|CREDENTIAL|AUTH|PASSWD|PWD|PRIVATE)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        safe_prefixes: Optional[List[str]] = None,
        allow_sensitive: bool = False,
    ) -> None:
        # safe_prefixes=None means allow everything (except blocked names
        # unless allow_sensitive=True)
        self._safe_prefixes: Optional[List[str]] = safe_prefixes
        self._allow_sensitive = allow_sensitive

    # ── internal ─────────────────────────────────────────────────────────

    def _is_visible(self, key: str) -> bool:
        """Return True if the variable name may be returned to the agent."""
        if not self._allow_sensitive and self._BLOCKED.search(key):
            return False
        if self._safe_prefixes:
            return any(key.startswith(p) for p in self._safe_prefixes)
        return True

    # ── public API ────────────────────────────────────────────────────────

    def env_get(self, key: str, default: Optional[str] = None) -> str:
        """Return the value of environment variable *key*.

        Returns *default* (or the string ``"null"`` if not given) when the
        variable is not set.  Blocked / non-visible variables return
        ``"[hidden]"`` rather than their value.
        """
        if not self._allow_sensitive and self._BLOCKED.search(key):
            return "[hidden — use env_check to verify it is set]"
        value = os.environ.get(key)
        if value is None:
            return default if default is not None else "null"
        return value

    def env_set(self, key: str, value: str) -> str:
        """Set environment variable *key* to *value* for the current process.

        Changes are visible to child processes spawned after this call, but
        do *not* propagate to the parent shell.
        """
        os.environ[key] = value
        return f"set {key}={value!r}"

    def env_list(self, prefix: str = "") -> str:
        """Return a JSON object of environment variables.

        Parameters
        ----------
        prefix:
            Only include variables whose names start with this string.
            Pass an empty string (default) to list all visible variables.
        """
        result: Dict[str, str] = {}
        for k, v in sorted(os.environ.items()):
            if prefix and not k.startswith(prefix):
                continue
            if not self._is_visible(k):
                continue
            result[k] = v
        return json.dumps(result)

    def env_check(self, keys: List[str]) -> str:
        """Check whether all required variables are set.

        Returns a JSON object with:

        * ``present``  — list of keys that are set (values hidden).
        * ``missing``  — list of keys that are not set.
        * ``ok``       — ``true`` when all keys are present.
        """
        present = [k for k in keys if os.environ.get(k) is not None]
        missing = [k for k in keys if os.environ.get(k) is None]
        result = {
            "ok": len(missing) == 0,
            "present": present,
            "missing": missing,
        }
        return json.dumps(result)

    def env_load(self, path: str = ".env") -> str:
        """Load ``key=value`` pairs from a ``.env`` file into ``os.environ``.

        Supports:

        * ``KEY=value`` and ``KEY="quoted value"``
        * Lines starting with ``#`` are treated as comments
        * Empty lines are skipped
        * Existing environment variables are **not** overwritten by default

        Parameters
        ----------
        path:
            Path to the ``.env`` file.  Defaults to ``.env`` in the current
            working directory.

        Returns
        -------
        JSON summary ``{"loaded": [...], "skipped": [...], "errors": [...]}``
        """
        p = Path(path).expanduser()
        if not p.exists():
            return json.dumps({"error": f"File not found: {path}"})

        loaded: List[str] = []
        skipped: List[str] = []
        errors: List[str] = []

        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return json.dumps({"error": str(exc)})

        for lineno, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                errors.append(f"line {lineno}: no '=' found")
                continue
            key, _, raw_value = line.partition("=")
            key = key.strip()
            raw_value = raw_value.strip()
            # Strip optional surrounding quotes
            if len(raw_value) >= 2 and raw_value[0] in ('"', "'") and raw_value[0] == raw_value[-1]:
                raw_value = raw_value[1:-1]
            if not key:
                errors.append(f"line {lineno}: empty key")
                continue
            if key in os.environ:
                skipped.append(key)
            else:
                os.environ[key] = raw_value
                loaded.append(key)

        return json.dumps({"loaded": loaded, "skipped": skipped, "errors": errors})

    # ── BaseTool dispatch ─────────────────────────────────────────────────

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "env_get":
            return self.env_get(arguments["key"], arguments.get("default"))
        if tool_name == "env_set":
            return self.env_set(arguments["key"], arguments["value"])
        if tool_name == "env_list":
            return self.env_list(arguments.get("prefix", ""))
        if tool_name == "env_check":
            return self.env_check(arguments["keys"])
        if tool_name == "env_load":
            return self.env_load(arguments.get("path", ".env"))
        return f"Unknown env tool: {tool_name}"

    # ── BaseTool protocol ─────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "env"

    @property
    def description(self) -> str:
        return (
            "Read, set, and verify environment variables for the running process. "
            "Load .env files. Check that required API keys are present before using them."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "env_get",
                "description": (
                    "Get the value of a single environment variable. "
                    "Returns the value as a string, or 'null' if not set. "
                    "Sensitive variables (containing KEY, TOKEN, SECRET, etc.) "
                    "are hidden and return '[hidden]'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Environment variable name (e.g. 'HOME', 'USER', 'PATH').",
                        },
                        "default": {
                            "type": ["string", "null"],
                            "description": "Value to return if the variable is not set. Defaults to 'null'.",
                        },
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "env_set",
                "description": (
                    "Set an environment variable for the current process. "
                    "Visible to child processes spawned after this call. "
                    "Does not affect the parent shell."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Variable name."},
                        "value": {"type": "string", "description": "Value to set."},
                    },
                    "required": ["key", "value"],
                },
            },
            {
                "name": "env_list",
                "description": (
                    "List environment variables as a JSON object. "
                    "Pass a prefix to filter (e.g. 'AWS_' lists all AWS vars). "
                    "Sensitive variables are excluded from the output."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "prefix": {
                            "type": "string",
                            "description": "Filter by name prefix (e.g. 'AWS_')",
                            "default": "",
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "env_check",
                "description": (
                    "Check whether all required environment variables are set. "
                    "Returns {ok: bool, present: [...], missing: [...]}. "
                    "Use this to verify API keys before attempting to call external services."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of variable names to check.",
                        },
                    },
                    "required": ["keys"],
                },
            },
            {
                "name": "env_load",
                "description": (
                    "Load key=value pairs from a .env file into the current process environment. "
                    "Existing variables are not overwritten. "
                    "Returns {loaded: [...], skipped: [...], errors: [...]}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the .env file. Defaults to '.env' in the current directory.",
                            "default": ".env",
                        },
                    },
                    "required": [],
                },
            },
        ]
