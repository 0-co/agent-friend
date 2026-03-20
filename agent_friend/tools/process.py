"""process.py — ProcessTool for agent-friend (stdlib only).

Agents can run shell commands and scripts: capture stdout/stderr, check
exit codes, set working directories, and find executables — all through
a safe subprocess wrapper with configurable timeouts.

Uses only Python stdlib: ``subprocess``, ``shutil``, ``os``, ``shlex``.

Usage::

    tool = ProcessTool()
    tool.run("git status")                              # run a command
    tool.run("ls -la", cwd="/tmp")                     # with working dir
    tool.run_script("pip install requests\\necho done") # multi-line script
    tool.which("python3")                               # find executable
    tool.run("python3 --version")                      # check version
"""

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from .base import BaseTool


class ProcessTool(BaseTool):
    """Run shell commands and scripts from your AI agent.

    Provides a subprocess wrapper that captures stdout/stderr, enforces
    timeouts, and returns structured results — so agents can shell out
    to any tool installed on the system without writing boilerplate.

    All stdlib — no external libraries required.

    Parameters
    ----------
    timeout:        Default command timeout in seconds (default 30).
    max_output:     Maximum bytes to capture per stream (default 65536 / 64 KB).
    default_cwd:    Working directory for commands (default: current dir).
    """

    def __init__(
        self,
        timeout: int = 30,
        max_output: int = 65_536,
        default_cwd: Optional[str] = None,
    ) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self.default_cwd = default_cwd

    @property
    def name(self) -> str:
        return "process"

    @property
    def description(self) -> str:
        return (
            "Run shell commands and scripts. Captures stdout, stderr, and exit code. "
            "Use run() for single commands, run_script() for multi-line scripts, "
            "which() to locate executables."
        )

    # ── public Python API ─────────────────────────────────────────────────

    def _run_subprocess(
        self,
        args: Any,
        *,
        cwd: Optional[str],
        timeout: Optional[int],
        shell: bool = False,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Internal: run a subprocess and return structured result."""
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_cwd = cwd or self.default_cwd

        effective_env: Optional[Dict[str, str]] = None
        if env:
            effective_env = dict(os.environ)
            effective_env.update(env)

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                cwd=effective_cwd,
                shell=shell,
                env=effective_env,
            )
            stdout = result.stdout[: self.max_output]
            stderr = result.stderr[: self.max_output]
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": (
                    len(result.stdout) > self.max_output
                    or len(result.stderr) > self.max_output
                ),
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Command timed out after {effective_timeout}s",
                "truncated": False,
            }
        except FileNotFoundError as exc:
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "truncated": False,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Unexpected error: {exc}",
                "truncated": False,
            }

    def run(
        self,
        command: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        shell: bool = False,
    ) -> str:
        """Run a shell command and return stdout, stderr, and exit code.

        Parameters
        ----------
        command:    The command to run (e.g. ``"git status"`` or ``"ls -la /tmp"``).
        timeout:    Seconds before the process is killed (default: tool default).
        cwd:        Working directory for the command.
        env:        Extra environment variables to merge (e.g. ``{"DEBUG": "1"}``).
        shell:      Run via shell (enables pipes/redirection). Default False.

        Returns
        -------
        JSON string with keys: success, returncode, stdout, stderr, truncated.
        """
        if not command or not command.strip():
            return json.dumps({"error": "command must not be empty"})

        if shell:
            args: Any = command
        else:
            try:
                args = shlex.split(command)
            except ValueError as exc:
                return json.dumps({"error": f"Failed to parse command: {exc}"})

        result = self._run_subprocess(args, cwd=cwd, timeout=timeout, shell=shell, env=env)
        return json.dumps(result)

    def run_script(
        self,
        script: str,
        timeout: Optional[int] = None,
        cwd: Optional[str] = None,
        interpreter: str = "bash",
    ) -> str:
        """Run a multi-line shell script.

        Writes the script to a temporary file and executes it with the
        given interpreter.

        Parameters
        ----------
        script:         The script content (multiple commands on separate lines).
        timeout:        Seconds before the process is killed.
        cwd:            Working directory for the script.
        interpreter:    Interpreter — ``"bash"`` (default), ``"sh"``, ``"python3"``, etc.

        Returns
        -------
        JSON string with keys: success, returncode, stdout, stderr, truncated.
        """
        if not script or not script.strip():
            return json.dumps({"error": "script must not be empty"})

        interp_path = shutil.which(interpreter)
        if interp_path is None:
            return json.dumps({"error": f"Interpreter not found: {interpreter!r}"})

        suffix = ".sh" if interpreter in ("bash", "sh") else f".{interpreter.split('/')[-1]}"
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as fh:
                fh.write(script)
                tmp_path = fh.name
            result = self._run_subprocess(
                [interp_path, tmp_path],
                cwd=cwd,
                timeout=timeout,
                shell=False,
            )
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        return json.dumps(result)

    def which(self, name: str) -> str:
        """Find the full path of an executable in PATH.

        Parameters
        ----------
        name:   Executable name (e.g. ``"python3"``, ``"git"``, ``"ffmpeg"``).

        Returns
        -------
        JSON string: ``{"path": "/usr/bin/python3"}`` if found,
        or ``{"path": null, "error": "..."}`` if not found.
        """
        if not name or not name.strip():
            return json.dumps({"error": "name must not be empty"})

        path = shutil.which(name.strip())
        if path:
            return json.dumps({"path": path})
        return json.dumps({"path": None, "error": f"{name!r} not found in PATH"})

    # ── LLM interface ─────────────────────────────────────────────────────

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "run",
                "description": (
                    "Run a shell command and return stdout, stderr, and exit code. "
                    "Returns JSON with: success, returncode, stdout, stderr, truncated."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run, e.g. 'git status' or 'ls -la /tmp'.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max seconds to wait before killing the process.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for the command.",
                        },
                        "shell": {
                            "type": "boolean",
                            "description": "Run via shell — enables pipes and redirection. Default false.",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "run_script",
                "description": (
                    "Run a multi-line shell script. Writes to a temp file and executes it. "
                    "Returns JSON with: success, returncode, stdout, stderr, truncated."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Multi-line script content.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max seconds to wait before killing the process.",
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory for the script.",
                        },
                        "interpreter": {
                            "type": "string",
                            "description": "Interpreter to use: 'bash' (default), 'sh', 'python3', etc.",
                        },
                    },
                    "required": ["script"],
                },
            },
            {
                "name": "which",
                "description": "Find the full path of an executable in PATH.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Executable name, e.g. 'python3', 'git', 'ffmpeg'.",
                        },
                    },
                    "required": ["name"],
                },
            },
        ]

    def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        try:
            if tool_name == "run":
                return self.run(
                    command=tool_input["command"],
                    timeout=tool_input.get("timeout"),
                    cwd=tool_input.get("cwd"),
                    env=tool_input.get("env"),
                    shell=tool_input.get("shell", False),
                )
            elif tool_name == "run_script":
                return self.run_script(
                    script=tool_input["script"],
                    timeout=tool_input.get("timeout"),
                    cwd=tool_input.get("cwd"),
                    interpreter=tool_input.get("interpreter", "bash"),
                )
            elif tool_name == "which":
                return self.which(name=tool_input["name"])
            else:
                return json.dumps(
                    {"error": f"Unknown tool: {tool_name!r}. Valid: run, run_script, which"}
                )
        except KeyError as exc:
            return json.dumps({"error": f"Missing required argument: {exc}"})
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)})
