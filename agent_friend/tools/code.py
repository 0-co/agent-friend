"""code.py — Sandboxed code execution tool for agent-friend."""

import subprocess
import sys
import tempfile
import os
from typing import Any, Dict, List

from .base import BaseTool


class CodeTool(BaseTool):
    """Run Python or bash code in a subprocess with a configurable timeout.

    Parameters
    ----------
    timeout_seconds:  Maximum seconds to allow a subprocess to run (default 30).
    allow_network:    Not enforced at OS level; documents intent (default False).
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        allow_network: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.allow_network = allow_network

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Execute Python or bash code and return stdout/stderr."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "run_code",
                "description": (
                    "Execute code and return the output. "
                    "Supported languages: python, bash."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The code to execute.",
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "bash"],
                            "description": "Programming language (default: python).",
                        },
                    },
                    "required": ["code"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "run_code":
            return f"Unknown code tool: {tool_name}"
        language = arguments.get("language", "python")
        code = arguments["code"]
        return self._run(code, language)

    def _run(self, code: str, language: str) -> str:
        """Write code to a temp file and execute it via subprocess."""
        if language == "python":
            return self._run_python(code)
        if language == "bash":
            return self._run_bash(code)
        return f"Unsupported language: {language}. Use 'python' or 'bash'."

    def _run_python(self, code: str) -> str:
        """Execute Python code in a subprocess."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmpfile:
            tmpfile.write(code)
            tmpfile_path = tmpfile.name

        try:
            result = subprocess.run(
                [sys.executable, tmpfile_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return self._format_result(result)
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {self.timeout_seconds}s."
        except Exception as error:
            return f"Execution error: {error}"
        finally:
            try:
                os.unlink(tmpfile_path)
            except OSError:
                pass

    def _run_bash(self, code: str) -> str:
        """Execute bash code in a subprocess."""
        try:
            result = subprocess.run(
                ["bash", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            return self._format_result(result)
        except subprocess.TimeoutExpired:
            return f"Code execution timed out after {self.timeout_seconds}s."
        except FileNotFoundError:
            return "bash not found on this system."
        except Exception as error:
            return f"Execution error: {error}"

    def _format_result(self, result: subprocess.CompletedProcess) -> str:
        """Format subprocess result into a readable string."""
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if result.returncode != 0:
            parts.append(f"[exit code: {result.returncode}]")
        return "\n".join(parts) if parts else "(no output)"
