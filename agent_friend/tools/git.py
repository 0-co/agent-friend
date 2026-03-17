"""git.py — GitTool for reading and committing to git repositories."""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


class GitTool(BaseTool):
    """Git operations for AI agents — read status, diffs, logs, and commit changes.

    Read-only operations (git_status, git_diff, git_log, git_branch_list) are
    safe to use freely. Write operations (git_add, git_commit, git_branch_create,
    git_checkout) modify repo state.

    All operations default to the current working directory but can be pointed
    at a specific repo with repo_dir.

    Usage::

        from agent_friend import Friend, GitTool

        git = GitTool(repo_dir="/path/to/repo")
        friend = Friend(tools=[git])

        # Or use default directory
        friend = Friend(tools=["git"])
        friend.chat("What's the current git status?")
        friend.chat("Show me the last 3 commits")
        friend.chat("Stage changes to main.py and commit with message 'Fix bug'")
    """

    def __init__(self, repo_dir: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        repo_dir:
            Path to the git repository. Defaults to current working directory.
        """
        self._repo_dir = str(Path(repo_dir).expanduser()) if repo_dir else None

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Git operations: read status, view diffs and logs, stage files, "
            "commit changes, and manage branches."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "git_status",
                "description": "Show the working tree status — staged, unstaged, and untracked files.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                },
            },
            {
                "name": "git_diff",
                "description": "Show changes in the working tree or staging area.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "staged": {"type": "boolean", "description": "If true, show staged changes (default: unstaged)"},
                        "path": {"type": "string", "description": "Limit diff to a specific file or directory (optional)"},
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                },
            },
            {
                "name": "git_log",
                "description": "Show recent commit history.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "description": "Number of commits to show (default: 5)"},
                        "oneline": {"type": "boolean", "description": "Show one line per commit (default: true)"},
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                },
            },
            {
                "name": "git_add",
                "description": "Stage files for commit.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths to stage. Use ['.'] to stage all changes.",
                        },
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                    "required": ["paths"],
                },
            },
            {
                "name": "git_commit",
                "description": "Commit staged changes with a message.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"},
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "git_branch_list",
                "description": "List all local branches.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                },
            },
            {
                "name": "git_branch_create",
                "description": "Create a new branch from the current HEAD.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Branch name"},
                        "checkout": {
                            "type": "boolean",
                            "description": "Switch to the new branch after creating (default: true)",
                        },
                        "repo_dir": {"type": "string", "description": "Repo path (default: cwd)"},
                    },
                    "required": ["name"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        handlers = {
            "git_status": self._status,
            "git_diff": self._diff,
            "git_log": self._log,
            "git_add": self._add,
            "git_commit": self._commit,
            "git_branch_list": self._branch_list,
            "git_branch_create": self._branch_create,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return f"Unknown git operation: {tool_name}"
        try:
            return handler(arguments)
        except Exception as exc:
            return f"Git error: {exc}"

    # ── private helpers ───────────────────────────────────────────────────────

    def _run(self, args: List[str], repo_dir: Optional[str] = None) -> str:
        """Run a git command and return combined stdout+stderr."""
        cwd = repo_dir or self._repo_dir or os.getcwd()
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "(no output)"

    def _status(self, args: Dict[str, Any]) -> str:
        return self._run(["status"], args.get("repo_dir"))

    def _diff(self, args: Dict[str, Any]) -> str:
        cmd = ["diff"]
        if args.get("staged"):
            cmd.append("--cached")
        if args.get("path"):
            cmd += ["--", args["path"]]
        result = self._run(cmd, args.get("repo_dir"))
        if result == "(no output)":
            return "No changes to show." if args.get("staged") else "Working tree is clean."
        # Truncate very large diffs
        if len(result) > 4000:
            result = result[:4000] + f"\n... (truncated, {len(result) - 4000} more chars)"
        return result

    def _log(self, args: Dict[str, Any]) -> str:
        n = args.get("n", 5)
        oneline = args.get("oneline", True)
        cmd = ["log", f"-{n}"]
        if oneline:
            cmd.append("--oneline")
        else:
            cmd += ["--format=%h %s%n  Author: %an <%ae>%n  Date:   %ad", "--date=short"]
        return self._run(cmd, args.get("repo_dir"))

    def _add(self, args: Dict[str, Any]) -> str:
        paths = args.get("paths", ["."])
        if isinstance(paths, str):
            paths = [paths]
        result = self._run(["add"] + paths, args.get("repo_dir"))
        # Confirm what was staged
        status = self._run(["status", "--short"], args.get("repo_dir"))
        return f"Staged. Current status:\n{status}"

    def _commit(self, args: Dict[str, Any]) -> str:
        message = args["message"]
        return self._run(["commit", "-m", message], args.get("repo_dir"))

    def _branch_list(self, args: Dict[str, Any]) -> str:
        return self._run(["branch", "-v"], args.get("repo_dir"))

    def _branch_create(self, args: Dict[str, Any]) -> str:
        name = args["name"]
        checkout = args.get("checkout", True)
        if checkout:
            return self._run(["checkout", "-b", name], args.get("repo_dir"))
        return self._run(["branch", name], args.get("repo_dir"))

    # ── convenience Python API ────────────────────────────────────────────────

    def status(self, repo_dir: Optional[str] = None) -> str:
        """Return git status as a string."""
        return self._run(["status"], repo_dir)

    def diff(self, staged: bool = False, path: Optional[str] = None, repo_dir: Optional[str] = None) -> str:
        """Return git diff output."""
        return self._diff({"staged": staged, "path": path, "repo_dir": repo_dir})

    def log(self, n: int = 5, oneline: bool = True, repo_dir: Optional[str] = None) -> str:
        """Return git log output."""
        return self._log({"n": n, "oneline": oneline, "repo_dir": repo_dir})

    def add(self, paths: List[str], repo_dir: Optional[str] = None) -> str:
        """Stage files for commit."""
        return self._add({"paths": paths, "repo_dir": repo_dir})

    def commit(self, message: str, repo_dir: Optional[str] = None) -> str:
        """Commit staged changes."""
        return self._commit({"message": message, "repo_dir": repo_dir})

    def branch_list(self, repo_dir: Optional[str] = None) -> str:
        """List all local branches."""
        return self._branch_list({"repo_dir": repo_dir})

    def branch_create(self, name: str, checkout: bool = True, repo_dir: Optional[str] = None) -> str:
        """Create a new branch."""
        return self._branch_create({"name": name, "checkout": checkout, "repo_dir": repo_dir})
