"""scheduler.py — SchedulerTool for agent-friend (no required dependencies).

Schedule tasks for your agent to run periodically or at a specific time.
State is persisted to ``~/.agent_friend/scheduler.json``.

Usage::

    tool = SchedulerTool()
    tool.schedule("daily_news", "Search for AI news and summarize top 3", interval_minutes=1440)
    tool.schedule("one_shot", "Check my email", run_at="2026-03-13T08:00:00")
    pending = tool.run_pending()   # returns list of task dicts for due tasks
    tool.list_scheduled()          # returns list of all task dicts
    tool.cancel("daily_news")      # removes a task by id
    tool.clear_all()               # removes all tasks
"""

import datetime
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BaseTool


_DEFAULT_STORAGE_DIR = Path.home() / ".agent_friend"


def _now_iso() -> str:
    """Return the current local time as an ISO-format string."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def _parse_iso(s: str) -> datetime.datetime:
    """Parse an ISO-format datetime string."""
    return datetime.datetime.fromisoformat(s)


class SchedulerTool(BaseTool):
    """Schedule tasks for your agent to run periodically or at a specific time.

    Stores the schedule in ``scheduler.json`` inside *storage_dir* (default
    ``~/.agent_friend/``).  All timestamps are stored as ISO-format strings
    and compared using :func:`datetime.datetime.fromisoformat`.

    Parameters
    ----------
    storage_dir:
        Directory in which ``scheduler.json`` is kept.  Defaults to
        ``~/.agent_friend/``.  Pass a custom path (e.g. a ``tmp_path``
        fixture) when testing.
    """

    def __init__(
        self,
        storage_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if storage_dir is None:
            storage_dir = _DEFAULT_STORAGE_DIR
        self._storage_dir = Path(storage_dir)
        self._schedule_file = self._storage_dir / "scheduler.json"
        self._tasks: List[Dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "scheduler"

    @property
    def description(self) -> str:
        return (
            "Schedule agent tasks to run periodically (every N minutes) or "
            "at a specific time. Persists state between runs."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        """Return LLM tool definitions for the scheduler."""
        return [
            {
                "name": "schedule_task",
                "description": (
                    "Create or update a scheduled task. Provide either "
                    "interval_minutes (recurring) or run_at (one-shot ISO datetime), "
                    "but not both."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Unique identifier for the task.",
                        },
                        "prompt": {
                            "type": "string",
                            "description": "The instruction/prompt to run when the task fires.",
                        },
                        "interval_minutes": {
                            "type": "number",
                            "description": "Run every N minutes (recurring). Must be > 0.",
                        },
                        "run_at": {
                            "type": "string",
                            "description": (
                                "ISO datetime string 'YYYY-MM-DDTHH:MM:SS' for a "
                                "one-shot task."
                            ),
                        },
                    },
                    "required": ["task_id", "prompt"],
                },
            },
            {
                "name": "list_scheduled",
                "description": "List all scheduled tasks.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "cancel_task",
                "description": "Remove a scheduled task by id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "task_id": {
                            "type": "string",
                            "description": "Id of the task to remove.",
                        },
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "run_pending",
                "description": (
                    "Check for tasks that are due and return them. "
                    "Recurring tasks are rescheduled; one-shot tasks are removed."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "clear_all",
                "description": "Remove all scheduled tasks.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call from the LLM to the appropriate method."""
        if tool_name == "schedule_task":
            task_id = arguments.get("task_id", "").strip()
            prompt = arguments.get("prompt", "").strip()
            interval_minutes = arguments.get("interval_minutes")
            run_at = arguments.get("run_at")
            if not task_id or not prompt:
                return "Error: task_id and prompt are required."
            try:
                task = self.schedule(
                    task_id,
                    prompt,
                    interval_minutes=interval_minutes,
                    run_at=run_at,
                )
            except ValueError as exc:
                return f"Error: {exc}"
            return f"Scheduled task '{task_id}'. Next run: {task['next_run']}"

        if tool_name == "list_scheduled":
            tasks = self.list_scheduled()
            if not tasks:
                return "No scheduled tasks."
            lines = [f"Scheduled tasks ({len(tasks)}):"]
            for t in tasks:
                interval_info = (
                    f"every {t['interval_minutes']} min"
                    if t["interval_minutes"] is not None
                    else f"one-shot at {t['run_at']}"
                )
                lines.append(
                    f"  {t['id']}: {t['prompt'][:60]} [{interval_info}] "
                    f"next={t['next_run']}"
                )
            return "\n".join(lines)

        if tool_name == "cancel_task":
            task_id = arguments.get("task_id", "").strip()
            removed = self.cancel(task_id)
            if removed:
                return f"Task '{task_id}' cancelled."
            return f"No task with id '{task_id}' found."

        if tool_name == "run_pending":
            due = self.run_pending()
            if not due:
                return "No tasks are due."
            lines = [f"{len(due)} task(s) due:"]
            for t in due:
                lines.append(f"  {t['id']}: {t['prompt']}")
            return "\n".join(lines)

        if tool_name == "clear_all":
            count = self.clear_all()
            return f"Cleared {count} task(s)."

        return f"Unknown scheduler tool: {tool_name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(
        self,
        task_id: str,
        prompt: str,
        interval_minutes: Optional[float] = None,
        run_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create or update a scheduled task.

        Parameters
        ----------
        task_id:
            Unique identifier.  If a task with this id already exists it is
            replaced.
        prompt:
            The instruction string the agent should run when the task fires.
        interval_minutes:
            Recurring schedule — run every *N* minutes.  Must be > 0.
            Mutually exclusive with *run_at*.
        run_at:
            One-shot schedule — ISO datetime string ``"YYYY-MM-DDTHH:MM:SS"``.
            Mutually exclusive with *interval_minutes*.

        Returns
        -------
        dict
            The newly created/updated task dict.

        Raises
        ------
        ValueError
            If neither or both of *interval_minutes* / *run_at* are supplied,
            or if *interval_minutes* is not positive.
        """
        if interval_minutes is None and run_at is None:
            raise ValueError("Provide exactly one of interval_minutes or run_at.")
        if interval_minutes is not None and run_at is not None:
            raise ValueError(
                "Provide exactly one of interval_minutes or run_at, not both."
            )
        if interval_minutes is not None and interval_minutes <= 0:
            raise ValueError("interval_minutes must be > 0.")

        now = datetime.datetime.now()
        now_iso = now.isoformat(timespec="seconds")

        if interval_minutes is not None:
            next_run = (
                now + datetime.timedelta(minutes=interval_minutes)
            ).isoformat(timespec="seconds")
        else:
            # Validate the run_at string
            _parse_iso(run_at)  # raises ValueError on bad format
            next_run = run_at

        task: Dict[str, Any] = {
            "id": task_id,
            "prompt": prompt,
            "interval_minutes": interval_minutes,
            "run_at": run_at,
            "last_run": None,
            "next_run": next_run,
            "created": now_iso,
        }

        # Replace existing task with the same id, or append.
        self._tasks = [t for t in self._tasks if t["id"] != task_id]
        self._tasks.append(task)
        self._save()
        return task

    def list_scheduled(self) -> List[Dict[str, Any]]:
        """Return a copy of all scheduled task dicts.

        Returns
        -------
        list of dict
            Each dict contains: id, prompt, interval_minutes, run_at,
            last_run, next_run, created.
        """
        return [dict(t) for t in self._tasks]

    def cancel(self, task_id: str) -> bool:
        """Remove a task by id.

        Parameters
        ----------
        task_id:
            The id of the task to remove.

        Returns
        -------
        bool
            ``True`` if the task was found and removed, ``False`` otherwise.
        """
        before = len(self._tasks)
        self._tasks = [t for t in self._tasks if t["id"] != task_id]
        if len(self._tasks) < before:
            self._save()
            return True
        return False

    def run_pending(self) -> List[Dict[str, Any]]:
        """Return and process all tasks whose next_run is <= now.

        - Recurring tasks: ``next_run`` is advanced by one interval and
          ``last_run`` is updated.
        - One-shot tasks: removed from the schedule after being returned.

        Returns
        -------
        list of dict
            Task dicts that were due.  (Snapshot taken before updating state.)
        """
        now = datetime.datetime.now()
        due: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []

        for task in self._tasks:
            next_run = _parse_iso(task["next_run"])
            if next_run <= now:
                due.append(dict(task))
                if task["interval_minutes"] is not None:
                    # Recurring — advance by exactly one interval.
                    new_next = (
                        now + datetime.timedelta(minutes=task["interval_minutes"])
                    ).isoformat(timespec="seconds")
                    task["last_run"] = now.isoformat(timespec="seconds")
                    task["next_run"] = new_next
                    remaining.append(task)
                # else: one-shot — drop it (do not append to remaining)
            else:
                remaining.append(task)

        self._tasks = remaining
        if due:
            self._save()
        return due

    def clear_all(self) -> int:
        """Remove all scheduled tasks.

        Returns
        -------
        int
            The number of tasks that were removed.
        """
        count = len(self._tasks)
        self._tasks = []
        self._save()
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load tasks from disk, creating the file/directory if absent."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        if not self._schedule_file.exists():
            self._tasks = []
            return
        try:
            with open(self._schedule_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                self._tasks = data
            else:
                self._tasks = []
        except (json.JSONDecodeError, OSError):
            self._tasks = []

    def _save(self) -> None:
        """Persist the current task list to disk."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        with open(self._schedule_file, "w", encoding="utf-8") as fh:
            json.dump(self._tasks, fh, indent=2)
