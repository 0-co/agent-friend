"""Tests for SchedulerTool — task scheduler with persistence."""

import datetime
import json
import os
import sys
import unittest

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.tools.scheduler import SchedulerTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _past(minutes: int = 5) -> str:
    """Return an ISO datetime string N minutes in the past."""
    dt = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
    return dt.isoformat(timespec="seconds")


def _future(minutes: int = 60) -> str:
    """Return an ISO datetime string N minutes in the future."""
    dt = datetime.datetime.now() + datetime.timedelta(minutes=minutes)
    return dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# schedule() — basic creation
# ---------------------------------------------------------------------------

class TestScheduleCreatesTask:
    def test_schedule_creates_task(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        task = tool.schedule("my_task", "Do something", interval_minutes=30)
        assert task["id"] == "my_task"
        assert task["prompt"] == "Do something"

    def test_schedule_with_interval(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        before = datetime.datetime.now().replace(microsecond=0)
        task = tool.schedule("t", "p", interval_minutes=10)
        after = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(seconds=1)
        next_run = datetime.datetime.fromisoformat(task["next_run"])
        assert next_run >= before + datetime.timedelta(minutes=10)
        assert next_run <= after + datetime.timedelta(minutes=10)

    def test_schedule_with_run_at(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        run_at = "2026-03-13T08:00:00"
        task = tool.schedule("t", "p", run_at=run_at)
        assert task["next_run"] == run_at

    def test_schedule_raises_without_schedule_type(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        with pytest.raises(ValueError):
            tool.schedule("t", "p")

    def test_schedule_raises_with_both(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        with pytest.raises(ValueError):
            tool.schedule("t", "p", interval_minutes=10, run_at="2026-03-13T08:00:00")

    def test_schedule_interval_must_be_positive(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        with pytest.raises(ValueError):
            tool.schedule("t", "p", interval_minutes=0)
        with pytest.raises(ValueError):
            tool.schedule("t", "p", interval_minutes=-5)

    def test_task_dict_fields(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        task = tool.schedule("t", "p", interval_minutes=60)
        for field in ("id", "prompt", "interval_minutes", "run_at", "last_run", "next_run", "created"):
            assert field in task

    def test_schedule_default_id_format(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        task = tool.schedule("my-task-id", "prompt", interval_minutes=1)
        assert task["id"] == "my-task-id"

    def test_schedule_overwrites_existing_id(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "original", interval_minutes=30)
        task = tool.schedule("t", "updated", interval_minutes=60)
        tasks = tool.list_scheduled()
        assert len(tasks) == 1
        assert tasks[0]["prompt"] == "updated"
        assert tasks[0]["interval_minutes"] == 60

    def test_interval_minutes_as_float_works(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        before = datetime.datetime.now().replace(microsecond=0)
        task = tool.schedule("t", "p", interval_minutes=0.5)
        after = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(seconds=1)
        next_run = datetime.datetime.fromisoformat(task["next_run"])
        assert next_run >= before + datetime.timedelta(seconds=30)
        assert next_run <= after + datetime.timedelta(seconds=30)

    def test_run_at_in_past_is_due(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", run_at=_past(10))
        due = tool.run_pending()
        assert len(due) == 1
        assert due[0]["id"] == "t"


# ---------------------------------------------------------------------------
# list_scheduled()
# ---------------------------------------------------------------------------

class TestListScheduled:
    def test_list_scheduled_empty(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        assert tool.list_scheduled() == []

    def test_list_scheduled_returns_tasks(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("a", "pa", interval_minutes=10)
        tool.schedule("b", "pb", run_at=_future())
        tasks = tool.list_scheduled()
        ids = {t["id"] for t in tasks}
        assert "a" in ids
        assert "b" in ids

    def test_list_returns_copy_not_reference(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=5)
        tasks = tool.list_scheduled()
        tasks.clear()
        assert len(tool.list_scheduled()) == 1


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------

class TestCancel:
    def test_cancel_existing_task(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=10)
        result = tool.cancel("t")
        assert result is True
        assert tool.list_scheduled() == []

    def test_cancel_nonexistent_task(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.cancel("does_not_exist")
        assert result is False

    def test_cancel_saves_state(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=10)
        tool.cancel("t")
        tool2 = SchedulerTool(storage_dir=tmp_path)
        assert tool2.list_scheduled() == []


# ---------------------------------------------------------------------------
# run_pending()
# ---------------------------------------------------------------------------

class TestRunPending:
    def test_run_pending_empty(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        assert tool.run_pending() == []

    def test_run_pending_no_due(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", run_at=_future(60))
        assert tool.run_pending() == []

    def test_run_pending_returns_due_task(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", run_at=_past(5))
        due = tool.run_pending()
        assert len(due) == 1
        assert due[0]["id"] == "t"

    def test_run_pending_removes_one_shot(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", run_at=_past(5))
        tool.run_pending()
        assert tool.list_scheduled() == []

    def test_run_pending_updates_recurring(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=30)
        # Force next_run into the past
        tool._tasks[0]["next_run"] = _past(5)
        before = datetime.datetime.now().replace(microsecond=0)
        tool.run_pending()
        after = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(seconds=1)
        remaining = tool.list_scheduled()
        assert len(remaining) == 1
        new_next = datetime.datetime.fromisoformat(remaining[0]["next_run"])
        assert new_next >= before + datetime.timedelta(minutes=30)
        assert new_next <= after + datetime.timedelta(minutes=30)

    def test_run_pending_updates_last_run(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=10)
        tool._tasks[0]["next_run"] = _past(2)
        before = datetime.datetime.now().replace(microsecond=0)
        tool.run_pending()
        after = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(seconds=1)
        remaining = tool.list_scheduled()
        last_run = datetime.datetime.fromisoformat(remaining[0]["last_run"])
        assert before <= last_run <= after

    def test_run_pending_handles_multiple(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("a", "pa", run_at=_past(10))
        tool.schedule("b", "pb", run_at=_past(5))
        tool.schedule("c", "pc", run_at=_future(60))
        due = tool.run_pending()
        due_ids = {t["id"] for t in due}
        assert due_ids == {"a", "b"}

    def test_run_pending_multiple_intervals(self, tmp_path):
        """Very overdue task advances by exactly one interval, not multiple."""
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=30)
        # Simulate task being overdue by 3 intervals
        tool._tasks[0]["next_run"] = _past(90)
        before = datetime.datetime.now().replace(microsecond=0)
        tool.run_pending()
        after = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(seconds=1)
        remaining = tool.list_scheduled()
        new_next = datetime.datetime.fromisoformat(remaining[0]["next_run"])
        # Should be ~now+30min, not now+30+90
        assert new_next >= before + datetime.timedelta(minutes=30)
        assert new_next <= after + datetime.timedelta(minutes=30)

    def test_run_pending_saves_state(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=10)
        tool._tasks[0]["next_run"] = _past(2)
        tool.run_pending()
        tool2 = SchedulerTool(storage_dir=tmp_path)
        tasks = tool2.list_scheduled()
        assert len(tasks) == 1
        assert tasks[0]["last_run"] is not None


# ---------------------------------------------------------------------------
# clear_all()
# ---------------------------------------------------------------------------

class TestClearAll:
    def test_clear_all_removes_all(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("a", "pa", interval_minutes=10)
        tool.schedule("b", "pb", run_at=_future())
        count = tool.clear_all()
        assert count == 2
        assert tool.list_scheduled() == []

    def test_clear_all_empty(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        count = tool.clear_all()
        assert count == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_persistence(self, tmp_path):
        """Tasks survive across SchedulerTool instances."""
        tool1 = SchedulerTool(storage_dir=tmp_path)
        tool1.schedule("persistent", "remember me", interval_minutes=60)
        tool2 = SchedulerTool(storage_dir=tmp_path)
        tasks = tool2.list_scheduled()
        assert len(tasks) == 1
        assert tasks[0]["id"] == "persistent"
        assert tasks[0]["prompt"] == "remember me"

    def test_schedule_creates_parent_dir(self, tmp_path):
        """SchedulerTool creates ~/.agent_friend/ equivalent if it doesn't exist."""
        nested = tmp_path / "nonexistent" / "subdir"
        assert not nested.exists()
        tool = SchedulerTool(storage_dir=nested)
        assert nested.exists()

    def test_json_file_written(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=5)
        schedule_file = tmp_path / "scheduler.json"
        assert schedule_file.exists()
        with open(schedule_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["id"] == "t"


# ---------------------------------------------------------------------------
# execute() / LLM dispatch
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def test_execute_schedule_task(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute(
            "schedule_task",
            {"task_id": "x", "prompt": "do stuff", "interval_minutes": 60},
        )
        assert "x" in result
        assert "next_run" in result.lower() or "Next run" in result

    def test_execute_schedule_task_missing_fields(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("schedule_task", {})
        assert "Error" in result

    def test_execute_schedule_task_no_type(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("schedule_task", {"task_id": "x", "prompt": "p"})
        assert "Error" in result

    def test_execute_list_empty(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("list_scheduled", {})
        assert "No" in result

    def test_execute_list_with_tasks(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=5)
        result = tool.execute("list_scheduled", {})
        assert "t" in result

    def test_execute_cancel_existing(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("t", "p", interval_minutes=5)
        result = tool.execute("cancel_task", {"task_id": "t"})
        assert "cancelled" in result.lower() or "t" in result

    def test_execute_cancel_nonexistent(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("cancel_task", {"task_id": "nope"})
        assert "No" in result or "not found" in result.lower()

    def test_execute_run_pending_none_due(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("run_pending", {})
        assert "No" in result

    def test_execute_clear_all(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        tool.schedule("a", "pa", interval_minutes=1)
        tool.schedule("b", "pb", interval_minutes=2)
        result = tool.execute("clear_all", {})
        assert "2" in result

    def test_execute_unknown_tool(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        result = tool.execute("nonexistent_method", {})
        assert "Unknown" in result

    def test_name_property(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        assert tool.name == "scheduler"

    def test_description_nonempty(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        assert len(tool.description) > 0

    def test_definitions_count(self, tmp_path):
        tool = SchedulerTool(storage_dir=tmp_path)
        assert len(tool.definitions()) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
