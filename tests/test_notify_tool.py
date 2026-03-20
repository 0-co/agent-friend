"""Tests for NotifyTool."""

import json
import os
import pytest

from agent_friend.tools.notify import NotifyTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def notifier(tmp_path):
    """Return a NotifyTool backed by a temp directory."""
    return NotifyTool(log_path=str(tmp_path / "notifications.log"))


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(notifier):
    assert notifier.name == "notify"


def test_description(notifier):
    assert "notif" in notifier.description.lower()


def test_definitions_returns_five(notifier):
    defs = notifier.definitions()
    assert len(defs) == 5


def test_definition_names(notifier):
    names = {d["name"] for d in notifier.definitions()}
    assert names == {"notify", "notify_desktop", "notify_file", "bell", "read_notifications"}


# ── notify (auto) ─────────────────────────────────────────────────────────────


def test_notify_returns_string(notifier):
    result = notifier.notify("Title", "Body")
    assert isinstance(result, str)
    assert len(result) > 0


def test_notify_logs_to_file(notifier):
    notifier.notify("Task done", "All good")
    entries = json.loads(notifier.read_notifications())
    assert len(entries) >= 1
    last = entries[-1]
    assert last["title"] == "Task done"
    assert last["message"] == "All good"


# ── notify_file ───────────────────────────────────────────────────────────────


def test_notify_file_creates_log(notifier):
    result = notifier.notify_file("Done", "Finished successfully")
    assert "notification" in result.lower()
    assert os.path.exists(notifier.log_path)


def test_notify_file_appends(notifier):
    notifier.notify_file("A", "First")
    notifier.notify_file("B", "Second")
    entries = json.loads(notifier.read_notifications())
    assert len(entries) == 2
    assert entries[0]["title"] == "A"
    assert entries[1]["title"] == "B"


def test_notify_file_custom_path(notifier, tmp_path):
    custom = str(tmp_path / "custom.log")
    notifier.notify_file("Test", "Custom path test", path=custom)
    assert os.path.exists(custom)
    with open(custom) as f:
        entry = json.loads(f.read().strip())
    assert entry["title"] == "Test"


def test_notify_file_entry_has_timestamp(notifier):
    notifier.notify_file("Ts", "Check timestamp")
    entries = json.loads(notifier.read_notifications())
    assert "timestamp" in entries[-1]
    assert "T" in entries[-1]["timestamp"]  # ISO format


# ── notify_desktop ────────────────────────────────────────────────────────────


def test_notify_desktop_returns_string(notifier):
    # May fail gracefully (no display in CI) — should not raise
    result = notifier.notify_desktop("Test", "Message")
    assert isinstance(result, str)


# ── bell ──────────────────────────────────────────────────────────────────────


def test_bell_returns_string(notifier):
    result = notifier.bell()
    assert isinstance(result, str)


# ── read_notifications ────────────────────────────────────────────────────────


def test_read_notifications_empty(notifier):
    result = notifier.read_notifications()
    assert result == "[]"


def test_read_notifications_limits_n(notifier):
    for i in range(15):
        notifier.notify_file(f"Title {i}", f"Body {i}")
    entries = json.loads(notifier.read_notifications(n=5))
    assert len(entries) == 5
    # Should be last 5
    assert entries[-1]["title"] == "Title 14"


def test_read_notifications_default_ten(notifier):
    for i in range(12):
        notifier.notify_file(f"T{i}", "m")
    entries = json.loads(notifier.read_notifications())
    assert len(entries) == 10


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_notify(notifier):
    result = notifier.execute("notify", {"title": "Done", "message": "Finished"})
    assert isinstance(result, str)


def test_execute_notify_file(notifier):
    result = notifier.execute("notify_file", {"title": "Log", "message": "Entry"})
    assert "notification" in result.lower()


def test_execute_bell(notifier):
    result = notifier.execute("bell", {})
    assert isinstance(result, str)


def test_execute_read_notifications(notifier):
    notifier.notify_file("A", "B")
    result = notifier.execute("read_notifications", {"n": 5})
    entries = json.loads(result)
    assert len(entries) == 1


def test_execute_unknown_tool(notifier):
    result = notifier.execute("nonexistent", {})
    assert "Unknown" in result


# ── tool definition schemas ───────────────────────────────────────────────────


def test_notify_schema(notifier):
    defs = {d["name"]: d for d in notifier.definitions()}
    schema = defs["notify"]["input_schema"]
    assert set(schema["required"]) == {"title", "message"}


def test_notify_file_schema_path_optional(notifier):
    defs = {d["name"]: d for d in notifier.definitions()}
    schema = defs["notify_file"]["input_schema"]
    assert "path" in schema["properties"]
    assert "path" not in schema["required"]


def test_bell_schema_no_required(notifier):
    defs = {d["name"]: d for d in notifier.definitions()}
    schema = defs["bell"]["input_schema"]
    assert schema.get("required", []) == []
