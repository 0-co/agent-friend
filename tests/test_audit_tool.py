"""Tests for AuditTool — structured audit log."""

import json
import time
import pytest
from agent_friend.tools.audit_tool import AuditTool


@pytest.fixture
def tool():
    return AuditTool()


@pytest.fixture
def populated(tool):
    """Tool with a handful of pre-logged events."""
    tool.audit_log("user.login",  actor="alice", resource="auth",   severity="info",    outcome="success", metadata={"ip": "1.1.1.1"})
    tool.audit_log("file.delete", actor="bob",   resource="doc.txt", severity="warning", outcome="success")
    tool.audit_log("api.call",    actor="alice", resource="/v1/data", severity="info",    outcome="success")
    tool.audit_log("user.login",  actor="eve",   resource="auth",   severity="error",   outcome="failure",  metadata={"ip": "9.9.9.9"})
    return tool


# ── audit_log ────────────────────────────────────────────────────────────────

def test_log_basic(tool):
    r = json.loads(tool.audit_log("user.login"))
    assert "id" in r
    assert r["type"] == "user.login"
    assert "timestamp" in r


def test_log_with_all_fields(tool):
    r = json.loads(tool.audit_log(
        "db.query", actor="svc", resource="users",
        metadata={"rows": 5}, severity="info", outcome="success"
    ))
    assert "id" in r
    assert r["actor"] == "svc"


def test_log_empty_type_error(tool):
    r = json.loads(tool.audit_log(""))
    assert "error" in r


def test_log_invalid_severity(tool):
    r = json.loads(tool.audit_log("x", severity="mega"))
    assert "error" in r


def test_log_invalid_outcome(tool):
    r = json.loads(tool.audit_log("x", outcome="maybe"))
    assert "error" in r


def test_log_all_severities(tool):
    for sev in ["info", "warning", "error", "critical"]:
        r = json.loads(tool.audit_log(f"ev.{sev}", severity=sev))
        assert "error" not in r


def test_log_all_outcomes(tool):
    for out in ["success", "failure", "denied", "unknown"]:
        r = json.loads(tool.audit_log(f"ev.{out}", outcome=out))
        assert "error" not in r


def test_log_returns_unique_ids(tool):
    r1 = json.loads(tool.audit_log("x"))
    r2 = json.loads(tool.audit_log("x"))
    assert r1["id"] != r2["id"]


# ── audit_search ─────────────────────────────────────────────────────────────

def test_search_all(populated):
    r = json.loads(populated.audit_search())
    assert r["total"] == 4


def test_search_by_type(populated):
    r = json.loads(populated.audit_search(event_type="user.login"))
    assert r["total"] == 2
    for e in r["events"]:
        assert e["type"] == "user.login"


def test_search_by_actor(populated):
    r = json.loads(populated.audit_search(actor="alice"))
    assert r["total"] == 2


def test_search_by_resource(populated):
    r = json.loads(populated.audit_search(resource="auth"))
    assert r["total"] == 2


def test_search_by_severity(populated):
    r = json.loads(populated.audit_search(severity="error"))
    assert r["total"] == 1


def test_search_by_outcome(populated):
    r = json.loads(populated.audit_search(outcome="failure"))
    assert r["total"] == 1


def test_search_by_text(populated):
    r = json.loads(populated.audit_search(text="9.9.9.9"))
    assert r["total"] == 1


def test_search_combined_filters(populated):
    r = json.loads(populated.audit_search(actor="alice", event_type="user.login"))
    assert r["total"] == 1


def test_search_after_filter(tool):
    tool.audit_log("old", actor="a")
    ts_mid = time.time()
    time.sleep(0.01)
    tool.audit_log("new", actor="b")
    r = json.loads(tool.audit_search(after=ts_mid))
    assert r["total"] == 1
    assert r["events"][0]["type"] == "new"


def test_search_before_filter(tool):
    tool.audit_log("first", actor="a")
    ts_mid = time.time()
    time.sleep(0.01)
    tool.audit_log("second", actor="b")
    r = json.loads(tool.audit_search(before=ts_mid))
    assert r["total"] == 1
    assert r["events"][0]["type"] == "first"


def test_search_limit(populated):
    r = json.loads(populated.audit_search(limit=2))
    assert r["count"] == 2
    assert r["total"] == 4


def test_search_offset(populated):
    r = json.loads(populated.audit_search(limit=2, offset=2))
    assert r["count"] == 2
    assert r["offset"] == 2


def test_search_no_match(populated):
    r = json.loads(populated.audit_search(actor="nobody"))
    assert r["total"] == 0
    assert r["events"] == []


def test_search_newest_first(tool):
    tool.audit_log("ev1")
    time.sleep(0.01)
    tool.audit_log("ev2")
    r = json.loads(tool.audit_search())
    assert r["events"][0]["type"] == "ev2"


# ── audit_get ────────────────────────────────────────────────────────────────

def test_get_by_id(tool):
    logged = json.loads(tool.audit_log("db.read", actor="svc"))
    eid = logged["id"]
    r = json.loads(tool.audit_get(eid))
    assert r["id"] == eid
    assert r["type"] == "db.read"


def test_get_not_found(tool):
    r = json.loads(tool.audit_get("00000000-0000-0000-0000-000000000000"))
    assert "error" in r


def test_get_has_metadata(tool):
    tool.audit_log("ev", metadata={"key": "val"})
    eid = json.loads(tool.audit_search())["events"][0]["id"]
    r = json.loads(tool.audit_get(eid))
    assert r["metadata"]["key"] == "val"


# ── audit_stats ───────────────────────────────────────────────────────────────

def test_stats_empty(tool):
    r = json.loads(tool.audit_stats())
    assert r["total"] == 0


def test_stats_by_type(populated):
    r = json.loads(populated.audit_stats())
    assert r["by_type"]["user.login"] == 2
    assert r["by_type"]["file.delete"] == 1


def test_stats_by_actor(populated):
    r = json.loads(populated.audit_stats())
    assert r["by_actor"]["alice"] == 2
    assert r["by_actor"]["bob"] == 1


def test_stats_by_severity(populated):
    r = json.loads(populated.audit_stats())
    assert r["by_severity"]["info"] == 2
    assert r["by_severity"]["error"] == 1


def test_stats_by_outcome(populated):
    r = json.loads(populated.audit_stats())
    assert r["by_outcome"]["success"] == 3
    assert r["by_outcome"]["failure"] == 1


def test_stats_with_time_filter(tool):
    tool.audit_log("old")
    ts = time.time()
    time.sleep(0.01)
    tool.audit_log("new")
    r = json.loads(tool.audit_stats(after=ts))
    assert r["total"] == 1


# ── audit_export ──────────────────────────────────────────────────────────────

def test_export_all(populated):
    r = json.loads(populated.audit_export())
    assert r["count"] == 4
    lines = r["lines"].split("\n")
    assert len(lines) == 4
    # Each line must be valid JSON
    for line in lines:
        obj = json.loads(line)
        assert "id" in obj


def test_export_filtered_type(populated):
    r = json.loads(populated.audit_export(event_type="user.login"))
    assert r["count"] == 2


def test_export_empty(tool):
    r = json.loads(tool.audit_export())
    assert r["count"] == 0
    assert r["lines"] == ""


# ── audit_clear ───────────────────────────────────────────────────────────────

def test_clear_all(populated):
    r = json.loads(populated.audit_clear())
    assert r["cleared"] == 4
    assert r["remaining"] == 0


def test_clear_before_ts(tool):
    tool.audit_log("old")
    ts = time.time()
    time.sleep(0.01)
    tool.audit_log("new")
    r = json.loads(tool.audit_clear(before=ts))
    assert r["cleared"] == 1
    assert r["remaining"] == 1


def test_clear_leaves_newer(tool):
    tool.audit_log("keep")
    ts = time.time()
    time.sleep(0.01)
    tool.audit_log("also_keep")
    tool.audit_clear(before=ts - 1)  # nothing older than 1s before ts
    r = json.loads(tool.audit_search())
    assert r["total"] == 2


# ── audit_types ───────────────────────────────────────────────────────────────

def test_types_empty(tool):
    r = json.loads(tool.audit_types())
    assert r["count"] == 0
    assert r["types"] == []


def test_types_distinct(populated):
    r = json.loads(populated.audit_types())
    assert r["count"] == 3
    assert "user.login" in r["types"]
    assert "file.delete" in r["types"]


# ── audit_timeline ────────────────────────────────────────────────────────────

def test_timeline_empty(tool):
    r = json.loads(tool.audit_timeline())
    assert r["total"] == 0
    assert r["buckets"] == []


def test_timeline_hour(tool):
    tool.audit_log("ev1")
    tool.audit_log("ev2")
    r = json.loads(tool.audit_timeline(bucket="hour"))
    assert r["total"] == 2
    # Both in same hour bucket
    assert len(r["buckets"]) == 1
    assert r["buckets"][0]["count"] == 2


def test_timeline_day(tool):
    tool.audit_log("ev1")
    r = json.loads(tool.audit_timeline(bucket="day"))
    assert r["total"] == 1
    assert r["buckets"][0]["count"] == 1


def test_timeline_invalid_bucket(tool):
    r = json.loads(tool.audit_timeline(bucket="minute"))
    assert "error" in r


# ── execute dispatch ──────────────────────────────────────────────────────────

def test_execute_log(tool):
    r = json.loads(tool.execute("audit_log", {"event_type": "test.ev"}))
    assert "id" in r


def test_execute_search(tool):
    r = json.loads(tool.execute("audit_search", {}))
    assert "events" in r


def test_execute_stats(tool):
    r = json.loads(tool.execute("audit_stats", {}))
    assert "total" in r


def test_execute_types(tool):
    r = json.loads(tool.execute("audit_types", {}))
    assert "types" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ─────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "audit"


def test_description(tool):
    assert "audit" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
