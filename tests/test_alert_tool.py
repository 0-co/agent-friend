"""Tests for AlertTool — threshold-based alerting."""

import json
import time
import pytest
from agent_friend.tools.alert_tool import AlertTool


@pytest.fixture
def tool():
    return AlertTool()


# ── alert_define ────────────────────────────────────────────────────────────

def test_define_basic(tool):
    r = json.loads(tool.alert_define("high_cpu", condition="gt", threshold=90))
    assert r["name"] == "high_cpu"
    assert r["condition"] == "gt"
    assert r["severity"] == "warning"


def test_define_all_conditions(tool):
    conditions = ["gt", "gte", "lt", "lte", "eq", "ne", "contains", "not_contains", "is_empty", "is_truthy"]
    for i, cond in enumerate(conditions):
        r = json.loads(tool.alert_define(f"r{i}", condition=cond, threshold=0))
        assert "error" not in r


def test_define_between(tool):
    r = json.loads(tool.alert_define("mid", condition="between", threshold=10, threshold_high=20))
    assert "error" not in r


def test_define_outside(tool):
    r = json.loads(tool.alert_define("out", condition="outside", threshold=0, threshold_high=100))
    assert "error" not in r


def test_define_between_missing_high(tool):
    r = json.loads(tool.alert_define("x", condition="between", threshold=10))
    assert "error" in r


def test_define_invalid_condition(tool):
    r = json.loads(tool.alert_define("x", condition="bogus"))
    assert "error" in r


def test_define_invalid_severity(tool):
    r = json.loads(tool.alert_define("x", condition="gt", threshold=1, severity="extreme"))
    assert "error" in r


def test_define_empty_name(tool):
    r = json.loads(tool.alert_define("", condition="gt", threshold=1))
    assert "error" in r


def test_define_all_severities(tool):
    for sev in ["info", "warning", "error", "critical"]:
        r = json.loads(tool.alert_define(f"r_{sev}", condition="gt", threshold=0, severity=sev))
        assert "error" not in r


def test_define_overwrite(tool):
    tool.alert_define("r1", condition="gt", threshold=10)
    r = json.loads(tool.alert_define("r1", condition="lt", threshold=5))
    assert "error" not in r


def test_define_with_cooldown(tool):
    r = json.loads(tool.alert_define("cool", condition="gt", threshold=0, cooldown_s=5.0))
    assert "error" not in r


# ── alert_evaluate — numeric conditions ────────────────────────────────────

def test_evaluate_gt_fires(tool):
    tool.alert_define("cpu", condition="gt", threshold=90)
    r = json.loads(tool.alert_evaluate("cpu", 95))
    assert r["fired"] is True
    assert r["rule"] == "cpu"


def test_evaluate_gt_nofires(tool):
    tool.alert_define("cpu", condition="gt", threshold=90)
    r = json.loads(tool.alert_evaluate("cpu", 85))
    assert r["fired"] is False


def test_evaluate_gte_fires_equal(tool):
    tool.alert_define("r", condition="gte", threshold=10)
    r = json.loads(tool.alert_evaluate("r", 10))
    assert r["fired"] is True


def test_evaluate_lt_fires(tool):
    tool.alert_define("r", condition="lt", threshold=5)
    r = json.loads(tool.alert_evaluate("r", 3))
    assert r["fired"] is True


def test_evaluate_lte_fires_equal(tool):
    tool.alert_define("r", condition="lte", threshold=5)
    r = json.loads(tool.alert_evaluate("r", 5))
    assert r["fired"] is True


def test_evaluate_eq_fires(tool):
    tool.alert_define("r", condition="eq", threshold="error")
    r = json.loads(tool.alert_evaluate("r", "error"))
    assert r["fired"] is True


def test_evaluate_ne_fires(tool):
    tool.alert_define("r", condition="ne", threshold="ok")
    r = json.loads(tool.alert_evaluate("r", "not_ok"))
    assert r["fired"] is True


def test_evaluate_ne_nofires(tool):
    tool.alert_define("r", condition="ne", threshold="ok")
    r = json.loads(tool.alert_evaluate("r", "ok"))
    assert r["fired"] is False


def test_evaluate_between_fires(tool):
    tool.alert_define("r", condition="between", threshold=10, threshold_high=20)
    r = json.loads(tool.alert_evaluate("r", 15))
    assert r["fired"] is True


def test_evaluate_between_nofires(tool):
    tool.alert_define("r", condition="between", threshold=10, threshold_high=20)
    r = json.loads(tool.alert_evaluate("r", 25))
    assert r["fired"] is False


def test_evaluate_outside_fires(tool):
    tool.alert_define("r", condition="outside", threshold=0, threshold_high=100)
    r = json.loads(tool.alert_evaluate("r", 150))
    assert r["fired"] is True


def test_evaluate_outside_nofires(tool):
    tool.alert_define("r", condition="outside", threshold=0, threshold_high=100)
    r = json.loads(tool.alert_evaluate("r", 50))
    assert r["fired"] is False


def test_evaluate_contains_fires(tool):
    tool.alert_define("r", condition="contains", threshold="error")
    r = json.loads(tool.alert_evaluate("r", "some error happened"))
    assert r["fired"] is True


def test_evaluate_contains_nofires(tool):
    tool.alert_define("r", condition="contains", threshold="error")
    r = json.loads(tool.alert_evaluate("r", "all clear"))
    assert r["fired"] is False


def test_evaluate_not_contains_fires(tool):
    tool.alert_define("r", condition="not_contains", threshold="ok")
    r = json.loads(tool.alert_evaluate("r", "failure"))
    assert r["fired"] is True


def test_evaluate_is_empty_fires(tool):
    tool.alert_define("r", condition="is_empty")
    r = json.loads(tool.alert_evaluate("r", []))
    assert r["fired"] is True


def test_evaluate_is_empty_nofires(tool):
    tool.alert_define("r", condition="is_empty")
    r = json.loads(tool.alert_evaluate("r", [1, 2]))
    assert r["fired"] is False


def test_evaluate_is_truthy_fires(tool):
    tool.alert_define("r", condition="is_truthy")
    r = json.loads(tool.alert_evaluate("r", 42))
    assert r["fired"] is True


def test_evaluate_is_truthy_nofires(tool):
    tool.alert_define("r", condition="is_truthy")
    r = json.loads(tool.alert_evaluate("r", 0))
    assert r["fired"] is False


def test_evaluate_not_found(tool):
    r = json.loads(tool.alert_evaluate("ghost", 10))
    assert "error" in r


def test_evaluate_returns_timestamp(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    r = json.loads(tool.alert_evaluate("r", 1))
    assert "timestamp" in r
    assert r["timestamp"] > 0


def test_evaluate_severity_on_fire(tool):
    tool.alert_define("r", condition="gt", threshold=0, severity="critical")
    r = json.loads(tool.alert_evaluate("r", 1))
    assert r["severity"] == "critical"


def test_evaluate_severity_none_when_not_fired(tool):
    tool.alert_define("r", condition="gt", threshold=100)
    r = json.loads(tool.alert_evaluate("r", 0))
    assert r["severity"] is None


def test_evaluate_with_metadata(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    r = json.loads(tool.alert_evaluate("r", 1, metadata={"host": "web01"}))
    assert r["metadata"]["host"] == "web01"


# ── cooldown ────────────────────────────────────────────────────────────────

def test_cooldown_suppresses_repeat(tool):
    tool.alert_define("r", condition="gt", threshold=0, cooldown_s=60)
    r1 = json.loads(tool.alert_evaluate("r", 1))
    r2 = json.loads(tool.alert_evaluate("r", 1))
    assert r1["fired"] is True
    assert r2["fired"] is False  # in cooldown


def test_no_cooldown_allows_repeat(tool):
    tool.alert_define("r", condition="gt", threshold=0, cooldown_s=0)
    r1 = json.loads(tool.alert_evaluate("r", 1))
    r2 = json.loads(tool.alert_evaluate("r", 1))
    assert r1["fired"] is True
    assert r2["fired"] is True


# ── alert_list / alert_get ──────────────────────────────────────────────────

def test_list_empty(tool):
    r = json.loads(tool.alert_list())
    assert r["count"] == 0


def test_list_after_define(tool):
    tool.alert_define("a", condition="gt", threshold=0)
    tool.alert_define("b", condition="lt", threshold=10)
    r = json.loads(tool.alert_list())
    assert r["count"] == 2
    names = {x["name"] for x in r["rules"]}
    assert "a" in names and "b" in names


def test_get_basic(tool):
    tool.alert_define("r", condition="gte", threshold=5, severity="error")
    r = json.loads(tool.alert_get("r"))
    assert r["condition"] == "gte"
    assert r["threshold"] == 5
    assert r["severity"] == "error"


def test_get_not_found(tool):
    r = json.loads(tool.alert_get("missing"))
    assert "error" in r


# ── alert_delete ─────────────────────────────────────────────────────────────

def test_delete_basic(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    r = json.loads(tool.alert_delete("r"))
    assert r["deleted"] is True
    assert json.loads(tool.alert_list())["count"] == 0


def test_delete_not_found(tool):
    r = json.loads(tool.alert_delete("ghost"))
    assert "error" in r


# ── alert_history ────────────────────────────────────────────────────────────

def test_history_empty(tool):
    r = json.loads(tool.alert_history())
    assert r["count"] == 0


def test_history_records_fired(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    tool.alert_evaluate("r", 5)
    tool.alert_evaluate("r", 10)
    r = json.loads(tool.alert_history())
    assert r["count"] == 2


def test_history_ignores_nonfired(tool):
    tool.alert_define("r", condition="gt", threshold=100)
    tool.alert_evaluate("r", 50)  # does not fire
    r = json.loads(tool.alert_history())
    assert r["count"] == 0


def test_history_filter_by_rule(tool):
    tool.alert_define("r1", condition="gt", threshold=0)
    tool.alert_define("r2", condition="gt", threshold=0)
    tool.alert_evaluate("r1", 1)
    tool.alert_evaluate("r2", 1)
    r = json.loads(tool.alert_history(rule="r1"))
    assert r["count"] == 1
    assert r["events"][0]["rule"] == "r1"


def test_history_filter_by_severity(tool):
    tool.alert_define("r1", condition="gt", threshold=0, severity="critical")
    tool.alert_define("r2", condition="gt", threshold=0, severity="info")
    tool.alert_evaluate("r1", 1)
    tool.alert_evaluate("r2", 1)
    r = json.loads(tool.alert_history(severity="critical"))
    assert r["count"] == 1


def test_history_limit(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    for i in range(10):
        tool.alert_evaluate("r", i + 1)
    r = json.loads(tool.alert_history(limit=3))
    assert r["count"] == 3


# ── alert_clear ──────────────────────────────────────────────────────────────

def test_clear_all(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    tool.alert_evaluate("r", 1)
    tool.alert_evaluate("r", 2)
    r = json.loads(tool.alert_clear())
    assert r["cleared"] == 2
    assert json.loads(tool.alert_history())["count"] == 0


def test_clear_by_rule(tool):
    tool.alert_define("r1", condition="gt", threshold=0)
    tool.alert_define("r2", condition="gt", threshold=0)
    tool.alert_evaluate("r1", 1)
    tool.alert_evaluate("r2", 1)
    tool.alert_clear(rule="r1")
    r = json.loads(tool.alert_history())
    assert r["count"] == 1
    assert r["events"][0]["rule"] == "r2"


# ── alert_stats ──────────────────────────────────────────────────────────────

def test_stats_empty(tool):
    r = json.loads(tool.alert_stats())
    assert r["total_fires"] == 0


def test_stats_counts(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    tool.alert_evaluate("r", 1)
    tool.alert_evaluate("r", 2)
    r = json.loads(tool.alert_stats())
    assert r["total_fires"] == 2
    assert r["stats"][0]["fire_count"] == 2


def test_stats_per_rule(tool):
    tool.alert_define("a", condition="gt", threshold=0)
    tool.alert_define("b", condition="gt", threshold=100)
    tool.alert_evaluate("a", 1)
    tool.alert_evaluate("b", 50)  # doesn't fire
    r = json.loads(tool.alert_stats())
    by_name = {s["name"]: s for s in r["stats"]}
    assert by_name["a"]["fire_count"] == 1
    assert by_name["b"]["fire_count"] == 0


# ── fire_count tracking ──────────────────────────────────────────────────────

def test_fire_count_increments(tool):
    tool.alert_define("r", condition="gt", threshold=0)
    tool.alert_evaluate("r", 1)
    tool.alert_evaluate("r", 2)
    tool.alert_evaluate("r", 3)
    r = json.loads(tool.alert_get("r"))
    assert r["fire_count"] == 3


# ── execute dispatch ─────────────────────────────────────────────────────────

def test_execute_define(tool):
    r = json.loads(tool.execute("alert_define", {"name": "x", "condition": "gt", "threshold": 0}))
    assert "name" in r


def test_execute_evaluate(tool):
    tool.alert_define("x", condition="gt", threshold=0)
    r = json.loads(tool.execute("alert_evaluate", {"name": "x", "value": 1}))
    assert r["fired"] is True


def test_execute_list(tool):
    r = json.loads(tool.execute("alert_list", {}))
    assert "rules" in r


def test_execute_stats(tool):
    r = json.loads(tool.execute("alert_stats", {}))
    assert "total_fires" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "alert"


def test_description(tool):
    assert "alert" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
