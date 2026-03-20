"""Tests for TimerTool — stopwatch, countdown, and benchmarking."""

import json
import time
import pytest
from agent_friend.tools.timer_tool import TimerTool


@pytest.fixture
def tool():
    return TimerTool()


# ── timer_start ────────────────────────────────────────────────────────────

def test_start_returns_started(tool):
    r = json.loads(tool.timer_start("t1"))
    assert r["started"] is True
    assert r["name"] == "t1"


def test_start_creates_timer(tool):
    tool.timer_start("t1")
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["running"] is True


def test_start_restarts_existing(tool):
    tool.timer_start("t1")
    time.sleep(0.01)
    tool.timer_start("t1")  # restart
    r = json.loads(tool.timer_elapsed("t1"))
    # Should be near 0ms after restart
    assert r["elapsed_ms"] < 50


# ── timer_stop ─────────────────────────────────────────────────────────────

def test_stop_returns_elapsed(tool):
    tool.timer_start("t1")
    time.sleep(0.01)
    r = json.loads(tool.timer_stop("t1"))
    assert r["elapsed_ms"] >= 10
    assert r["elapsed_s"] >= 0.01
    assert r["running"] is False


def test_stop_missing_timer(tool):
    r = json.loads(tool.timer_stop("nope"))
    assert "error" in r


def test_stop_twice_same_result(tool):
    tool.timer_start("t1")
    time.sleep(0.01)
    r1 = json.loads(tool.timer_stop("t1"))
    r2 = json.loads(tool.timer_stop("t1"))
    # Second stop should return same elapsed (timer is already stopped)
    assert abs(r1["elapsed_ms"] - r2["elapsed_ms"]) < 5


# ── timer_elapsed ──────────────────────────────────────────────────────────

def test_elapsed_while_running(tool):
    tool.timer_start("t1")
    time.sleep(0.01)
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["elapsed_ms"] >= 10
    assert r["running"] is True


def test_elapsed_missing(tool):
    r = json.loads(tool.timer_elapsed("nope"))
    assert "error" in r


def test_elapsed_does_not_stop(tool):
    tool.timer_start("t1")
    json.loads(tool.timer_elapsed("t1"))
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["running"] is True


# ── timer_lap ──────────────────────────────────────────────────────────────

def test_lap_records_split(tool):
    tool.timer_start("t1")
    time.sleep(0.01)
    r = json.loads(tool.timer_lap("t1"))
    assert r["lap_number"] == 1
    assert r["lap_ms"] >= 10


def test_lap_multiple(tool):
    tool.timer_start("t1")
    time.sleep(0.005)
    tool.timer_lap("t1")
    time.sleep(0.005)
    r = json.loads(tool.timer_lap("t1"))
    assert r["lap_number"] == 2


def test_lap_shows_in_list(tool):
    tool.timer_start("t1")
    tool.timer_lap("t1")
    r = json.loads(tool.timer_elapsed("t1"))
    assert len(r["laps"]) == 1


def test_lap_stopped_timer_error(tool):
    tool.timer_start("t1")
    tool.timer_stop("t1")
    r = json.loads(tool.timer_lap("t1"))
    assert "error" in r


def test_lap_missing_timer(tool):
    r = json.loads(tool.timer_lap("nope"))
    assert "error" in r


# ── timer_reset ────────────────────────────────────────────────────────────

def test_reset_clears_elapsed(tool):
    tool.timer_start("t1")
    time.sleep(0.02)
    tool.timer_reset("t1")
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["elapsed_ms"] < 20  # reset clears


def test_reset_clears_laps(tool):
    tool.timer_start("t1")
    tool.timer_lap("t1")
    tool.timer_reset("t1")
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["laps"] == []


def test_reset_missing_timer(tool):
    r = json.loads(tool.timer_reset("nope"))
    assert "error" in r


def test_reset_restarts_running(tool):
    tool.timer_start("t1")
    tool.timer_stop("t1")
    tool.timer_reset("t1")
    r = json.loads(tool.timer_elapsed("t1"))
    assert r["running"] is True


# ── timer_delete ───────────────────────────────────────────────────────────

def test_delete_existing(tool):
    tool.timer_start("t1")
    r = json.loads(tool.timer_delete("t1"))
    assert r["deleted"] is True
    r2 = json.loads(tool.timer_elapsed("t1"))
    assert "error" in r2


def test_delete_missing(tool):
    r = json.loads(tool.timer_delete("nope"))
    assert r["deleted"] is False


# ── timer_list ─────────────────────────────────────────────────────────────

def test_list_empty(tool):
    r = json.loads(tool.timer_list())
    assert r == []


def test_list_multiple(tool):
    tool.timer_start("a")
    tool.timer_start("b")
    r = json.loads(tool.timer_list())
    names = [t["name"] for t in r]
    assert "a" in names
    assert "b" in names


def test_list_shows_running_state(tool):
    tool.timer_start("t1")
    tool.timer_stop("t1")
    tool.timer_start("t2")
    r = json.loads(tool.timer_list())
    by_name = {t["name"]: t for t in r}
    assert by_name["t1"]["running"] is False
    assert by_name["t2"]["running"] is True


# ── countdown_start ────────────────────────────────────────────────────────

def test_countdown_start(tool):
    r = json.loads(tool.countdown_start("cd1", 60.0))
    assert r["started"] is True
    assert r["duration_ms"] == 60000.0


def test_countdown_invalid_seconds(tool):
    r = json.loads(tool.countdown_start("cd1", -1))
    assert "error" in r


def test_countdown_zero_seconds(tool):
    r = json.loads(tool.countdown_start("cd1", 0))
    assert "error" in r


# ── countdown_remaining ────────────────────────────────────────────────────

def test_countdown_remaining_positive(tool):
    tool.countdown_start("cd1", 60.0)
    r = json.loads(tool.countdown_remaining("cd1"))
    assert r["remaining_s"] > 59.0
    assert r["expired"] is False


def test_countdown_remaining_expired(tool):
    tool.countdown_start("cd1", 0.01)
    time.sleep(0.02)
    r = json.loads(tool.countdown_remaining("cd1"))
    assert r["expired"] is True
    assert r["remaining_ms"] == 0.0


def test_countdown_remaining_missing(tool):
    r = json.loads(tool.countdown_remaining("nope"))
    assert "error" in r


# ── countdown_list ─────────────────────────────────────────────────────────

def test_countdown_list_empty(tool):
    r = json.loads(tool.countdown_list())
    assert r == []


def test_countdown_list_multiple(tool):
    tool.countdown_start("a", 30)
    tool.countdown_start("b", 60)
    r = json.loads(tool.countdown_list())
    names = [c["name"] for c in r]
    assert "a" in names
    assert "b" in names


# ── timer_benchmark ────────────────────────────────────────────────────────

def test_benchmark_basic(tool):
    r = json.loads(tool.timer_benchmark("echo hello", runs=2))
    assert r["runs"] == 2
    assert r["avg_ms"] >= 0
    assert r["min_ms"] <= r["avg_ms"]
    assert r["avg_ms"] <= r["max_ms"]
    assert len(r["results"]) == 2


def test_benchmark_total_equals_sum(tool):
    r = json.loads(tool.timer_benchmark("echo x", runs=3))
    assert abs(r["total_ms"] - sum(r["results"])) < 1.0


def test_benchmark_invalid_runs_zero(tool):
    r = json.loads(tool.timer_benchmark("echo x", runs=0))
    assert "error" in r


def test_benchmark_invalid_runs_too_many(tool):
    r = json.loads(tool.timer_benchmark("echo x", runs=51))
    assert "error" in r


def test_benchmark_has_command(tool):
    r = json.loads(tool.timer_benchmark("true", runs=1))
    assert r["command"] == "true"


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_timer_start(tool):
    r = json.loads(tool.execute("timer_start", {"name": "t"}))
    assert r["started"] is True


def test_execute_timer_list(tool):
    r = json.loads(tool.execute("timer_list", {}))
    assert isinstance(r, list)


def test_execute_countdown_list(tool):
    r = json.loads(tool.execute("countdown_list", {}))
    assert isinstance(r, list)


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "timer"


def test_description(tool):
    assert "timer" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 11


def test_definitions_required_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
