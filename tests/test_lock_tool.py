"""Tests for LockTool — named mutex-style locking."""

import json
import time
import pytest
from agent_friend.tools.lock_tool import LockTool


@pytest.fixture
def tool():
    return LockTool()


# ── lock_acquire ─────────────────────────────────────────────────────────────

def test_acquire_basic(tool):
    r = json.loads(tool.lock_acquire("db", owner="w1"))
    assert r["acquired"] is True
    assert r["lock"] == "db"
    assert r["owner"] == "w1"


def test_acquire_second_fails(tool):
    tool.lock_acquire("db", owner="w1")
    r = json.loads(tool.lock_acquire("db", owner="w2"))
    assert r["acquired"] is False
    assert r["held_by"] == "w1"


def test_acquire_different_names(tool):
    r1 = json.loads(tool.lock_acquire("a", owner="w1"))
    r2 = json.loads(tool.lock_acquire("b", owner="w2"))
    assert r1["acquired"] is True
    assert r2["acquired"] is True


def test_acquire_empty_name_error(tool):
    r = json.loads(tool.lock_acquire(""))
    assert "error" in r


def test_acquire_empty_owner_error(tool):
    r = json.loads(tool.lock_acquire("db", owner=""))
    assert "error" in r


def test_acquire_negative_ttl_error(tool):
    r = json.loads(tool.lock_acquire("db", ttl_s=-1.0))
    assert "error" in r


def test_acquire_negative_wait_error(tool):
    r = json.loads(tool.lock_acquire("db", wait_s=-1.0))
    assert "error" in r


def test_acquire_with_ttl(tool):
    r = json.loads(tool.lock_acquire("db", owner="w1", ttl_s=60.0))
    assert r["acquired"] is True
    assert r["ttl_s"] == 60.0


def test_acquire_returns_waited_ms(tool):
    r = json.loads(tool.lock_acquire("db", owner="w1"))
    assert "waited_ms" in r
    assert isinstance(r["waited_ms"], float)


def test_acquire_same_owner_blocked(tool):
    # Same owner can't re-acquire — no reentrance
    tool.lock_acquire("db", owner="w1")
    r = json.loads(tool.lock_acquire("db", owner="w1"))
    assert r["acquired"] is False


# ── lock_release ─────────────────────────────────────────────────────────────

def test_release_basic(tool):
    tool.lock_acquire("db", owner="w1")
    r = json.loads(tool.lock_release("db", owner="w1"))
    assert r["released"] is True
    assert r["lock"] == "db"


def test_release_wrong_owner_error(tool):
    tool.lock_acquire("db", owner="w1")
    r = json.loads(tool.lock_release("db", owner="w2"))
    assert "error" in r


def test_release_not_held_error(tool):
    r = json.loads(tool.lock_release("db", owner="w1"))
    assert "error" in r


def test_release_allows_reacquire(tool):
    tool.lock_acquire("db", owner="w1")
    tool.lock_release("db", owner="w1")
    r = json.loads(tool.lock_acquire("db", owner="w2"))
    assert r["acquired"] is True


# ── lock_try ─────────────────────────────────────────────────────────────────

def test_try_succeeds(tool):
    r = json.loads(tool.lock_try("x", owner="w1"))
    assert r["acquired"] is True


def test_try_fails_immediately(tool):
    tool.lock_acquire("x", owner="w1")
    r = json.loads(tool.lock_try("x", owner="w2"))
    assert r["acquired"] is False
    assert "held_by" in r


def test_try_no_wait(tool):
    tool.lock_acquire("x", owner="w1")
    t0 = time.monotonic()
    tool.lock_try("x", owner="w2")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.1  # should not block


# ── lock_status ───────────────────────────────────────────────────────────────

def test_status_held(tool):
    tool.lock_acquire("x", owner="w1", ttl_s=60)
    r = json.loads(tool.lock_status("x"))
    assert r["held"] is True
    assert r["owner"] == "w1"
    assert r["remaining_s"] is not None


def test_status_not_held(tool):
    r = json.loads(tool.lock_status("x"))
    assert r["held"] is False


def test_status_after_release(tool):
    tool.lock_acquire("x", owner="w1")
    tool.lock_release("x", owner="w1")
    r = json.loads(tool.lock_status("x"))
    assert r["held"] is False


def test_status_no_ttl(tool):
    tool.lock_acquire("x", owner="w1")  # no ttl
    r = json.loads(tool.lock_status("x"))
    assert r["held"] is True
    assert r["remaining_s"] is None


# ── TTL / expiry ──────────────────────────────────────────────────────────────

def test_ttl_expires(tool):
    tool.lock_acquire("x", owner="w1", ttl_s=0.05)
    time.sleep(0.1)
    r = json.loads(tool.lock_status("x"))
    assert r["held"] is False


def test_ttl_allows_reacquire_after_expiry(tool):
    tool.lock_acquire("x", owner="w1", ttl_s=0.05)
    time.sleep(0.1)
    r = json.loads(tool.lock_acquire("x", owner="w2"))
    assert r["acquired"] is True


# ── lock_list ─────────────────────────────────────────────────────────────────

def test_list_empty(tool):
    r = json.loads(tool.lock_list())
    assert r["count"] == 0


def test_list_shows_held(tool):
    tool.lock_acquire("a", owner="w1")
    tool.lock_acquire("b", owner="w2")
    r = json.loads(tool.lock_list())
    assert r["count"] == 2
    names = {lk["name"] for lk in r["locks"]}
    assert "a" in names and "b" in names


def test_list_excludes_expired(tool):
    tool.lock_acquire("x", owner="w1", ttl_s=0.05)
    time.sleep(0.1)
    r = json.loads(tool.lock_list())
    assert r["count"] == 0


# ── lock_release_all ──────────────────────────────────────────────────────────

def test_release_all_basic(tool):
    tool.lock_acquire("a", owner="worker")
    tool.lock_acquire("b", owner="worker")
    tool.lock_acquire("c", owner="other")
    r = json.loads(tool.lock_release_all("worker"))
    assert r["released_count"] == 2
    assert sorted(r["locks"]) == ["a", "b"]


def test_release_all_empty_owner_error(tool):
    r = json.loads(tool.lock_release_all(""))
    assert "error" in r


def test_release_all_no_locks(tool):
    r = json.loads(tool.lock_release_all("nobody"))
    assert r["released_count"] == 0


# ── lock_expire ───────────────────────────────────────────────────────────────

def test_expire_basic(tool):
    tool.lock_acquire("x", owner="w1")
    r = json.loads(tool.lock_expire("x"))
    assert r["expired"] is True
    assert r["was_owner"] == "w1"


def test_expire_not_held_error(tool):
    r = json.loads(tool.lock_expire("ghost"))
    assert "error" in r


def test_expire_allows_reacquire(tool):
    tool.lock_acquire("x", owner="w1")
    tool.lock_expire("x")
    r = json.loads(tool.lock_acquire("x", owner="w2"))
    assert r["acquired"] is True


# ── lock_stats ────────────────────────────────────────────────────────────────

def test_stats_empty(tool):
    r = json.loads(tool.lock_stats())
    assert r["total_acquisitions"] == 0
    assert r["currently_held"] == 0


def test_stats_counts_acquisitions(tool):
    tool.lock_acquire("a", owner="w1")
    tool.lock_acquire("b", owner="w1")
    r = json.loads(tool.lock_stats())
    assert r["total_acquisitions"] == 2
    assert r["currently_held"] == 2


def test_stats_currently_held_decreases_after_release(tool):
    tool.lock_acquire("x", owner="w1")
    tool.lock_release("x", owner="w1")
    r = json.loads(tool.lock_stats())
    assert r["currently_held"] == 0
    assert r["total_acquisitions"] == 1


# ── execute dispatch ──────────────────────────────────────────────────────────

def test_execute_acquire(tool):
    r = json.loads(tool.execute("lock_acquire", {"name": "x", "owner": "w1"}))
    assert r["acquired"] is True


def test_execute_release(tool):
    tool.lock_acquire("x", owner="w1")
    r = json.loads(tool.execute("lock_release", {"name": "x", "owner": "w1"}))
    assert r["released"] is True


def test_execute_list(tool):
    r = json.loads(tool.execute("lock_list", {}))
    assert "locks" in r


def test_execute_stats(tool):
    r = json.loads(tool.execute("lock_stats", {}))
    assert "total_acquisitions" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ─────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "lock"


def test_description(tool):
    assert "lock" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
