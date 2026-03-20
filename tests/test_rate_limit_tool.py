"""Tests for RateLimitTool."""

import json
import time
import pytest

from agent_friend.tools.rate_limit import RateLimitTool


@pytest.fixture
def tool():
    return RateLimitTool()


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "rate_limit"


def test_description(tool):
    desc = tool.description.lower()
    assert "rate" in desc or "limit" in desc


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "limiter_create", "limiter_check", "limiter_consume",
        "limiter_acquire", "limiter_status", "limiter_reset",
        "limiter_delete", "limiter_list",
    }


# ── limiter_create ────────────────────────────────────────────────────────────


def test_create_fixed(tool):
    result = json.loads(tool.limiter_create("test", max_calls=5, window_seconds=60, algorithm="fixed"))
    assert result["created"] is True
    assert result["algorithm"] == "fixed"


def test_create_sliding(tool):
    result = json.loads(tool.limiter_create("sl", algorithm="sliding"))
    assert result["created"] is True
    assert result["algorithm"] == "sliding"


def test_create_token_bucket(tool):
    result = json.loads(tool.limiter_create("tb", algorithm="token_bucket"))
    assert result["created"] is True


def test_create_duplicate_fails(tool):
    tool.limiter_create("dup")
    result = json.loads(tool.limiter_create("dup"))
    assert "error" in result


def test_create_unknown_algorithm(tool):
    result = json.loads(tool.limiter_create("x", algorithm="nonexistent"))
    assert "error" in result


def test_create_max_limiters():
    t = RateLimitTool(max_limiters=2)
    t.limiter_create("a")
    t.limiter_create("b")
    result = json.loads(t.limiter_create("c"))
    assert "error" in result


# ── limiter_check — fixed ──────────────────────────────────────────────────────


def test_check_allowed_initially(tool):
    tool.limiter_create("f", max_calls=5, window_seconds=60, algorithm="fixed")
    result = json.loads(tool.limiter_check("f"))
    assert result["allowed"] is True
    assert result["remaining"] == 5


def test_check_does_not_consume(tool):
    tool.limiter_create("f2", max_calls=5, window_seconds=60)
    tool.limiter_check("f2")
    result = json.loads(tool.limiter_check("f2"))
    assert result["count"] == 0


def test_check_unknown_name(tool):
    result = json.loads(tool.limiter_check("ghost"))
    assert "error" in result


# ── limiter_consume — fixed ────────────────────────────────────────────────────


def test_consume_decrements(tool):
    tool.limiter_create("c", max_calls=3, window_seconds=60)
    tool.limiter_consume("c")
    result = json.loads(tool.limiter_check("c"))
    assert result["count"] == 1
    assert result["remaining"] == 2


def test_consume_exhausts_limit(tool):
    tool.limiter_create("ex", max_calls=2, window_seconds=60)
    tool.limiter_consume("ex")
    tool.limiter_consume("ex")
    result = json.loads(tool.limiter_consume("ex"))
    assert result["consumed"] is False


def test_consume_unknown(tool):
    result = json.loads(tool.limiter_consume("nope"))
    assert "error" in result


# ── limiter_acquire ────────────────────────────────────────────────────────────


def test_acquire_allowed(tool):
    tool.limiter_create("acq", max_calls=3, window_seconds=60)
    result = json.loads(tool.limiter_acquire("acq"))
    assert result["allowed"] is True


def test_acquire_consumes_token(tool):
    tool.limiter_create("acq2", max_calls=3, window_seconds=60)
    tool.limiter_acquire("acq2")
    result = json.loads(tool.limiter_check("acq2"))
    assert result["count"] == 1


def test_acquire_denied_when_exhausted(tool):
    tool.limiter_create("acq3", max_calls=1, window_seconds=60)
    tool.limiter_acquire("acq3")
    result = json.loads(tool.limiter_acquire("acq3"))
    assert result["allowed"] is False


def test_acquire_denied_does_not_consume(tool):
    tool.limiter_create("acq4", max_calls=1, window_seconds=60)
    tool.limiter_acquire("acq4")
    tool.limiter_acquire("acq4")  # denied
    result = json.loads(tool.limiter_check("acq4"))
    assert result["count"] == 1  # still just 1


# ── limiter_status ─────────────────────────────────────────────────────────────


def test_status_fixed(tool):
    tool.limiter_create("st", max_calls=10, window_seconds=30, algorithm="fixed")
    status = json.loads(tool.limiter_status("st"))
    assert status["algorithm"] == "fixed"
    assert status["limit"] == 10
    assert status["window_seconds"] == 30


def test_status_sliding(tool):
    tool.limiter_create("sl2", max_calls=5, window_seconds=60, algorithm="sliding")
    status = json.loads(tool.limiter_status("sl2"))
    assert status["algorithm"] == "sliding"


def test_status_token_bucket(tool):
    tool.limiter_create("tb2", algorithm="token_bucket", max_calls=10, window_seconds=60)
    status = json.loads(tool.limiter_status("tb2"))
    assert status["algorithm"] == "token_bucket"
    assert "tokens" in status


def test_status_unknown(tool):
    result = json.loads(tool.limiter_status("notfound"))
    assert "error" in result


# ── limiter_reset ──────────────────────────────────────────────────────────────


def test_reset_clears_count(tool):
    tool.limiter_create("r", max_calls=3, window_seconds=60)
    tool.limiter_consume("r")
    tool.limiter_consume("r")
    tool.limiter_reset("r")
    result = json.loads(tool.limiter_check("r"))
    assert result["count"] == 0
    assert result["remaining"] == 3


def test_reset_unknown(tool):
    result = json.loads(tool.limiter_reset("nope"))
    assert "error" in result


# ── limiter_delete ─────────────────────────────────────────────────────────────


def test_delete_removes_limiter(tool):
    tool.limiter_create("del", max_calls=5, window_seconds=60)
    tool.limiter_delete("del")
    result = json.loads(tool.limiter_check("del"))
    assert "error" in result


def test_delete_unknown(tool):
    result = json.loads(tool.limiter_delete("ghost"))
    assert "error" in result


# ── limiter_list ───────────────────────────────────────────────────────────────


def test_list_empty(tool):
    result = json.loads(tool.limiter_list())
    assert result == []


def test_list_shows_all(tool):
    tool.limiter_create("a", max_calls=5, window_seconds=60)
    tool.limiter_create("b", max_calls=10, window_seconds=60)
    result = json.loads(tool.limiter_list())
    names = {r["name"] for r in result}
    assert names == {"a", "b"}


# ── sliding window behaviour ───────────────────────────────────────────────────


def test_sliding_allows_up_to_limit(tool):
    tool.limiter_create("sw", max_calls=3, window_seconds=60, algorithm="sliding")
    for _ in range(3):
        r = json.loads(tool.limiter_acquire("sw"))
        assert r["allowed"] is True
    r = json.loads(tool.limiter_acquire("sw"))
    assert r["allowed"] is False


def test_sliding_check_includes_reset_in(tool):
    tool.limiter_create("sw2", max_calls=2, window_seconds=60, algorithm="sliding")
    tool.limiter_acquire("sw2")
    tool.limiter_acquire("sw2")
    result = json.loads(tool.limiter_check("sw2"))
    assert result["allowed"] is False
    assert result["reset_in_seconds"] > 0


# ── token bucket behaviour ─────────────────────────────────────────────────────


def test_token_bucket_starts_full(tool):
    tool.limiter_create("tk", max_calls=10, window_seconds=60, algorithm="token_bucket")
    status = json.loads(tool.limiter_status("tk"))
    assert status["tokens"] == pytest.approx(10.0, abs=0.1)


def test_token_bucket_consumes(tool):
    tool.limiter_create("tk2", max_calls=5, window_seconds=60, algorithm="token_bucket")
    tool.limiter_acquire("tk2")
    status = json.loads(tool.limiter_status("tk2"))
    assert status["tokens"] < 5.0


def test_token_bucket_exhausted(tool):
    tool.limiter_create("tk3", max_calls=2, window_seconds=60, algorithm="token_bucket")
    tool.limiter_acquire("tk3")
    tool.limiter_acquire("tk3")
    result = json.loads(tool.limiter_acquire("tk3"))
    assert result["allowed"] is False
    assert result["wait_seconds"] > 0


def test_token_bucket_reset_refills(tool):
    tool.limiter_create("tk4", max_calls=3, window_seconds=60, algorithm="token_bucket")
    tool.limiter_acquire("tk4")
    tool.limiter_acquire("tk4")
    tool.limiter_reset("tk4")
    status = json.loads(tool.limiter_status("tk4"))
    assert status["tokens"] == pytest.approx(3.0, abs=0.1)


def test_token_bucket_custom_rate(tool):
    tool.limiter_create(
        "tk5", algorithm="token_bucket",
        rate_per_second=2.0, burst_capacity=5.0
    )
    status = json.loads(tool.limiter_status("tk5"))
    assert status["rate_per_second"] == pytest.approx(2.0)
    assert status["capacity"] == pytest.approx(5.0)


# ── window reset — fixed ───────────────────────────────────────────────────────


def test_fixed_window_resets_after_window(tool):
    tool.limiter_create("fw", max_calls=2, window_seconds=0.05)  # 50ms window
    tool.limiter_acquire("fw")
    tool.limiter_acquire("fw")
    # Window should be exhausted
    result = json.loads(tool.limiter_acquire("fw"))
    assert result["allowed"] is False
    # Wait for window to reset
    time.sleep(0.06)
    result = json.loads(tool.limiter_acquire("fw"))
    assert result["allowed"] is True


def test_fixed_window_reset_in_positive_when_active(tool):
    tool.limiter_create("fwt", max_calls=5, window_seconds=10)
    tool.limiter_acquire("fwt")
    result = json.loads(tool.limiter_check("fwt"))
    assert result["reset_in_seconds"] > 0


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_create(tool):
    result = json.loads(tool.execute("limiter_create", {"name": "ex1", "max_calls": 5, "window_seconds": 60}))
    assert result["created"] is True


def test_execute_check(tool):
    tool.execute("limiter_create", {"name": "ex2"})
    result = json.loads(tool.execute("limiter_check", {"name": "ex2"}))
    assert "allowed" in result


def test_execute_consume(tool):
    tool.execute("limiter_create", {"name": "ex3"})
    result = json.loads(tool.execute("limiter_consume", {"name": "ex3"}))
    assert "consumed" in result


def test_execute_acquire(tool):
    tool.execute("limiter_create", {"name": "ex4"})
    result = json.loads(tool.execute("limiter_acquire", {"name": "ex4"}))
    assert result["allowed"] is True


def test_execute_status(tool):
    tool.execute("limiter_create", {"name": "ex5"})
    result = json.loads(tool.execute("limiter_status", {"name": "ex5"}))
    assert "algorithm" in result


def test_execute_reset(tool):
    tool.execute("limiter_create", {"name": "ex6"})
    result = json.loads(tool.execute("limiter_reset", {"name": "ex6"}))
    assert result["reset"] is True


def test_execute_delete(tool):
    tool.execute("limiter_create", {"name": "ex7"})
    result = json.loads(tool.execute("limiter_delete", {"name": "ex7"}))
    assert result["deleted"] is True


def test_execute_list(tool):
    result = json.loads(tool.execute("limiter_list", {}))
    assert isinstance(result, list)


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
