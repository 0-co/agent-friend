"""Tests for RetryTool."""

import json
import time
import pytest

from agent_friend.tools.retry import RetryTool


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tool():
    t = RetryTool(default_max_attempts=3, default_delay_seconds=0.0)
    # Disable actual sleeping in all tests
    t._sleep = lambda s: None
    return t


def _http_ok(status=200, body="ok"):
    return {"ok": True, "status": status, "body": body}


def _http_fail(status=500, error="server error"):
    return {"ok": False, "status": status, "body": "", "error": error}


def _http_net_error():
    return {"ok": False, "status": 0, "body": "", "error": "connection refused"}


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "retry"


def test_description(tool):
    desc = tool.description.lower()
    assert "retry" in desc


def test_definitions_count(tool):
    defs = tool.definitions()
    assert len(defs) == 7


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "retry_http",
        "retry_shell",
        "retry_status",
        "circuit_create",
        "circuit_call",
        "circuit_status",
        "circuit_reset",
    }


# ── retry_http: success ────────────────────────────────────────────────────────


def test_retry_http_success_first_attempt(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.retry_http(url="http://example.com"))
    assert result["ok"] is True
    assert result["attempts"] == 1


def test_retry_http_success_after_one_failure(tool, monkeypatch):
    calls = [_http_fail(500), _http_ok()]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    result = json.loads(tool.retry_http(url="http://example.com"))
    assert result["ok"] is True
    assert result["attempts"] == 2


def test_retry_http_success_after_two_failures(tool, monkeypatch):
    calls = [_http_fail(500), _http_fail(500), _http_ok()]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    result = json.loads(tool.retry_http(url="http://example.com", max_attempts=3))
    assert result["ok"] is True
    assert result["attempts"] == 3


def test_retry_http_all_fail(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(503))
    result = json.loads(tool.retry_http(url="http://example.com", max_attempts=3))
    assert result["ok"] is False
    assert result["attempts"] == 3


def test_retry_http_network_error_retries(tool, monkeypatch):
    calls = [_http_net_error(), _http_net_error(), _http_ok()]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    result = json.loads(tool.retry_http(url="http://example.com", max_attempts=3))
    assert result["ok"] is True


def test_retry_http_429_retries(tool, monkeypatch):
    calls = [_http_fail(429, "rate limited"), _http_ok()]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    result = json.loads(tool.retry_http(url="http://example.com"))
    assert result["ok"] is True
    assert result["attempts"] == 2


def test_retry_http_404_does_not_retry(tool, monkeypatch):
    call_count = [0]

    def mock_req(*a, **kw):
        call_count[0] += 1
        return {"ok": False, "status": 404, "body": "not found", "error": "404"}

    monkeypatch.setattr(tool, "_http_request", mock_req)
    result = json.loads(tool.retry_http(url="http://example.com", max_attempts=3))
    assert result["ok"] is False
    # 404 is not retried
    assert call_count[0] == 1


def test_retry_http_custom_max_attempts(tool, monkeypatch):
    call_count = [0]

    def mock_req(*a, **kw):
        call_count[0] += 1
        return _http_fail(500)

    monkeypatch.setattr(tool, "_http_request", mock_req)
    tool.retry_http(url="http://example.com", max_attempts=5)
    assert call_count[0] == 5


def test_retry_http_passes_method_and_body(tool, monkeypatch):
    captured = {}

    def mock_req(method, url, body, headers):
        captured["method"] = method
        captured["body"] = body
        return _http_ok()

    monkeypatch.setattr(tool, "_http_request", mock_req)
    tool.retry_http(method="POST", url="http://x.com", body='{"a":1}')
    assert captured["method"] == "POST"
    assert captured["body"] == '{"a":1}'


def test_retry_http_updates_stats(tool, monkeypatch):
    calls = [_http_fail(500), _http_ok()]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    tool.retry_http(url="http://example.com")
    status = json.loads(tool.retry_status())
    assert status["total_calls"] == 1
    assert status["total_retries"] == 1
    assert status["total_successes"] == 1


def test_retry_http_jitter_disabled(tool, monkeypatch):
    """Jitter=False should still work and not crash."""
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.retry_http(url="http://x.com", jitter=False))
    assert result["ok"] is True


def test_retry_http_returns_json(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok(200, "hello"))
    raw = tool.retry_http(url="http://x.com")
    parsed = json.loads(raw)
    assert "ok" in parsed
    assert "status" in parsed
    assert "body" in parsed
    assert "attempts" in parsed


def test_retry_http_extra_status_codes(tool, monkeypatch):
    """retry_on_status=[202] means 202 triggers retry."""
    calls = [_http_ok(202, "pending"), _http_ok(200, "done")]
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: calls.pop(0))
    result = json.loads(tool.retry_http(url="http://x.com", retry_on_status=[202]))
    assert result["status"] == 200


# ── retry_shell ────────────────────────────────────────────────────────────────


def test_retry_shell_success(tool):
    result = json.loads(tool.retry_shell(command="echo hello"))
    assert result["ok"] is True
    assert "hello" in result["stdout"]
    assert result["attempts"] == 1


def test_retry_shell_fail_then_succeed(tool, monkeypatch):
    call_count = [0]
    original_run = __import__("subprocess").run

    def mock_run(cmd, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            import subprocess
            class FakeResult:
                returncode = 1
                stdout = ""
                stderr = "error"
            return FakeResult()
        return original_run(cmd, **kwargs)

    import subprocess
    monkeypatch.setattr(subprocess, "run", mock_run)
    result = json.loads(tool.retry_shell(command="echo hi", max_attempts=3))
    # Should succeed on second attempt
    assert result["ok"] is True


def test_retry_shell_all_fail(tool):
    result = json.loads(tool.retry_shell(command="exit 1", max_attempts=3))
    assert result["ok"] is False
    assert result["attempts"] == 3


def test_retry_shell_zero_exit_code(tool):
    result = json.loads(tool.retry_shell(command="true"))
    assert result["ok"] is True


def test_retry_shell_nonzero_exit_code(tool):
    result = json.loads(tool.retry_shell(command="false", max_attempts=2))
    assert result["ok"] is False


def test_retry_shell_captures_stdout(tool):
    result = json.loads(tool.retry_shell(command="echo test_output"))
    assert "test_output" in result["stdout"]


def test_retry_shell_captures_stderr(tool):
    result = json.loads(tool.retry_shell(command="echo err >&2; exit 1", max_attempts=1))
    assert result["ok"] is False


def test_retry_shell_returns_json(tool):
    raw = tool.retry_shell(command="echo hi")
    parsed = json.loads(raw)
    assert "ok" in parsed
    assert "returncode" in parsed
    assert "stdout" in parsed
    assert "stderr" in parsed
    assert "attempts" in parsed


# ── retry_status ───────────────────────────────────────────────────────────────


def test_retry_status_initial(tool):
    status = json.loads(tool.retry_status())
    assert status["total_calls"] == 0
    assert status["total_retries"] == 0
    assert status["total_successes"] == 0
    assert status["total_failures"] == 0


def test_retry_status_after_success(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    tool.retry_http(url="http://x.com")
    status = json.loads(tool.retry_status())
    assert status["total_calls"] == 1
    assert status["total_successes"] == 1
    assert status["total_failures"] == 0


def test_retry_status_after_failure(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.retry_http(url="http://x.com", max_attempts=3)
    status = json.loads(tool.retry_status())
    assert status["total_failures"] == 1
    assert status["total_retries"] == 2  # 3 attempts = 2 retries


def test_retry_status_accumulates(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    tool.retry_http(url="http://x.com")
    tool.retry_http(url="http://x.com")
    status = json.loads(tool.retry_status())
    assert status["total_calls"] == 2
    assert status["total_successes"] == 2


# ── circuit_create ─────────────────────────────────────────────────────────────


def test_circuit_create(tool):
    result = json.loads(tool.circuit_create("svc"))
    assert result["ok"] is True
    assert result["name"] == "svc"
    assert result["state"] == "closed"


def test_circuit_create_custom_params(tool):
    result = json.loads(tool.circuit_create("svc", max_failures=10, reset_timeout_seconds=120))
    assert result["ok"] is True
    status = json.loads(tool.circuit_status("svc"))
    assert status["max_failures"] == 10
    assert status["reset_timeout_seconds"] == 120


def test_circuit_create_overwrites_existing(tool):
    tool.circuit_create("svc", max_failures=3)
    tool.circuit_call("svc", url="http://x.com")  # will fail (no real http)
    tool.circuit_create("svc", max_failures=3)  # reset
    status = json.loads(tool.circuit_status("svc"))
    assert status["failures"] == 0
    assert status["state"] == "closed"


# ── circuit_status ─────────────────────────────────────────────────────────────


def test_circuit_status_initial(tool):
    tool.circuit_create("svc")
    status = json.loads(tool.circuit_status("svc"))
    assert status["state"] == "closed"
    assert status["failures"] == 0
    assert status["opened_at"] is None


def test_circuit_status_unknown(tool):
    result = json.loads(tool.circuit_status("missing"))
    assert result["ok"] is False
    assert "not found" in result["error"]


# ── circuit_call ───────────────────────────────────────────────────────────────


def test_circuit_call_success(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=3)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.circuit_call("svc", url="http://x.com"))
    assert result["ok"] is True
    assert result["circuit_state"] == "closed"


def test_circuit_call_failure_increments(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=3)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.circuit_call("svc", url="http://x.com")
    status = json.loads(tool.circuit_status("svc"))
    assert status["failures"] == 1
    assert status["state"] == "closed"


def test_circuit_opens_after_max_failures(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=3)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    for _ in range(3):
        tool.circuit_call("svc", url="http://x.com")
    status = json.loads(tool.circuit_status("svc"))
    assert status["state"] == "open"


def test_circuit_open_rejects_calls(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=2)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.circuit_call("svc", url="http://x.com")
    tool.circuit_call("svc", url="http://x.com")
    # Circuit is now open
    call_count = [0]

    def counting_mock(*a, **kw):
        call_count[0] += 1
        return _http_ok()

    monkeypatch.setattr(tool, "_http_request", counting_mock)
    result = json.loads(tool.circuit_call("svc", url="http://x.com"))
    # Should be rejected without calling _http_request
    assert result["ok"] is False
    assert call_count[0] == 0


def test_circuit_half_open_after_timeout(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=1, reset_timeout_seconds=0.0)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.circuit_call("svc", url="http://x.com")
    # Circuit is open; reset_timeout=0 so it should immediately go half-open
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.circuit_call("svc", url="http://x.com"))
    assert result["ok"] is True
    # After success in half-open, circuit should close
    status = json.loads(tool.circuit_status("svc"))
    assert status["state"] == "closed"


def test_circuit_unknown_name(tool):
    result = json.loads(tool.circuit_call("missing", url="http://x.com"))
    assert result["ok"] is False


def test_circuit_success_resets_failures(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=5)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.circuit_call("svc", url="http://x.com")
    tool.circuit_call("svc", url="http://x.com")
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    tool.circuit_call("svc", url="http://x.com")
    status = json.loads(tool.circuit_status("svc"))
    assert status["failures"] == 0
    assert status["state"] == "closed"


# ── circuit_reset ──────────────────────────────────────────────────────────────


def test_circuit_reset(tool, monkeypatch):
    tool.circuit_create("svc", max_failures=1)
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_fail(500))
    tool.circuit_call("svc", url="http://x.com")
    assert json.loads(tool.circuit_status("svc"))["state"] == "open"
    result = json.loads(tool.circuit_reset("svc"))
    assert result["ok"] is True
    assert result["state"] == "closed"
    status = json.loads(tool.circuit_status("svc"))
    assert status["state"] == "closed"
    assert status["failures"] == 0


def test_circuit_reset_unknown(tool):
    result = json.loads(tool.circuit_reset("missing"))
    assert result["ok"] is False


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_retry_http(tool, monkeypatch):
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.execute("retry_http", {"url": "http://x.com"}))
    assert result["ok"] is True


def test_execute_retry_shell(tool):
    result = json.loads(tool.execute("retry_shell", {"command": "echo ok"}))
    assert result["ok"] is True


def test_execute_retry_status(tool):
    result = json.loads(tool.execute("retry_status", {}))
    assert "total_calls" in result


def test_execute_circuit_create(tool):
    result = json.loads(tool.execute("circuit_create", {"name": "svc"}))
    assert result["ok"] is True


def test_execute_circuit_call(tool, monkeypatch):
    tool.circuit_create("svc")
    monkeypatch.setattr(tool, "_http_request", lambda *a, **kw: _http_ok())
    result = json.loads(tool.execute("circuit_call", {"name": "svc", "url": "http://x.com"}))
    assert result["ok"] is True


def test_execute_circuit_status(tool):
    tool.circuit_create("svc")
    result = json.loads(tool.execute("circuit_status", {"name": "svc"}))
    assert result["state"] == "closed"


def test_execute_circuit_reset(tool):
    tool.circuit_create("svc")
    result = json.loads(tool.execute("circuit_reset", {"name": "svc"}))
    assert result["ok"] is True


def test_execute_unknown_tool(tool):
    result = json.loads(tool.execute("no_such_tool", {}))
    assert "error" in result


# ── compute_delay helper ───────────────────────────────────────────────────────


def test_compute_delay_no_jitter(tool):
    d = tool._compute_delay(0, delay=1.0, backoff=2.0, jitter=False)
    assert d == pytest.approx(1.0)


def test_compute_delay_backoff(tool):
    d0 = tool._compute_delay(0, delay=1.0, backoff=2.0, jitter=False)
    d1 = tool._compute_delay(1, delay=1.0, backoff=2.0, jitter=False)
    d2 = tool._compute_delay(2, delay=1.0, backoff=2.0, jitter=False)
    assert d0 == pytest.approx(1.0)
    assert d1 == pytest.approx(2.0)
    assert d2 == pytest.approx(4.0)


def test_compute_delay_jitter_in_range(tool):
    for _ in range(20):
        d = tool._compute_delay(0, delay=1.0, backoff=2.0, jitter=True)
        assert 0.75 <= d <= 1.25
