"""Tests for WebhookTool — HTTP webhook receiver."""

import json
import os
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.tools.webhook import WebhookTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _send_after_delay(url: str, data: str, delay: float = 0.05, content_type: str = "application/json"):
    """Send a POST request after a short delay (in a daemon thread)."""
    def _send():
        time.sleep(delay)
        req = urllib.request.Request(
            url,
            data=data.encode("utf-8"),
            method="POST",
            headers={"Content-Type": content_type},
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    threading.Thread(target=_send, daemon=True).start()


def _find_free_port() -> int:
    """Return a free TCP port by binding to port 0 and immediately closing."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestBasicProperties:
    def test_name(self):
        tool = WebhookTool()
        assert tool.name == "webhook"

    def test_description(self):
        tool = WebhookTool()
        assert len(tool.description) > 0
        assert "webhook" in tool.description.lower() or "HTTP" in tool.description

    def test_definitions_count(self):
        tool = WebhookTool()
        defs = tool.definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "wait_for_webhook"


# ---------------------------------------------------------------------------
# listen() — core functionality
# ---------------------------------------------------------------------------

class TestListen:
    def test_listen_receives_post(self):
        """Sending a POST returns a result dict with expected keys."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        payload = json.dumps({"hello": "world"})

        def _listen():
            return tool.listen(path="/webhook", timeout=5.0)

        # Start listen in background, send from main thread
        result_holder = []
        exc_holder = []

        def _run():
            try:
                result_holder.append(tool.listen(path="/webhook", timeout=5.0))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        # Give server a moment to bind
        time.sleep(0.1)
        port = tool.get_port() or tool._actual_port

        # Discover port by peeking at the actual port after thread starts
        # Wait for port to be assigned
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        port = tool._actual_port
        url = f"http://127.0.0.1:{port}/webhook"
        req = urllib.request.Request(url, data=payload.encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)

        t.join(timeout=5)
        assert not exc_holder, f"Exception in listen thread: {exc_holder[0]}"
        assert len(result_holder) == 1
        result = result_holder[0]
        for key in ("path", "method", "headers", "body", "json", "received_at"):
            assert key in result, f"Missing key: {key}"

    def test_listen_returns_path(self):
        """Result contains the request path."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/mypath", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/mypath"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert result_holder[0]["path"] == "/mypath"

    def test_listen_returns_body(self):
        """Result contains the raw request body."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        body_text = '{"key": "value"}'
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=body_text.encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert result_holder[0]["body"] == body_text

    def test_listen_returns_json_parsed(self):
        """A valid JSON body sets the json key to a dict."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        payload = json.dumps({"event": "payment", "amount": 42})
        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=payload.encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        result = result_holder[0]
        assert isinstance(result["json"], dict)
        assert result["json"]["event"] == "payment"
        assert result["json"]["amount"] == 42

    def test_listen_non_json_body(self):
        """A non-JSON body sets the json key to None."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=b"not-json-at-all", method="POST",
                                     headers={"Content-Type": "text/plain"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert result_holder[0]["json"] is None
        assert result_holder[0]["body"] == "not-json-at-all"

    def test_listen_port_assigned(self):
        """Port 0 results in an actual assigned port > 0."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        assigned_port = tool._actual_port
        assert assigned_port is not None
        assert assigned_port > 0

        url = f"http://127.0.0.1:{assigned_port}/webhook"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

    def test_listen_timeout(self):
        """No request within timeout raises TimeoutError."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        with pytest.raises(TimeoutError):
            tool.listen(path="/webhook", timeout=0.3)

    def test_listen_path_mismatch_returns_404(self):
        """A request to the wrong path returns 404 and the tool keeps waiting (then times out)."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []
        exc_holder = []

        def _run():
            try:
                result_holder.append(tool.listen(path="/correct", timeout=0.5))
            except TimeoutError as e:
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/wrong"
        req = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            assert resp.status == 404
        except urllib.request.HTTPError as e:
            assert e.code == 404

        t.join(timeout=2)
        # Should have timed out since the only request was to the wrong path
        assert len(exc_holder) == 1
        assert isinstance(exc_holder[0], TimeoutError)
        assert len(result_holder) == 0

    def test_get_port_before_listen(self):
        """get_port() returns None before listening starts."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        assert tool.get_port() is None

    def test_listen_custom_path(self):
        """Custom path is accepted and returned in result."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/payment/callback", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/payment/callback"
        req = urllib.request.Request(url, data=b'{"paid": true}', method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert result_holder[0]["path"] == "/payment/callback"
        assert result_holder[0]["json"]["paid"] is True

    def test_listen_custom_port(self):
        """A specific port is used when requested."""
        free_port = _find_free_port()
        tool = WebhookTool(port=free_port, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        assert tool._actual_port == free_port

        url = f"http://127.0.0.1:{free_port}/webhook"
        req = urllib.request.Request(url, data=b'{}', method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert len(result_holder) == 1


# ---------------------------------------------------------------------------
# execute() / LLM dispatch
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def _listen_and_get_port(self, tool, path="/webhook", timeout=5.0):
        """Start listen in a thread and return (thread, port_getter)."""
        result_holder = []
        exc_holder = []

        def _run():
            try:
                result_holder.append(tool.listen(path=path, timeout=timeout))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)
        return t, result_holder, exc_holder

    def test_tool_call_wait_for_webhook(self):
        """execute() dispatches wait_for_webhook and returns result."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []
        exc_holder = []

        def _run():
            try:
                result_holder.append(
                    tool.execute("wait_for_webhook", {"path": "/webhook", "timeout": 5.0})
                )
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=b'{"test": 1}', method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert not exc_holder
        assert len(result_holder) == 1
        parsed = json.loads(result_holder[0])
        assert parsed["path"] == "/webhook"
        assert parsed["json"]["test"] == 1

    def test_tool_call_returns_json_string(self):
        """execute() returns a str (JSON-encoded), not a dict."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(
                tool.execute("wait_for_webhook", {"path": "/webhook", "timeout": 5.0})
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=b'{}', method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert len(result_holder) == 1
        assert isinstance(result_holder[0], str)
        # Must be valid JSON
        parsed = json.loads(result_holder[0])
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# Multiple sequential listens
# ---------------------------------------------------------------------------

class TestSequentialListens:
    def test_multiple_sequential_listens(self):
        """Two separate listen() calls on the same tool both work."""
        tool = WebhookTool(port=0, host="127.0.0.1")

        results = []
        for i in range(2):
            result_holder = []

            def _run(rh=result_holder):
                rh.append(tool.listen(path="/webhook", timeout=5.0))

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            for _ in range(50):
                if tool._actual_port is not None:
                    break
                time.sleep(0.05)

            url = f"http://127.0.0.1:{tool._actual_port}/webhook"
            payload = json.dumps({"request": i})
            req = urllib.request.Request(url, data=payload.encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
            t.join(timeout=5)

            assert len(result_holder) == 1
            results.append(result_holder[0])

        assert results[0]["json"]["request"] == 0
        assert results[1]["json"]["request"] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_listen_empty_body(self):
        """An empty POST body is handled: body is empty string, json is None."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        # Send POST with no body
        req = urllib.request.Request(url, data=b"", method="POST")
        urllib.request.urlopen(req, timeout=5)
        t.join(timeout=5)

        assert len(result_holder) == 1
        result = result_holder[0]
        assert result["body"] == ""
        assert result["json"] is None

    def test_listen_large_body(self):
        """A 10 KB body is received correctly."""
        tool = WebhookTool(port=0, host="127.0.0.1")
        result_holder = []

        def _run():
            result_holder.append(tool.listen(path="/webhook", timeout=5.0))

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        for _ in range(50):
            if tool._actual_port is not None:
                break
            time.sleep(0.05)

        large_body = "x" * 10_240  # 10 KB of plain text
        url = f"http://127.0.0.1:{tool._actual_port}/webhook"
        req = urllib.request.Request(url, data=large_body.encode(), method="POST",
                                     headers={"Content-Type": "text/plain"})
        urllib.request.urlopen(req, timeout=10)
        t.join(timeout=10)

        assert len(result_holder) == 1
        result = result_holder[0]
        assert result["body"] == large_body
        assert len(result["body"]) == 10_240
        assert result["json"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
