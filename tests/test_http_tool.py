"""Tests for HTTPTool — generic REST client."""

import json
import unittest
from unittest.mock import patch, MagicMock
import io
import urllib.error

from agent_friend.tools.http import HTTPTool


def _make_response(body: bytes, status: int = 200, content_type: str = "text/plain"):
    """Create a mock urllib response."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.headers = {"Content-Type": content_type}
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestHTTPToolInit(unittest.TestCase):
    def test_name(self):
        self.assertEqual(HTTPTool().name, "http")

    def test_description_nonempty(self):
        self.assertTrue(len(HTTPTool().description) > 0)

    def test_definitions_count(self):
        self.assertEqual(len(HTTPTool().definitions()), 1)

    def test_definition_name(self):
        self.assertEqual(HTTPTool().definitions()[0]["name"], "http_request")

    def test_definition_has_input_schema(self):
        defn = HTTPTool().definitions()[0]
        self.assertIn("input_schema", defn)
        props = defn["input_schema"]["properties"]
        self.assertIn("method", props)
        self.assertIn("url", props)

    def test_definition_required_fields(self):
        required = HTTPTool().definitions()[0]["input_schema"]["required"]
        self.assertIn("method", required)
        self.assertIn("url", required)

    def test_custom_timeout(self):
        tool = HTTPTool(timeout=30)
        self.assertEqual(tool.timeout, 30)

    def test_custom_max_body(self):
        tool = HTTPTool(max_body=1024)
        self.assertEqual(tool.max_body, 1024)

    def test_default_headers(self):
        tool = HTTPTool(default_headers={"Authorization": "Bearer token"})
        self.assertEqual(tool.default_headers["Authorization"], "Bearer token")

    def test_empty_default_headers(self):
        self.assertEqual(HTTPTool().default_headers, {})


class TestHTTPToolValidation(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    def test_unknown_tool_name(self):
        result = self.tool.execute("other", {"method": "GET", "url": "https://example.com"})
        self.assertIn("Unknown", result)

    def test_missing_url(self):
        result = self.tool.execute("http_request", {"method": "GET", "url": ""})
        self.assertIn("Error", result)

    def test_invalid_scheme(self):
        result = self.tool.execute("http_request", {"method": "GET", "url": "ftp://example.com"})
        self.assertIn("Error", result)
        self.assertIn("http", result)

    def test_invalid_method(self):
        result = self.tool.execute("http_request", {"method": "CONNECT", "url": "https://example.com"})
        self.assertIn("Error", result)
        self.assertIn("method", result.lower())

    def test_missing_method_defaults(self):
        # method is required in schema — but test graceful handling if omitted
        result = self.tool.execute("http_request", {"url": "https://example.com"})
        # Should handle missing method somehow (default GET or error)
        self.assertIsInstance(result, str)


class TestHTTPToolGET(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_get_plain_text(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"hello world", 200, "text/plain")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/hello"})
        data = json.loads(result)
        self.assertEqual(data["status"], 200)
        self.assertIn("hello world", data["body"])

    @patch("urllib.request.urlopen")
    def test_get_json_response(self, mock_urlopen):
        body = json.dumps({"key": "value"}).encode()
        mock_urlopen.return_value = _make_response(body, 200, "application/json")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/data"})
        data = json.loads(result)
        self.assertEqual(data["status"], 200)
        self.assertIn("json", data)
        self.assertEqual(data["json"]["key"], "value")

    @patch("urllib.request.urlopen")
    def test_get_returns_status_code(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"created", 201, "text/plain")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/"})
        data = json.loads(result)
        self.assertEqual(data["status"], 201)

    @patch("urllib.request.urlopen")
    def test_get_has_headers(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/"})
        data = json.loads(result)
        self.assertIn("headers", data)


class TestHTTPToolPOST(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_post_with_json_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b'{"id": 1}', 201, "application/json")
        result = self.tool.execute("http_request", {
            "method": "POST",
            "url": "https://api.example.com/items",
            "body": {"name": "test"},
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 201)
        self.assertEqual(data["json"]["id"], 1)

    @patch("urllib.request.urlopen")
    def test_post_with_body_text(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = self.tool.execute("http_request", {
            "method": "POST",
            "url": "https://api.example.com/raw",
            "body_text": "raw payload",
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 200)

    @patch("urllib.request.urlopen")
    def test_post_no_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = self.tool.execute("http_request", {
            "method": "POST",
            "url": "https://api.example.com/trigger",
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 200)

    @patch("urllib.request.urlopen")
    def test_put_with_json_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b'{"updated": true}', 200, "application/json")
        result = self.tool.execute("http_request", {
            "method": "PUT",
            "url": "https://api.example.com/items/1",
            "body": {"name": "updated"},
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 200)
        self.assertTrue(data["json"]["updated"])

    @patch("urllib.request.urlopen")
    def test_patch_with_json_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b'{"ok": true}', 200, "application/json")
        result = self.tool.execute("http_request", {
            "method": "PATCH",
            "url": "https://api.example.com/items/1",
            "body": {"field": "value"},
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 200)


class TestHTTPToolDELETE(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_delete_request(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"", 204, "text/plain")
        result = self.tool.execute("http_request", {
            "method": "DELETE",
            "url": "https://api.example.com/items/1",
        })
        data = json.loads(result)
        self.assertEqual(data["status"], 204)


class TestHTTPToolHeaders(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_custom_headers_passed(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = self.tool.execute("http_request", {
            "method": "GET",
            "url": "https://api.example.com/protected",
            "headers": {"Authorization": "Bearer sk-test"},
        })
        # Verify the request was made (headers checked via Request object)
        self.assertTrue(mock_urlopen.called)
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test")

    @patch("urllib.request.urlopen")
    def test_default_headers_merged(self, mock_urlopen):
        tool = HTTPTool(default_headers={"X-Api-Version": "2"})
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = tool.execute("http_request", {
            "method": "GET",
            "url": "https://api.example.com/",
        })
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("X-api-version"), "2")

    @patch("urllib.request.urlopen")
    def test_extra_headers_override_defaults(self, mock_urlopen):
        tool = HTTPTool(default_headers={"X-Token": "default"})
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        result = tool.execute("http_request", {
            "method": "GET",
            "url": "https://api.example.com/",
            "headers": {"X-Token": "override"},
        })
        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("X-token"), "override")


class TestHTTPToolErrors(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_http_404_returned_as_status(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://api.example.com/missing",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(items=lambda: []),
            fp=io.BytesIO(b"not found"),
        )
        err.headers = {"Content-Type": "text/plain"}
        mock_urlopen.side_effect = err
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/missing"})
        data = json.loads(result)
        self.assertEqual(data["status"], 404)

    @patch("urllib.request.urlopen")
    def test_http_500_returned_as_status(self, mock_urlopen):
        err = urllib.error.HTTPError(
            url="https://api.example.com/crash",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(items=lambda: []),
            fp=io.BytesIO(b"server error"),
        )
        err.headers = {"Content-Type": "text/plain"}
        mock_urlopen.side_effect = err
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://api.example.com/crash"})
        data = json.loads(result)
        self.assertEqual(data["status"], 500)

    @patch("urllib.request.urlopen")
    def test_url_error_returns_error_json(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("name or service not known")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://unreachable.invalid/"})
        data = json.loads(result)
        self.assertIn("error", data)

    @patch("urllib.request.urlopen")
    def test_timeout_returns_error_json(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        result = self.tool.execute("http_request", {"method": "GET", "url": "https://slow.example.com/"})
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("Timeout", data["error"])


class TestHTTPToolJSONBody(unittest.TestCase):
    def setUp(self):
        self.tool = HTTPTool()

    @patch("urllib.request.urlopen")
    def test_json_body_sets_content_type(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b'{}', 200, "application/json")
        self.tool.execute("http_request", {
            "method": "POST",
            "url": "https://api.example.com/",
            "body": {"key": "value"},
        })
        req = mock_urlopen.call_args[0][0]
        self.assertIn("application/json", req.get_header("Content-type"))

    @patch("urllib.request.urlopen")
    def test_json_body_is_encoded(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b'{}', 200, "application/json")
        self.tool.execute("http_request", {
            "method": "POST",
            "url": "https://api.example.com/",
            "body": {"number": 42},
        })
        req = mock_urlopen.call_args[0][0]
        body_bytes = req.data
        body_dict = json.loads(body_bytes.decode())
        self.assertEqual(body_dict["number"], 42)

    @patch("urllib.request.urlopen")
    def test_get_does_not_send_body(self, mock_urlopen):
        mock_urlopen.return_value = _make_response(b"ok", 200, "text/plain")
        self.tool.execute("http_request", {
            "method": "GET",
            "url": "https://api.example.com/",
            "body": {"ignored": True},
        })
        req = mock_urlopen.call_args[0][0]
        # GET should not have a body
        self.assertIsNone(req.data)


class TestHTTPToolImport(unittest.TestCase):
    def test_importable_from_agent_friend(self):
        from agent_friend import HTTPTool
        self.assertTrue(callable(HTTPTool))

    def test_in_tools_init(self):
        from agent_friend.tools import HTTPTool
        self.assertTrue(callable(HTTPTool))


if __name__ == "__main__":
    unittest.main()
