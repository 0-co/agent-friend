"""Tests for FetchTool — URL fetching."""

import unittest
from unittest.mock import patch, MagicMock
import io
import urllib.error

from agent_friend.tools.fetch import FetchTool, _TextExtractor


class TestFetchToolInit(unittest.TestCase):
    def test_name(self):
        self.assertEqual(FetchTool().name, "fetch")

    def test_description_nonempty(self):
        self.assertTrue(len(FetchTool().description) > 0)

    def test_definitions_count(self):
        self.assertEqual(len(FetchTool().definitions()), 1)

    def test_definition_name(self):
        self.assertEqual(FetchTool().definitions()[0]["name"], "fetch")

    def test_definition_has_input_schema(self):
        defn = FetchTool().definitions()[0]
        self.assertIn("input_schema", defn)
        self.assertIn("url", defn["input_schema"]["properties"])

    def test_definition_url_required(self):
        defn = FetchTool().definitions()[0]
        self.assertIn("url", defn["input_schema"]["required"])

    def test_custom_timeout(self):
        tool = FetchTool(timeout=30)
        self.assertEqual(tool.timeout, 30)

    def test_custom_max_chars(self):
        tool = FetchTool(max_chars=1000)
        self.assertEqual(tool.max_chars, 1000)


class TestFetchToolValidation(unittest.TestCase):
    def setUp(self):
        self.tool = FetchTool()

    def test_unknown_tool_name(self):
        result = self.tool.execute("other", {"url": "https://example.com"})
        self.assertIn("Unknown", result)

    def test_missing_url(self):
        result = self.tool.execute("fetch", {})
        self.assertIn("Error", result)
        self.assertIn("url", result)

    def test_empty_url(self):
        result = self.tool.execute("fetch", {"url": ""})
        self.assertIn("Error", result)

    def test_non_http_url(self):
        result = self.tool.execute("fetch", {"url": "ftp://example.com/file"})
        self.assertIn("Error", result)
        self.assertIn("http", result)

    def test_file_url_rejected(self):
        result = self.tool.execute("fetch", {"url": "file:///etc/passwd"})
        self.assertIn("Error", result)


class TestFetchToolNetworkErrors(unittest.TestCase):
    def setUp(self):
        self.tool = FetchTool(timeout=5)

    @patch("urllib.request.urlopen")
    def test_http_error_404(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com", 404, "Not Found", {}, None
        )
        result = self.tool.execute("fetch", {"url": "https://example.com/missing"})
        self.assertIn("404", result)

    @patch("urllib.request.urlopen")
    def test_url_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("Name or service not known")
        result = self.tool.execute("fetch", {"url": "https://nonexistent.invalid/"})
        self.assertIn("Could not fetch", result)

    @patch("urllib.request.urlopen")
    def test_timeout_error(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        result = self.tool.execute("fetch", {"url": "https://example.com"})
        self.assertIn("Timeout", result)


class TestFetchToolSuccess(unittest.TestCase):
    def setUp(self):
        self.tool = FetchTool()

    def _make_response(self, body: str, content_type: str = "text/plain"):
        mock_resp = MagicMock()
        mock_resp.headers = {"Content-Type": content_type}
        mock_resp.read.return_value = body.encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_plain_text_returned(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response("Hello, world!", "text/plain")
        result = self.tool.execute("fetch", {"url": "https://example.com/text"})
        self.assertIn("Hello, world!", result)

    @patch("urllib.request.urlopen")
    def test_html_stripped_to_text(self, mock_urlopen):
        html = "<html><body><p>Article content here.</p><script>evil()</script></body></html>"
        mock_urlopen.return_value = self._make_response(html, "text/html")
        result = self.tool.execute("fetch", {"url": "https://example.com/"})
        self.assertIn("Article content here.", result)
        self.assertNotIn("evil()", result)
        self.assertNotIn("<p>", result)

    @patch("urllib.request.urlopen")
    def test_script_tags_removed(self, mock_urlopen):
        html = "<html><body><script>alert('xss')</script><p>Safe text.</p></body></html>"
        mock_urlopen.return_value = self._make_response(html, "text/html")
        result = self.tool.execute("fetch", {"url": "https://example.com/"})
        self.assertNotIn("alert", result)
        self.assertIn("Safe text.", result)

    @patch("urllib.request.urlopen")
    def test_style_tags_removed(self, mock_urlopen):
        html = "<html><head><style>body { color: red; }</style></head><body><p>Content.</p></body></html>"
        mock_urlopen.return_value = self._make_response(html, "text/html")
        result = self.tool.execute("fetch", {"url": "https://example.com/"})
        self.assertNotIn("color: red", result)
        self.assertIn("Content.", result)

    @patch("urllib.request.urlopen")
    def test_max_chars_truncates(self, mock_urlopen):
        long_text = "x" * 20000
        mock_urlopen.return_value = self._make_response(long_text, "text/plain")
        tool = FetchTool(max_chars=100)
        result = tool.execute("fetch", {"url": "https://example.com/"})
        self.assertIn("truncated", result)
        self.assertLess(len(result), 500)  # well under 20000

    @patch("urllib.request.urlopen")
    def test_max_chars_argument_overrides_default(self, mock_urlopen):
        long_text = "y" * 5000
        mock_urlopen.return_value = self._make_response(long_text, "text/plain")
        result = self.tool.execute("fetch", {"url": "https://example.com/", "max_chars": 50})
        self.assertIn("truncated", result)

    @patch("urllib.request.urlopen")
    def test_empty_page_message(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response("", "text/plain")
        result = self.tool.execute("fetch", {"url": "https://example.com/"})
        self.assertIn("no text content", result)

    @patch("urllib.request.urlopen")
    def test_doctype_html_detected(self, mock_urlopen):
        html = "<!DOCTYPE html><html><body><p>Doctype page.</p></body></html>"
        # content-type is plain/text but starts with <!DOCTYPE
        mock_urlopen.return_value = self._make_response(html, "text/plain")
        result = self.tool.execute("fetch", {"url": "https://example.com/"})
        self.assertIn("Doctype page.", result)


class TestTextExtractor(unittest.TestCase):
    def _extract(self, html: str) -> str:
        e = _TextExtractor()
        e.feed(html)
        return e.get_text()

    def test_plain_text_preserved(self):
        self.assertIn("hello", self._extract("<p>hello</p>"))

    def test_script_skipped(self):
        result = self._extract("<script>bad()</script><p>good</p>")
        self.assertNotIn("bad", result)
        self.assertIn("good", result)

    def test_style_skipped(self):
        result = self._extract("<style>.cls{color:red}</style><p>text</p>")
        self.assertNotIn("color", result)
        self.assertIn("text", result)

    def test_nested_skip_tags(self):
        result = self._extract("<script><script>inner</script>outer</script><p>visible</p>")
        self.assertNotIn("inner", result)
        self.assertIn("visible", result)

    def test_nav_skipped(self):
        result = self._extract("<nav>Menu Item</nav><main>Article</main>")
        self.assertNotIn("Menu Item", result)
        self.assertIn("Article", result)

    def test_empty_html(self):
        result = self._extract("")
        self.assertEqual(result, "")

    def test_malformed_html_no_crash(self):
        # Should not raise
        e = _TextExtractor()
        e.feed("<p>unclosed")
        result = e.get_text()
        self.assertIn("unclosed", result)


if __name__ == "__main__":
    unittest.main()
