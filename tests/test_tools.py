"""Tests for agent-friend tools: MemoryTool, CodeTool, SearchTool, BrowserTool."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.tools.memory import MemoryTool
from agent_friend.tools.code import CodeTool
from agent_friend.tools.search import SearchTool
from agent_friend.tools.browser import BrowserTool


# ---------------------------------------------------------------------------
# MemoryTool tests
# ---------------------------------------------------------------------------

class TestMemoryTool(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_memory.db")
        self.memory = MemoryTool(db_path=self.db_path)

    def test_name(self):
        self.assertEqual(self.memory.name, "memory")

    def test_definitions_count(self):
        defs = self.memory.definitions()
        self.assertEqual(len(defs), 3)

    def test_definitions_names(self):
        names = {d["name"] for d in self.memory.definitions()}
        self.assertIn("remember", names)
        self.assertIn("recall", names)
        self.assertIn("forget", names)

    def test_remember_stores_fact(self):
        result = self.memory.execute("remember", {"key": "user_name", "value": "Alice"})
        self.assertIn("user_name", result)

    def test_remember_returns_confirmation(self):
        result = self.memory.execute("remember", {"key": "color", "value": "blue"})
        self.assertIn("Remembered", result)

    def test_recall_finds_stored_fact(self):
        self.memory.execute("remember", {"key": "city", "value": "New York"})
        result = self.memory.execute("recall", {"query": "New York"})
        self.assertIn("New York", result)

    def test_recall_no_results_message(self):
        result = self.memory.execute("recall", {"query": "xyzzy_nonexistent"})
        self.assertIn("No memories", result)

    def test_forget_removes_fact(self):
        self.memory.execute("remember", {"key": "temp_key", "value": "temp_value"})
        result = self.memory.execute("forget", {"key": "temp_key"})
        self.assertIn("Forgot", result)
        # Verify it's gone
        recall_result = self.memory.execute("recall", {"query": "temp_value"})
        self.assertIn("No memories", recall_result)

    def test_forget_nonexistent_key(self):
        result = self.memory.execute("forget", {"key": "does_not_exist"})
        self.assertIn("No memory found", result)

    def test_remember_overwrites_existing_key(self):
        self.memory.execute("remember", {"key": "color", "value": "blue"})
        self.memory.execute("remember", {"key": "color", "value": "red"})
        result = self.memory.execute("recall", {"query": "color"})
        self.assertIn("red", result)

    def test_unknown_tool_name(self):
        result = self.memory.execute("invalid_tool", {})
        self.assertIn("Unknown", result)

    def test_definitions_have_input_schema(self):
        for definition in self.memory.definitions():
            self.assertIn("input_schema", definition)
            self.assertIn("required", definition["input_schema"])

    def test_recall_multiple_facts(self):
        self.memory.execute("remember", {"key": "fact1", "value": "Python is a language"})
        self.memory.execute("remember", {"key": "fact2", "value": "Python was created by Guido"})
        result = self.memory.execute("recall", {"query": "Python"})
        self.assertIn("Python", result)


# ---------------------------------------------------------------------------
# CodeTool tests
# ---------------------------------------------------------------------------

class TestCodeTool(unittest.TestCase):
    def setUp(self):
        self.code = CodeTool(timeout_seconds=10)

    def test_name(self):
        self.assertEqual(self.code.name, "code")

    def test_definitions_count(self):
        defs = self.code.definitions()
        self.assertEqual(len(defs), 1)

    def test_definitions_has_run_code(self):
        self.assertEqual(self.code.definitions()[0]["name"], "run_code")

    def test_run_python_basic(self):
        result = self.code.execute("run_code", {"code": "print('hello world')"})
        self.assertIn("hello world", result)

    def test_run_python_math(self):
        result = self.code.execute("run_code", {"code": "print(2 + 2)"})
        self.assertIn("4", result)

    def test_run_python_multiline(self):
        code = "x = 10\ny = 20\nprint(x + y)"
        result = self.code.execute("run_code", {"code": code})
        self.assertIn("30", result)

    def test_run_python_stderr_captured(self):
        code = "import sys\nsys.stderr.write('error message\\n')"
        result = self.code.execute("run_code", {"code": code})
        self.assertIn("error message", result)

    def test_run_python_exception(self):
        result = self.code.execute("run_code", {"code": "raise ValueError('boom')"})
        self.assertIn("ValueError", result)

    def test_run_bash(self):
        result = self.code.execute("run_code", {"code": "echo 'bash works'", "language": "bash"})
        self.assertIn("bash works", result)

    def test_run_bash_exit_code(self):
        result = self.code.execute("run_code", {"code": "exit 1", "language": "bash"})
        self.assertIn("exit code: 1", result)

    def test_unsupported_language(self):
        result = self.code.execute("run_code", {"code": "code", "language": "ruby"})
        self.assertIn("Unsupported language", result)

    def test_timeout(self):
        code_tool = CodeTool(timeout_seconds=1)
        result = code_tool.execute("run_code", {"code": "import time; time.sleep(10)"})
        self.assertIn("timed out", result.lower())

    def test_no_output_message(self):
        result = self.code.execute("run_code", {"code": "x = 1 + 1"})
        self.assertIn("no output", result.lower())

    def test_unknown_tool_name(self):
        result = self.code.execute("unknown_tool", {})
        self.assertIn("Unknown", result)

    def test_default_language_is_python(self):
        result = self.code.execute("run_code", {"code": "print('default lang')"})
        self.assertIn("default lang", result)


# ---------------------------------------------------------------------------
# SearchTool tests (mock HTTP)
# ---------------------------------------------------------------------------

_MOCK_DDG_HTML = """
<html>
<body>
<div class="result results_links results_links_deep web-result">
  <div class="result__body">
    <a class="result__a" href="https://duckduckgo.com/?uddg=https%3A%2F%2Fexample.com%2Farticle">
      Example Article Title
    </a>
    <a class="result__snippet">This is the snippet for the example result.</a>
  </div>
</div>
<div class="result results_links results_links_deep web-result">
  <div class="result__body">
    <a class="result__a" href="https://duckduckgo.com/?uddg=https%3A%2F%2Fanother.com%2Fpage">
      Another Result Title
    </a>
    <a class="result__snippet">Another snippet here.</a>
  </div>
</div>
</body>
</html>
"""


class TestSearchTool(unittest.TestCase):
    def setUp(self):
        self.search = SearchTool(max_results=5, timeout=10)

    def test_name(self):
        self.assertEqual(self.search.name, "search")

    def test_definitions_count(self):
        self.assertEqual(len(self.search.definitions()), 1)

    def test_definition_name(self):
        self.assertEqual(self.search.definitions()[0]["name"], "search")

    def test_unknown_tool_name(self):
        result = self.search.execute("wrong_name", {"query": "test"})
        self.assertIn("Unknown", result)

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_returns_results(self, mock_fetch):
        mock_fetch.return_value = _MOCK_DDG_HTML
        result = self.search.execute("search", {"query": "test query"})
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "No results found.")

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_extracts_urls(self, mock_fetch):
        mock_fetch.return_value = _MOCK_DDG_HTML
        result = self.search.execute("search", {"query": "test"})
        self.assertIn("example.com", result)

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_respects_max_results(self, mock_fetch):
        # Generate HTML with many results
        blocks = ""
        for i in range(10):
            blocks += f"""
            <div class="result results_links web-result">
              <div class="result__body">
                <a class="result__a" href="https://duckduckgo.com/?uddg=https%3A%2F%2Fsite{i}.com">
                  Title {i}
                </a>
              </div>
            </div>
            """
        mock_fetch.return_value = f"<html><body>{blocks}</body></html>"

        search = SearchTool(max_results=3)
        result = search.execute("search", {"query": "many results"})
        # Count URL occurrences — should not exceed max_results
        url_count = result.count("URL:")
        self.assertLessEqual(url_count, 3)

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_handles_empty_results(self, mock_fetch):
        mock_fetch.return_value = "<html><body><p>No results</p></body></html>"
        result = self.search.execute("search", {"query": "zzzqqqxxx"})
        self.assertEqual(result, "No results found.")

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_handles_network_error(self, mock_fetch):
        import urllib.error
        mock_fetch.side_effect = urllib.error.URLError("Network unreachable")
        result = self.search.execute("search", {"query": "test"})
        self.assertIn("Search failed", result)

    def test_definitions_have_required_query(self):
        definition = self.search.definitions()[0]
        self.assertIn("query", definition["input_schema"]["required"])

    @patch("agent_friend.tools.search.SearchTool._fetch")
    def test_search_max_results_override_in_args(self, mock_fetch):
        blocks = ""
        for i in range(10):
            blocks += f"""
            <div class="result results_links web-result">
              <div class="result__body">
                <a class="result__a" href="https://duckduckgo.com/?uddg=https%3A%2F%2Fsite{i}.com">
                  Title {i}
                </a>
              </div>
            </div>
            """
        mock_fetch.return_value = f"<html><body>{blocks}</body></html>"
        result = self.search.execute("search", {"query": "test", "max_results": 2})
        url_count = result.count("URL:")
        self.assertLessEqual(url_count, 2)


# ---------------------------------------------------------------------------
# BrowserTool tests
# ---------------------------------------------------------------------------

class TestBrowserTool(unittest.TestCase):
    def setUp(self):
        self.browser = BrowserTool()

    def test_name(self):
        self.assertEqual(self.browser.name, "browser")

    def test_definitions_count(self):
        self.assertEqual(len(self.browser.definitions()), 1)

    def test_definition_name(self):
        self.assertEqual(self.browser.definitions()[0]["name"], "browse")

    def test_unknown_tool_name(self):
        result = self.browser.execute("wrong_name", {"url": "https://example.com"})
        self.assertIn("Unknown", result)

    @patch("agent_friend.tools.browser.BrowserTool._agent_browser_available")
    def test_returns_error_when_not_installed(self, mock_available):
        mock_available.return_value = False
        result = self.browser.execute("browse", {"url": "https://example.com"})
        self.assertIn("agent-browser is not installed", result)

    @patch("agent_friend.tools.browser.BrowserTool._agent_browser_available")
    @patch("subprocess.run")
    def test_browse_calls_agent_browser(self, mock_run, mock_available):
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"text": "Page content here"}',
            stderr="",
        )
        result = self.browser.execute("browse", {"url": "https://example.com"})
        self.assertIn("Page content here", result)

    @patch("agent_friend.tools.browser.BrowserTool._agent_browser_available")
    @patch("subprocess.run")
    def test_browse_handles_open_failure(self, mock_run, mock_available):
        mock_available.return_value = True
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Failed to open"
        )
        result = self.browser.execute("browse", {"url": "https://example.com"})
        self.assertIn("failed", result.lower())

    @patch("agent_friend.tools.browser.BrowserTool._agent_browser_available")
    @patch("subprocess.run")
    def test_browse_handles_timeout(self, mock_run, mock_available):
        import subprocess
        mock_available.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="agent-browser", timeout=30)
        result = self.browser.execute("browse", {"url": "https://example.com"})
        self.assertIn("timed out", result.lower())

    @patch("agent_friend.tools.browser.BrowserTool._agent_browser_available")
    @patch("subprocess.run")
    def test_browse_handles_non_json_snapshot(self, mock_run, mock_available):
        mock_available.return_value = True

        def side_effect(args, **kwargs):
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            if "open" in args:
                mock_result.stdout = ""
            else:
                mock_result.stdout = "Plain text content, not JSON"
            return mock_result

        mock_run.side_effect = side_effect
        result = self.browser.execute("browse", {"url": "https://example.com"})
        self.assertIn("Plain text content", result)

    def test_agent_browser_available_returns_bool(self):
        result = self.browser._agent_browser_available()
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
