"""Tests for RSSFeedTool — RSS/Atom feed reader."""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_friend.tools.rss import RSSFeedTool, _strip_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RSS_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Blog</title>
    <link>https://example.com</link>
    <item>
      <title>First Post</title>
      <link>https://example.com/first</link>
      <description>First post content</description>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/second</link>
      <description><![CDATA[<p>Second post</p>]]></description>
    </item>
    <item>
      <title>Third Post</title>
      <link>https://example.com/third</link>
      <description>Third post content</description>
    </item>
  </channel>
</rss>"""

ATOM_SAMPLE = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test Feed</title>
  <entry>
    <title>Atom Entry 1</title>
    <link href="https://example.com/atom/1"/>
    <summary>Atom entry one summary</summary>
  </entry>
  <entry>
    <title>Atom Entry 2</title>
    <link href="https://example.com/atom/2"/>
    <summary>Atom entry two summary</summary>
  </entry>
</feed>"""

MALFORMED_XML = b"<not valid xml <<"


def _mock_urlopen(content: bytes, content_type: str = "application/rss+xml"):
    """Return a mock context manager that yields `content`."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = content
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# _strip_html helper
# ---------------------------------------------------------------------------

class TestStripHtml(unittest.TestCase):
    def test_removes_tags(self):
        self.assertEqual(_strip_html("<p>Hello</p>"), "Hello")

    def test_removes_anchor(self):
        result = _strip_html('<a href="https://example.com">Click</a>')
        self.assertEqual(result, "Click")

    def test_decodes_amp(self):
        self.assertEqual(_strip_html("A &amp; B"), "A & B")

    def test_decodes_lt_gt(self):
        self.assertEqual(_strip_html("&lt;tag&gt;"), "<tag>")

    def test_decodes_nbsp(self):
        result = _strip_html("a&nbsp;b")
        self.assertEqual(result, "a b")

    def test_collapses_whitespace(self):
        self.assertEqual(_strip_html("a   b\n c"), "a b c")

    def test_empty_string(self):
        self.assertEqual(_strip_html(""), "")

    def test_plain_text_unchanged(self):
        self.assertEqual(_strip_html("plain text"), "plain text")


# ---------------------------------------------------------------------------
# Initialization and basics
# ---------------------------------------------------------------------------

class TestRSSFeedToolInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))

    def test_name(self):
        self.assertEqual(self.tool.name, "rss")

    def test_description_nonempty(self):
        self.assertTrue(len(self.tool.description) > 0)

    def test_definitions_count(self):
        self.assertEqual(len(self.tool.definitions()), 5)

    def test_definition_names(self):
        names = {d["name"] for d in self.tool.definitions()}
        self.assertIn("fetch_feed", names)
        self.assertIn("subscribe", names)
        self.assertIn("list_feeds", names)
        self.assertIn("read_feed", names)
        self.assertIn("unsubscribe", names)

    def test_db_created(self):
        self.assertTrue(os.path.exists(self.tool.db_path))

    def test_unknown_tool_name(self):
        result = self.tool.execute("nonexistent", {})
        self.assertIn("Unknown", result)


# ---------------------------------------------------------------------------
# Subscribe / unsubscribe / list
# ---------------------------------------------------------------------------

class TestRSSSubscribe(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))

    def test_subscribe_success(self):
        result = self.tool.execute("subscribe", {"url": "https://example.com/rss", "name": "ex"})
        self.assertIn("ex", result)

    def test_subscribe_appears_in_list(self):
        self.tool.execute("subscribe", {"url": "https://example.com/rss", "name": "blog"})
        result = self.tool.execute("list_feeds", {})
        self.assertIn("blog", result)
        self.assertIn("https://example.com/rss", result)

    def test_subscribe_overwrite(self):
        self.tool.execute("subscribe", {"url": "https://old.com/rss", "name": "feed"})
        self.tool.execute("subscribe", {"url": "https://new.com/rss", "name": "feed"})
        result = self.tool.execute("list_feeds", {})
        self.assertIn("https://new.com/rss", result)
        self.assertNotIn("https://old.com/rss", result)

    def test_list_empty(self):
        result = self.tool.execute("list_feeds", {})
        self.assertIn("No subscribed", result)

    def test_unsubscribe_success(self):
        self.tool.execute("subscribe", {"url": "https://example.com/rss", "name": "ex"})
        result = self.tool.execute("unsubscribe", {"name": "ex"})
        self.assertIn("Unsubscribed", result)
        listing = self.tool.execute("list_feeds", {})
        self.assertNotIn("ex", listing)

    def test_unsubscribe_nonexistent(self):
        result = self.tool.execute("unsubscribe", {"name": "nope"})
        self.assertIn("No feed", result)

    def test_subscribe_missing_url(self):
        result = self.tool.execute("subscribe", {"name": "test"})
        self.assertIn("Error", result)

    def test_subscribe_missing_name(self):
        result = self.tool.execute("subscribe", {"url": "https://example.com/rss"})
        self.assertIn("Error", result)

    def test_multiple_feeds_listed(self):
        self.tool.execute("subscribe", {"url": "https://a.com/rss", "name": "a"})
        self.tool.execute("subscribe", {"url": "https://b.com/rss", "name": "b"})
        result = self.tool.execute("list_feeds", {})
        self.assertIn("a", result)
        self.assertIn("b", result)


# ---------------------------------------------------------------------------
# RSS 2.0 parsing
# ---------------------------------------------------------------------------

class TestRSSParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))

    @patch("urllib.request.urlopen")
    def test_rss_returns_items(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss", "count": 3})
        self.assertIn("First Post", result)
        self.assertIn("Second Post", result)
        self.assertIn("Third Post", result)

    @patch("urllib.request.urlopen")
    def test_rss_count_limits_items(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss", "count": 1})
        self.assertIn("First Post", result)
        self.assertNotIn("Second Post", result)

    @patch("urllib.request.urlopen")
    def test_rss_count_max_20(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss", "count": 100})
        # Should not crash and should return all 3 available items
        self.assertIn("First Post", result)

    @patch("urllib.request.urlopen")
    def test_rss_links_included(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss", "count": 1})
        self.assertIn("https://example.com/first", result)

    @patch("urllib.request.urlopen")
    def test_rss_html_stripped_from_description(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss", "count": 2})
        self.assertNotIn("<p>", result)
        self.assertIn("Second post", result)

    @patch("urllib.request.urlopen")
    def test_rss_default_count_five(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss"})
        # Should not crash with default count
        self.assertIn("First Post", result)


# ---------------------------------------------------------------------------
# Atom parsing
# ---------------------------------------------------------------------------

class TestAtomParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))

    @patch("urllib.request.urlopen")
    def test_atom_entries_returned(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(ATOM_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/atom", "count": 5})
        self.assertIn("Atom Entry 1", result)
        self.assertIn("Atom Entry 2", result)

    @patch("urllib.request.urlopen")
    def test_atom_links_included(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(ATOM_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/atom", "count": 5})
        self.assertIn("https://example.com/atom/1", result)

    @patch("urllib.request.urlopen")
    def test_atom_summary_included(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(ATOM_SAMPLE)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/atom", "count": 1})
        self.assertIn("Atom entry one summary", result)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestRSSErrorHandling(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))

    @patch("urllib.request.urlopen")
    def test_url_error_reported(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("DNS failure")
        result = self.tool.execute("fetch_feed", {"url": "https://notreal.invalid/rss"})
        self.assertIn("Error", result)

    @patch("urllib.request.urlopen")
    def test_http_error_reported(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            "https://example.com/rss", 404, "Not Found", {}, None
        )
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss"})
        self.assertIn("Error", result)

    @patch("urllib.request.urlopen")
    def test_malformed_xml_reported(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(MALFORMED_XML)
        result = self.tool.execute("fetch_feed", {"url": "https://example.com/rss"})
        self.assertIn("Error", result)

    def test_read_nonexistent_feed_by_name(self):
        result = self.tool.execute("read_feed", {"name": "doesnotexist"})
        self.assertIn("No feed", result)


# ---------------------------------------------------------------------------
# read_feed (by name)
# ---------------------------------------------------------------------------

class TestReadFeedByName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tool = RSSFeedTool(db_path=os.path.join(self.tmp, "feeds.db"))
        self.tool.execute("subscribe", {"url": "https://example.com/rss", "name": "test"})

    @patch("urllib.request.urlopen")
    def test_read_by_name_returns_items(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("read_feed", {"name": "test", "count": 2})
        self.assertIn("First Post", result)

    @patch("urllib.request.urlopen")
    def test_read_by_name_shows_label(self, mock_urlopen):
        mock_urlopen.return_value = _mock_urlopen(RSS_SAMPLE)
        result = self.tool.execute("read_feed", {"name": "test"})
        self.assertIn("test", result)


if __name__ == "__main__":
    unittest.main()
