"""rss.py — RSS/Atom feed reader for agent-friend (no required dependencies)."""

import os
import re
import sqlite3
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTool


def _expand(path: str) -> str:
    return str(Path(path).expanduser())


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


class RSSFeedTool(BaseTool):
    """Subscribe to and read RSS/Atom feeds.

    Maintains a local list of subscribed feeds in SQLite. Fetches and parses
    feeds using only the standard library (``urllib`` + ``xml.etree``).

    Parameters
    ----------
    db_path: Path to the SQLite database for storing subscribed feeds.
             Defaults to ``~/.agent_friend/feeds.db``.
    timeout: HTTP request timeout in seconds. Defaults to 15.
    """

    def __init__(
        self,
        db_path: str = "~/.agent_friend/feeds.db",
        timeout: int = 15,
    ) -> None:
        self.db_path = _expand(db_path)
        self.timeout = timeout
        self._ensure_db()

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rss"

    @property
    def description(self) -> str:
        return (
            "Subscribe to RSS/Atom feeds and read the latest items. "
            "No API key required."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "fetch_feed",
                "description": (
                    "Fetch the latest items from an RSS or Atom feed URL. "
                    "Returns titles, links, and summaries."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The RSS/Atom feed URL.",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of items to return (default 5, max 20).",
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "subscribe",
                "description": "Save a feed URL under a short name for quick future access.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Feed URL."},
                        "name": {
                            "type": "string",
                            "description": "Short name to identify this feed (e.g. 'hn', 'arxiv-ai').",
                        },
                    },
                    "required": ["url", "name"],
                },
            },
            {
                "name": "list_feeds",
                "description": "List all subscribed feed names and their URLs.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "read_feed",
                "description": (
                    "Fetch the latest items from a subscribed feed by name. "
                    "Use list_feeds to see available names."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Subscribed feed name.",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of items to return (default 5, max 20).",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "unsubscribe",
                "description": "Remove a subscribed feed by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Feed name to remove."},
                    },
                    "required": ["name"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "fetch_feed":
            url = arguments.get("url", "")
            count = min(int(arguments.get("count", 5)), 20)
            return self._fetch_and_format(url, count)

        if tool_name == "subscribe":
            url = arguments.get("url", "")
            name = arguments.get("name", "").strip()
            if not url or not name:
                return "Error: url and name are required."
            return self._subscribe(url, name)

        if tool_name == "list_feeds":
            return self._list_feeds()

        if tool_name == "read_feed":
            name = arguments.get("name", "").strip()
            count = min(int(arguments.get("count", 5)), 20)
            return self._read_by_name(name, count)

        if tool_name == "unsubscribe":
            name = arguments.get("name", "").strip()
            return self._unsubscribe(name)

        return f"Unknown RSS tool: {tool_name}"

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _ensure_db(self) -> None:
        """Create the SQLite database and feeds table if needed."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feeds (
                    name TEXT PRIMARY KEY,
                    url  TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _subscribe(self, url: str, name: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO feeds (name, url) VALUES (?, ?)",
                (name, url),
            )
            conn.commit()
        return f"Subscribed to '{name}': {url}"

    def _unsubscribe(self, name: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("DELETE FROM feeds WHERE name = ?", (name,))
            conn.commit()
        if cur.rowcount:
            return f"Unsubscribed from '{name}'."
        return f"No feed named '{name}' found."

    def _list_feeds(self) -> str:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT name, url FROM feeds ORDER BY name").fetchall()
        if not rows:
            return "No subscribed feeds. Use subscribe(url, name) to add one."
        lines = ["Subscribed feeds:"]
        for name, url in rows:
            lines.append(f"  {name}: {url}")
        return "\n".join(lines)

    def _get_feed_url(self, name: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT url FROM feeds WHERE name = ?", (name,)
            ).fetchone()
        return row[0] if row else None

    def _read_by_name(self, name: str, count: int) -> str:
        url = self._get_feed_url(name)
        if not url:
            return f"No feed named '{name}'. Use list_feeds to see subscribed feeds."
        return self._fetch_and_format(url, count, label=name)

    def _fetch_and_format(self, url: str, count: int, label: str = "") -> str:
        items, error = self._fetch_items(url, count)
        if error:
            return f"Error fetching feed: {error}"
        if not items:
            return "Feed fetched but no items found."

        header = f"Feed: {label or url} ({len(items)} items)\n"
        lines = [header]
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {item['title']}")
            if item.get("link"):
                lines.append(f"   URL: {item['link']}")
            if item.get("summary"):
                summary = _strip_html(item["summary"])[:200]
                lines.append(f"   {summary}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _fetch_items(
        self, url: str, count: int
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Fetch and parse an RSS or Atom feed. Returns (items, error)."""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "agent-friend/0.6 (RSS reader)"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            return [], str(exc)
        except Exception as exc:
            return [], str(exc)

        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            return [], f"XML parse error: {exc}"

        ns = self._detect_namespaces(root)

        # RSS 2.0
        channel = root.find("channel")
        if channel is not None:
            return self._parse_rss(channel, count, ns), None

        # Atom
        atom_ns = "http://www.w3.org/2005/Atom"
        if root.tag == f"{{{atom_ns}}}feed" or root.tag == "feed":
            return self._parse_atom(root, count, ns), None

        # RSS 1.0 / RDF
        rdf_item_tag = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}item"
        items_rdf = root.findall(f"{{http://purl.org/rss/1.0/}}item")
        if items_rdf:
            return self._parse_rss1(items_rdf, count), None

        return [], "Unrecognised feed format."

    def _detect_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Return commonly used namespace URIs."""
        return {
            "content": "http://purl.org/rss/1.0/modules/content/",
            "dc": "http://purl.org/dc/elements/1.1/",
            "atom": "http://www.w3.org/2005/Atom",
        }

    def _text(self, el: Optional[ET.Element]) -> str:
        if el is None:
            return ""
        return (el.text or "").strip()

    def _parse_rss(
        self, channel: ET.Element, count: int, ns: Dict[str, str]
    ) -> List[Dict[str, str]]:
        results = []
        for item in channel.findall("item")[:count]:
            title = self._text(item.find("title")) or "(no title)"
            link = self._text(item.find("link"))
            desc = (
                self._text(item.find("description"))
                or self._text(item.find(f"{{{ns['content']}}}encoded"))
            )
            results.append({"title": title, "link": link, "summary": desc})
        return results

    def _parse_atom(
        self, feed: ET.Element, count: int, ns: Dict[str, str]
    ) -> List[Dict[str, str]]:
        atom_ns = ns["atom"]
        results = []
        for entry in feed.findall(f"{{{atom_ns}}}entry")[:count]:
            title = self._text(entry.find(f"{{{atom_ns}}}title")) or "(no title)"
            # Atom link has href attribute
            link_el = entry.find(f"{{{atom_ns}}}link")
            link = (link_el.get("href", "") if link_el is not None else "")
            summary = self._text(entry.find(f"{{{atom_ns}}}summary")) or self._text(
                entry.find(f"{{{atom_ns}}}content")
            )
            results.append({"title": title, "link": link, "summary": summary})
        return results

    def _parse_rss1(
        self, items: List[ET.Element], count: int
    ) -> List[Dict[str, str]]:
        rss1_ns = "http://purl.org/rss/1.0/"
        results = []
        for item in items[:count]:
            title = self._text(item.find(f"{{{rss1_ns}}}title")) or "(no title)"
            link = self._text(item.find(f"{{{rss1_ns}}}link"))
            desc = self._text(item.find(f"{{{rss1_ns}}}description"))
            results.append({"title": title, "link": link, "summary": desc})
        return results
