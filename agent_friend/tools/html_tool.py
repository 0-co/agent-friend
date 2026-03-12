"""html_tool.py — HTMLTool for agent-friend (stdlib only).

Agents that fetch web pages with FetchTool get raw HTML.  HTMLTool parses
that HTML and extracts the parts agents actually need: plain text, links,
tables, headings, and meta tags.  No third-party libraries — uses the
standard-library ``html.parser``.

Usage::

    tool = HTMLTool()

    html = "<h1>Agent News</h1><p>New tool <a href='/retry'>RetryTool</a> shipped.</p>"

    tool.html_text(html)
    # "Agent News\\nNew tool RetryTool shipped."

    tool.html_links(html)
    # [{"text": "RetryTool", "href": "/retry"}]

    tool.html_headings(html)
    # [{"level": 1, "text": "Agent News"}]
"""

import html as _html_module
import json
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseTool


# ── internal parsers ──────────────────────────────────────────────────────────


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, skipping script/style blocks."""

    # Note: void elements (meta, link, br, img) must NOT be in _SKIP_TAGS because
    # html.parser never calls handle_endtag for them — they would permanently inflate _skip_depth.
    _SKIP_TAGS = frozenset(["script", "style", "noscript"])
    _BLOCK_TAGS = frozenset([
        "p", "div", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "br", "hr", "section", "article", "aside", "nav", "header",
        "footer", "main", "blockquote", "pre", "figure", "figcaption",
    ])

    def __init__(self) -> None:
        super().__init__()
        self._parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth == 0:
            char = _html_module.unescape(f"&{name};")
            self._parts.append(char)

    def handle_charref(self, name: str) -> None:
        if self._skip_depth == 0:
            char = _html_module.unescape(f"&#{name};")
            self._parts.append(char)

    def get_text(self) -> str:
        text = " ".join(self._parts)
        # Collapse multiple newlines/spaces
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()


class _LinkExtractor(HTMLParser):
    """Extract <a href> links from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._links: List[Dict[str, str]] = []
        self._current_href: Optional[str] = None
        self._current_text_parts: List[str] = []
        self._in_anchor = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            d = dict(attrs)
            self._current_href = d.get("href")
            self._current_text_parts = []
            self._in_anchor = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_anchor:
            text = " ".join(self._current_text_parts).strip()
            if self._current_href:
                self._links.append({"text": text, "href": self._current_href})
            self._in_anchor = False
            self._current_href = None
            self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_anchor:
            stripped = data.strip()
            if stripped:
                self._current_text_parts.append(stripped)

    def get_links(self) -> List[Dict[str, str]]:
        return self._links


class _HeadingExtractor(HTMLParser):
    """Extract <h1>–<h6> headings."""

    _HEADING_TAGS = frozenset(["h1", "h2", "h3", "h4", "h5", "h6"])

    def __init__(self) -> None:
        super().__init__()
        self._headings: List[Dict[str, Any]] = []
        self._current_level: Optional[int] = None
        self._current_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in self._HEADING_TAGS:
            self._current_level = int(tag[1])
            self._current_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self._HEADING_TAGS and self._current_level is not None:
            text = " ".join(self._current_parts).strip()
            if text:
                self._headings.append({"level": self._current_level, "text": text})
            self._current_level = None
            self._current_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_level is not None:
            stripped = data.strip()
            if stripped:
                self._current_parts.append(stripped)

    def get_headings(self) -> List[Dict[str, Any]]:
        return self._headings


class _MetaExtractor(HTMLParser):
    """Extract <meta> tags and <title>."""

    def __init__(self) -> None:
        super().__init__()
        self._meta: List[Dict[str, str]] = []
        self._title_parts: List[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "meta":
            d = {k: v for k, v in attrs if v is not None}
            if d:
                self._meta.append(d)
        elif tag == "title":
            self._in_title = True
            self._title_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data)

    def get_meta(self) -> List[Dict[str, str]]:
        return self._meta

    def get_title(self) -> str:
        return "".join(self._title_parts).strip()


class _TableExtractor(HTMLParser):
    """Extract HTML tables as list-of-rows."""

    def __init__(self) -> None:
        super().__init__()
        self._tables: List[List[List[str]]] = []
        self._current_table: Optional[List[List[str]]] = None
        self._current_row: Optional[List[str]] = None
        self._current_cell_parts: List[str] = []
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "table":
            self._current_table = []
        elif tag == "tr" and self._current_table is not None:
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._in_cell = True
            self._current_cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self._current_table is not None:
            self._tables.append(self._current_table)
            self._current_table = None
        elif tag == "tr" and self._current_table is not None and self._current_row is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._in_cell:
            cell_text = " ".join(self._current_cell_parts).strip()
            if self._current_row is not None:
                self._current_row.append(cell_text)
            self._in_cell = False
            self._current_cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            stripped = data.strip()
            if stripped:
                self._current_cell_parts.append(stripped)

    def get_tables(self) -> List[List[List[str]]]:
        return self._tables


# ── HTMLTool ──────────────────────────────────────────────────────────────────


class HTMLTool(BaseTool):
    """Parse HTML and extract text, links, headings, tables, and meta tags.

    Designed to complement FetchTool: fetch the page, then extract what
    the agent actually needs.  All stdlib — no BeautifulSoup or lxml required.

    Parameters
    ----------
    max_text_chars:
        Maximum characters returned by ``html_text``.  Avoids flooding the
        LLM context with enormous pages.  Default 20 000.
    """

    def __init__(self, max_text_chars: int = 20_000) -> None:
        self.max_text_chars = max_text_chars

    # ── public API ────────────────────────────────────────────────────────

    def html_text(self, html: str, max_chars: Optional[int] = None) -> str:
        """Extract visible text from *html*.

        Returns plain text with headings and paragraphs separated by newlines.
        """
        extractor = _TextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        text = extractor.get_text()
        limit = max_chars if max_chars is not None else self.max_text_chars
        if len(text) > limit:
            text = text[:limit] + f"\n...[truncated at {limit} chars]"
        return text

    def html_links(self, html: str, base_url: str = "") -> List[Dict[str, str]]:
        """Return list of {text, href} dicts for every <a href> in *html*.

        Parameters
        ----------
        base_url:
            When provided, relative hrefs are resolved against this URL.
        """
        extractor = _LinkExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        links = extractor.get_links()
        if base_url:
            resolved = []
            for link in links:
                href = link["href"]
                if href.startswith("http://") or href.startswith("https://"):
                    resolved.append(link)
                elif href.startswith("/"):
                    # Absolute path — prepend scheme+host
                    from urllib.parse import urlparse
                    parsed = urlparse(base_url)
                    full = f"{parsed.scheme}://{parsed.netloc}{href}"
                    resolved.append({**link, "href": full})
                else:
                    # Relative path
                    base = base_url.rstrip("/")
                    resolved.append({**link, "href": f"{base}/{href}"})
            return resolved
        return links

    def html_headings(self, html: str) -> List[Dict[str, Any]]:
        """Return list of {level: int, text: str} for all <h1>–<h6> in *html*."""
        extractor = _HeadingExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        return extractor.get_headings()

    def html_meta(self, html: str) -> Dict[str, Any]:
        """Return page title and meta tags.

        Returns dict with:
          - ``title``: page title string
          - ``meta``: list of {name/property/charset: value} dicts
        """
        extractor = _MetaExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        return {"title": extractor.get_title(), "meta": extractor.get_meta()}

    def html_tables(self, html: str) -> List[List[List[str]]]:
        """Extract all <table> elements as list-of-rows, each row a list of cell strings."""
        extractor = _TableExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        return extractor.get_tables()

    def html_select(self, html: str, tag: str, attrs: Optional[Dict[str, str]] = None) -> List[str]:
        """Find all occurrences of *tag* matching optional *attrs* and return their text content.

        This is a simple tag-based selector — not a full CSS selector engine.
        Useful for extracting specific repeated elements like `<code>` blocks,
        `<article>` summaries, or `<span class="price">` nodes.

        Parameters
        ----------
        tag:    HTML tag name (e.g. ``"code"``, ``"span"``, ``"article"``).
        attrs:  Dict of attribute name → value to match (e.g. ``{"class": "price"}``).
        """
        attrs = attrs or {}

        class _Selector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results: List[str] = []
                self._depth = 0
                self._target_depth: Optional[int] = None
                self._parts: List[str] = []

            def handle_starttag(self, t, a):
                self._depth += 1
                if t == tag and self._target_depth is None:
                    d = dict(a)
                    if all(d.get(k) == v for k, v in attrs.items()):
                        self._target_depth = self._depth
                        self._parts = []

            def handle_endtag(self, t):
                if self._target_depth is not None and self._depth == self._target_depth and t == tag:
                    self.results.append(" ".join(self._parts).strip())
                    self._target_depth = None
                    self._parts = []
                self._depth -= 1

            def handle_data(self, data):
                if self._target_depth is not None:
                    s = data.strip()
                    if s:
                        self._parts.append(s)

        sel = _Selector()
        try:
            sel.feed(html)
        except Exception:
            pass
        return sel.results

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "html"

    @property
    def description(self) -> str:
        return (
            "Parse HTML and extract text, links, headings, tables, and meta tags. "
            "Use after FetchTool or HTTPTool to turn raw HTML into readable content. "
            "All stdlib — no external dependencies."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "html_text",
                "description": (
                    "Extract all visible text from an HTML string. "
                    "Strips tags, skips <script>/<style> blocks, preserves paragraph breaks. "
                    "Use after FetchTool to make web content readable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string to parse"},
                        "max_chars": {
                            "type": "integer",
                            "description": "Max characters to return (default 20000)",
                        },
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "html_links",
                "description": (
                    "Extract all hyperlinks from HTML. "
                    "Returns list of {text, href} dicts. "
                    "Pass base_url to resolve relative links."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string"},
                        "base_url": {
                            "type": "string",
                            "description": "Base URL for resolving relative links (optional)",
                        },
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "html_headings",
                "description": (
                    "Extract all headings (h1–h6) from HTML. "
                    "Returns list of {level: int, text: str} dicts. "
                    "Useful for understanding page structure."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string"},
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "html_meta",
                "description": (
                    "Extract <title> and <meta> tags from HTML. "
                    "Returns {title: str, meta: [...]}.  "
                    "Useful for Open Graph tags, descriptions, and keywords."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string"},
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "html_tables",
                "description": (
                    "Extract all HTML tables. "
                    "Returns list of tables; each table is a list of rows; each row is a list of cell strings. "
                    "Use to extract structured tabular data from web pages."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string"},
                    },
                    "required": ["html"],
                },
            },
            {
                "name": "html_select",
                "description": (
                    "Find all elements matching a tag name and optional attributes, "
                    "returning their text content. "
                    "E.g. html_select(html, 'code') returns all code blocks. "
                    "html_select(html, 'span', {'class': 'price'}) extracts prices."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "html": {"type": "string", "description": "Raw HTML string"},
                        "tag": {"type": "string", "description": "HTML tag name to select"},
                        "attrs": {
                            "type": "object",
                            "description": "Optional attribute key-value pairs to match",
                        },
                    },
                    "required": ["html", "tag"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "html_text":
            result = self.html_text(**arguments)
            return result
        if tool_name == "html_links":
            result = self.html_links(**arguments)
            return json.dumps(result)
        if tool_name == "html_headings":
            result = self.html_headings(**arguments)
            return json.dumps(result)
        if tool_name == "html_meta":
            result = self.html_meta(**arguments)
            return json.dumps(result)
        if tool_name == "html_tables":
            result = self.html_tables(**arguments)
            return json.dumps(result)
        if tool_name == "html_select":
            result = self.html_select(**arguments)
            return json.dumps(result)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
