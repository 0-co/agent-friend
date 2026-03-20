"""search.py — DuckDuckGo web search tool for agent-friend (no API key required)."""

import urllib.parse
import urllib.request
import urllib.error
import html
import re
from typing import Any, Dict, List

from .base import BaseTool


_DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
)


class SearchTool(BaseTool):
    """Web search using DuckDuckGo's public HTML interface (no API key needed).

    Parameters
    ----------
    max_results:  Maximum number of results to return (default 5).
    timeout:      HTTP request timeout in seconds (default 10).
    """

    def __init__(self, max_results: int = 5, timeout: int = 10) -> None:
        self.max_results = max_results
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Search the web using DuckDuckGo (no API key required)."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "search",
                "description": "Search the web and return titles, URLs, and snippets.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query.",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results (default 5).",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "search":
            return f"Unknown search tool: {tool_name}"
        query = arguments["query"]
        max_results = int(arguments.get("max_results", self.max_results))
        return self._search(query, max_results)

    def _search(self, query: str, max_results: int) -> str:
        """Fetch and parse DuckDuckGo HTML results."""
        try:
            html_content = self._fetch(query)
        except urllib.error.URLError as error:
            return f"Search failed (network error): {error}"
        except Exception as error:
            return f"Search failed: {error}"

        results = self._parse(html_content, max_results)
        if not results:
            return "No results found."

        lines = []
        for index, result in enumerate(results, 1):
            lines.append(f"{index}. {result['title']}")
            lines.append(f"   URL: {result['url']}")
            if result["snippet"]:
                lines.append(f"   {result['snippet']}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def _fetch(self, query: str) -> str:
        """Fetch the DuckDuckGo HTML search page."""
        params = urllib.parse.urlencode({"q": query, "kl": "us-en"})
        url = f"{_DUCKDUCKGO_URL}?{params}"
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def _parse(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Extract search results from DuckDuckGo HTML response."""
        results = []

        # Each result block is wrapped in a div with class "result"
        # We use regex to extract title, URL, and snippet without external deps.
        result_blocks = re.findall(
            r'<div[^>]*class="[^"]*result[^"]*"[^>]*>(.*?)</div>\s*</div>',
            html_content,
            re.DOTALL,
        )

        for block in result_blocks:
            if len(results) >= max_results:
                break

            title = self._extract_title(block)
            url = self._extract_url(block)
            snippet = self._extract_snippet(block)

            if title and url:
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet,
                    }
                )

        # Fallback: try link-based extraction if no results found
        if not results:
            results = self._parse_fallback(html_content, max_results)

        return results

    def _extract_title(self, block: str) -> str:
        """Extract title text from a result block."""
        match = re.search(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*>(.*?)</a>', block, re.DOTALL)
        if match:
            return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
        return ""

    def _extract_url(self, block: str) -> str:
        """Extract URL from a result block."""
        # DuckDuckGo wraps URLs in a redirect; look for uddg param or direct href
        match = re.search(r'uddg=([^&"]+)', block)
        if match:
            return urllib.parse.unquote(match.group(1))

        # Try href directly on result link
        match = re.search(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"', block)
        if match:
            url = html.unescape(match.group(1))
            if url.startswith("http"):
                return url
        return ""

    def _extract_snippet(self, block: str) -> str:
        """Extract snippet text from a result block."""
        match = re.search(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )
        if match:
            return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()

        # Try span-based snippet
        match = re.search(
            r'<span[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</span>',
            block,
            re.DOTALL,
        )
        if match:
            return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
        return ""

    def _parse_fallback(self, html_content: str, max_results: int) -> List[Dict[str, str]]:
        """Simpler fallback parser that finds any result__a links."""
        results = []
        links = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            html_content,
            re.DOTALL,
        )
        for href, title_html in links[:max_results]:
            url = html.unescape(href)
            # Handle DuckDuckGo redirect URLs
            uddg_match = re.search(r"uddg=([^&]+)", url)
            if uddg_match:
                url = urllib.parse.unquote(uddg_match.group(1))
            title = html.unescape(re.sub(r"<[^>]+>", "", title_html)).strip()
            if title and url.startswith("http"):
                results.append({"title": title, "url": url, "snippet": ""})
        return results
