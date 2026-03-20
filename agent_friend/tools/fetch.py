"""fetch.py — URL fetching tool for agent-friend (no API key required)."""

import re
import urllib.request
import urllib.error
from html.parser import HTMLParser
from typing import Any, Dict, List

from .base import BaseTool


_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
)

# Tags whose content we skip (scripts, styles, navigation, etc.)
_SKIP_TAGS = frozenset([
    "script", "style", "nav", "header", "footer",
    "noscript", "iframe", "svg", "meta", "link",
])


class _TextExtractor(HTMLParser):
    """Minimal HTML-to-text extractor using stdlib HTMLParser."""

    def __init__(self) -> None:
        super().__init__()
        self.texts: List[str] = []
        self._skip_depth: int = 0

    def handle_starttag(self, tag: str, attrs: List) -> None:
        if tag.lower() in _SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self.texts.append(text)

    def get_text(self) -> str:
        return " ".join(self.texts)


class FetchTool(BaseTool):
    """Fetch the text content of any URL (no API key required).

    Reads web pages, documentation, articles, and plain text files.
    HTML is automatically converted to plain text. Use ``search`` to
    find URLs; use ``fetch`` to read a specific URL you already have.

    Parameters
    ----------
    timeout:    HTTP request timeout in seconds (default 10).
    max_chars:  Maximum characters to return (default 8000).
    """

    def __init__(self, timeout: int = 10, max_chars: int = 8000) -> None:
        self.timeout = timeout
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "fetch"

    @property
    def description(self) -> str:
        return "Fetch the text content of a URL (web pages, docs, articles). No API key needed."

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "fetch",
                "description": (
                    "Fetch the text content of a URL. "
                    "HTML pages are automatically converted to readable plain text. "
                    "Use 'search' to find URLs; use 'fetch' to read a specific URL."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL to fetch (must start with http:// or https://).",
                        },
                        "max_chars": {
                            "type": "integer",
                            "description": "Maximum characters to return (default 8000).",
                        },
                    },
                    "required": ["url"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "fetch":
            return f"Unknown tool: {tool_name}"

        url = arguments.get("url", "").strip()
        max_chars = int(arguments.get("max_chars", self.max_chars))

        if not url:
            return "Error: url is required."
        if not url.startswith(("http://", "https://")):
            return f"Error: URL must start with http:// or https://. Got: {url!r}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                content_type = resp.headers.get("Content-Type", "text/plain")
                # Read up to 10× max_chars bytes, then trim after decoding
                raw = resp.read(max_chars * 10)

            # Detect charset from Content-Type header
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            text = raw.decode(charset, errors="replace")

            # Extract plain text from HTML
            if self._looks_like_html(content_type, text):
                text = self._html_to_text(text)

            # Normalise whitespace
            text = re.sub(r"[ \t]{2,}", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()

            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[truncated at {max_chars} characters]"

            return text or "(no text content found)"

        except urllib.error.HTTPError as e:
            return f"HTTP {e.code} {e.reason}: {url}"
        except urllib.error.URLError as e:
            return f"Could not fetch {url}: {e.reason}"
        except TimeoutError:
            return f"Timeout after {self.timeout}s: {url}"
        except Exception as e:  # noqa: BLE001
            return f"Error fetching {url}: {e}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _looks_like_html(self, content_type: str, text: str) -> bool:
        if "html" in content_type.lower():
            return True
        preview = text[:500].lower()
        return "<!doctype" in preview or "<html" in preview

    def _html_to_text(self, html: str) -> str:
        extractor = _TextExtractor()
        try:
            extractor.feed(html)
        except Exception:  # noqa: BLE001
            pass
        return extractor.get_text()
