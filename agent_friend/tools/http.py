"""http.py — Generic HTTP client tool for agent-friend (stdlib only)."""

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .base import BaseTool


_USER_AGENT = "agent-friend/0.13 (https://github.com/0-co/agent-friend)"

_METHODS_WITH_BODY = frozenset(["POST", "PUT", "PATCH"])


class HTTPTool(BaseTool):
    """Generic HTTP client — GET/POST/PUT/PATCH/DELETE with custom headers and JSON body.

    Unlike ``FetchTool`` (which is read-only and returns plain text),
    ``HTTPTool`` is designed for interacting with REST APIs:
    full method support, custom auth headers, JSON request/response bodies,
    and structured response metadata (status code, headers).

    All stdlib — no requests library required.

    Parameters
    ----------
    timeout:        Request timeout in seconds (default 10).
    max_body:       Maximum response body bytes to read (default 65536 / 64 KB).
    default_headers: Headers added to every request (e.g. {"Authorization": "Bearer sk-..."}).
    """

    def __init__(
        self,
        timeout: int = 10,
        max_body: int = 65_536,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.timeout = timeout
        self.max_body = max_body
        self.default_headers: Dict[str, str] = default_headers or {}

    @property
    def name(self) -> str:
        return "http"

    @property
    def description(self) -> str:
        return (
            "Generic HTTP client for REST APIs. Supports GET, POST, PUT, PATCH, DELETE "
            "with custom headers and JSON bodies. Returns status code, headers, and body."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "http_request",
                "description": (
                    "Make an HTTP request to any URL. "
                    "Use for REST APIs, webhooks, form submissions, or any HTTP endpoint. "
                    "Returns a JSON object with 'status', 'headers', and 'body'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "HTTP method: GET, POST, PUT, PATCH, or DELETE.",
                            "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                        },
                        "url": {
                            "type": "string",
                            "description": "Full URL including scheme (https://...).",
                        },
                        "headers": {
                            "type": "object",
                            "description": "HTTP headers as key-value pairs",
                        },
                        "body": {
                            "type": "object",
                            "description": "JSON request body (POST/PUT/PATCH)",
                        },
                        "body_text": {
                            "type": "string",
                            "description": (
                                "Optional raw string body (alternative to 'body'). "
                                "Use when you need to send a non-JSON body."
                            ),
                        },
                    },
                    "required": ["method", "url"],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name != "http_request":
            return f"Unknown tool: {tool_name}"

        method = str(arguments.get("method", "GET")).upper().strip()
        url = str(arguments.get("url", "")).strip()
        extra_headers: Dict[str, str] = arguments.get("headers") or {}
        body_obj: Optional[Dict[str, Any]] = arguments.get("body")
        body_text: Optional[str] = arguments.get("body_text")

        if not url:
            return "Error: url is required."
        if not url.startswith(("http://", "https://")):
            return f"Error: URL must start with http:// or https://. Got: {url!r}"
        if method not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
            return f"Error: unsupported method {method!r}. Use GET, POST, PUT, PATCH, or DELETE."

        # Build headers
        headers: Dict[str, str] = {
            "User-Agent": _USER_AGENT,
            **self.default_headers,
            **extra_headers,
        }

        # Build body bytes
        data: Optional[bytes] = None
        if method in _METHODS_WITH_BODY:
            if body_obj is not None:
                data = json.dumps(body_obj).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            elif body_text is not None:
                data = body_text.encode("utf-8")
                headers.setdefault("Content-Type", "text/plain; charset=utf-8")

        try:
            req = urllib.request.Request(url, data=data, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status = resp.status
                resp_headers = dict(resp.headers)
                body_bytes = resp.read(self.max_body)
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers) if e.headers else {}
            body_bytes = e.read(self.max_body) if e.fp else b""
        except urllib.error.URLError as e:
            return json.dumps({"error": f"Could not reach {url}: {e.reason}"})
        except TimeoutError:
            return json.dumps({"error": f"Timeout after {self.timeout}s: {url}"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"error": f"Request failed: {e}"})

        # Decode body
        content_type = resp_headers.get("Content-Type", "")
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()

        body_str = body_bytes.decode(charset, errors="replace")

        # Try to return JSON body as parsed object for readability
        response: Dict[str, Any] = {
            "status": status,
            "headers": {k: v for k, v in resp_headers.items() if k.lower() not in (
                "server", "date", "x-request-id", "cf-ray", "cf-cache-status",
            )},
            "body": body_str,
        }

        # If response body is JSON, include parsed version
        if "json" in content_type.lower() and body_str:
            try:
                response["json"] = json.loads(body_str)
            except json.JSONDecodeError:
                pass

        return json.dumps(response, ensure_ascii=False)
