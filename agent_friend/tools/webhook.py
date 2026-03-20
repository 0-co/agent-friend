"""webhook.py — WebhookTool for agent-friend.

Starts an HTTP server and waits for incoming POST requests.
Use this when your agent needs to react to external events:
payment callbacks, GitHub webhooks, Slack events, etc.

Usage::

    tool = WebhookTool()
    result = tool.listen(path="/webhook", timeout=30.0)
    print(result["body"])
    print(result["json"])   # dict if body was valid JSON, else None
"""

import http.server
import json
import queue
import socketserver
import threading
import time
from typing import Any, Dict, List, Optional

from .base import BaseTool


class WebhookTool(BaseTool):
    """HTTP webhook receiver for agents.

    Starts an HTTP server and waits for incoming POST requests.
    Use this when your agent needs to react to external events:
    payment callbacks, GitHub webhooks, Slack events, etc.

    The server runs in a background thread and is cleaned up automatically.
    """

    def __init__(self, port: int = 0, host: str = "0.0.0.0") -> None:
        self._port = port
        self._host = host
        self._actual_port: Optional[int] = None

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "webhook"

    @property
    def description(self) -> str:
        return (
            "Start an HTTP server and wait for an incoming POST webhook. "
            "Returns the request body and headers when received. "
            "Use for payment callbacks, GitHub webhooks, form submissions, etc."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        """Return LLM tool definitions for the webhook receiver."""
        return [
            {
                "name": "wait_for_webhook",
                "description": (
                    "Start an HTTP server and wait for an incoming POST webhook. "
                    "Returns the request body and headers when received. "
                    "Use for payment callbacks, GitHub webhooks, form submissions, etc."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": (
                                "URL path to listen on (e.g., '/webhook', '/payment'). "
                                "Default: '/webhook'"
                            ),
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Seconds to wait before giving up. Default: 30",
                        },
                        "port": {
                            "type": "integer",
                            "description": "Port to listen on. Default: random available port",
                        },
                    },
                    "required": [],
                },
            }
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call from the LLM to the appropriate method."""
        if tool_name == "wait_for_webhook":
            path = arguments.get("path", "/webhook")
            timeout = arguments.get("timeout", 30.0)
            port = arguments.get("port", self._port)
            # Override port if provided in arguments
            if port:
                self._port = port
            try:
                result = self.listen(path=path, timeout=float(timeout))
                return json.dumps(result)
            except TimeoutError as exc:
                return json.dumps({"error": str(exc)})
        return f"Unknown webhook tool: {tool_name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def listen(self, path: str = "/webhook", timeout: float = 30.0) -> dict:
        """Start HTTP server and wait for one POST request at `path`.

        Parameters
        ----------
        path:
            URL path to listen on (e.g. '/webhook').
        timeout:
            Seconds to wait before raising TimeoutError.

        Returns
        -------
        dict with keys:
            path (str), method (str), headers (dict), body (str),
            json (dict or None), received_at (float)

        Raises
        ------
        TimeoutError
            If no request arrives within `timeout` seconds.
        """
        result_queue: queue.Queue = queue.Queue()
        expected_path = path

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self_handler):
                if self_handler.path != expected_path:
                    self_handler.send_response(404)
                    self_handler.end_headers()
                    self_handler.wfile.write(b'{"status": "not found"}')
                    return

                content_length = int(self_handler.headers.get("Content-Length", 0))
                raw_body = self_handler.rfile.read(content_length) if content_length > 0 else b""
                body_str = raw_body.decode("utf-8", errors="replace")

                parsed_json: Optional[dict] = None
                try:
                    parsed_json = json.loads(body_str)
                except (json.JSONDecodeError, ValueError):
                    parsed_json = None

                request_data = {
                    "path": self_handler.path,
                    "method": "POST",
                    "headers": dict(self_handler.headers),
                    "body": body_str,
                    "json": parsed_json,
                    "received_at": time.time(),
                }

                self_handler.send_response(200)
                self_handler.send_header("Content-Type", "application/json")
                self_handler.end_headers()
                self_handler.wfile.write(b'{"status": "received"}')

                result_queue.put(request_data)

            def log_message(self, format, *args):  # noqa: A002
                # Suppress default request logging
                pass

        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer((self._host, self._port), _Handler)
        self._actual_port = server.server_address[1]

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            result = result_queue.get(timeout=timeout)
        except queue.Empty:
            # Shut down server before raising
            shutdown_thread = threading.Thread(target=server.shutdown, daemon=True)
            shutdown_thread.start()
            self._actual_port = None
            raise TimeoutError(
                f"No webhook received within {timeout} seconds on {self._host}:{self._actual_port or self._port}{path}"
            )

        # Shut down in a daemon thread to avoid deadlock
        shutdown_thread = threading.Thread(target=server.shutdown, daemon=True)
        shutdown_thread.start()
        self._actual_port = None

        return result

    def get_port(self) -> Optional[int]:
        """Return the port the server is listening on, or None if not started."""
        return self._actual_port
