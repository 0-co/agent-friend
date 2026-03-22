#!/usr/bin/env python3
"""
agent-friend Quality API Server
REST API for programmatic MCP schema grading.

Endpoints:
  GET  /health         → {"status": "ok", "version": "..."}
  POST /v1/grade       → grade tools from request body {"tools": [...]}
  GET  /v1/grade?url=  → fetch remote schema and grade it
  GET  /v1/servers     → list top graded servers from leaderboard

Usage:
  python3 api_server.py [--port 8082] [--host 0.0.0.0]

Runs on port 8082 (TTS server is on 8081).
"""

import http.server
import json
import sys
import time
import urllib.request
import urllib.parse
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_friend.grade import grade_tools
from agent_friend.audit import detect_format

DEFAULT_PORT = 8082
RATE_LIMIT_WINDOW = 60   # seconds
RATE_LIMIT_MAX = 20      # requests per window per IP
_rate_data = defaultdict(list)

VERSION = "1.0.0"


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    window = _rate_data[ip]
    window[:] = [t for t in window if now - t < RATE_LIMIT_WINDOW]
    if len(window) >= RATE_LIMIT_MAX:
        return False
    window.append(now)
    return True


def fetch_remote_schema(url: str) -> list:
    """Fetch a JSON schema from a URL and extract tools list."""
    req = urllib.request.Request(url, headers={"User-Agent": "agent-friend-api/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    # Handle various schema formats
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # OpenAI: {"tools": [...]}
        if "tools" in data:
            return data["tools"]
        # Anthropic: {"tools": [...]}
        # MCP result: {"tools": [...]}
        for key in ["tools", "functions", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]
    raise ValueError(f"Cannot extract tools list from schema at {url}")


def handle_grade(tools_data: list) -> dict:
    """Run grading and return API response."""
    result = grade_tools(tools_data)
    return {
        "score": round(result["overall_score"], 1),
        "grade": result["overall_grade"],
        "tool_count": result["tool_count"],
        "total_tokens": result["total_tokens"],
        "scores": {
            "correctness": result.get("correctness", {}).get("score", 0) if isinstance(result.get("correctness"), dict) else result.get("correctness", 0),
            "efficiency": result.get("efficiency", {}).get("score", 0) if isinstance(result.get("efficiency"), dict) else result.get("efficiency", 0),
            "quality": result.get("quality", {}).get("score", 0) if isinstance(result.get("quality"), dict) else result.get("quality", 0),
        },
        "issue_count": (
            (result.get("correctness", {}).get("errors", 0) + result.get("correctness", {}).get("warnings", 0))
            if isinstance(result.get("correctness"), dict) else 0
        ),
        "issues": [],  # issues detail requires separate run
        "detected_format": result.get("detected_format", "unknown"),
    }


class APIHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # Minimal logging
        ts = time.strftime("%H:%M:%SZ", time.gmtime())
        print(f"[{ts}] {self.address_string()} {fmt % args}", flush=True)

    def send_json(self, code: int, data: dict):
        body = json.dumps(data, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = dict(urllib.parse.parse_qsl(parsed.query))
        ip = self.client_address[0]

        if not check_rate_limit(ip):
            self.send_json(429, {"error": "rate limit exceeded"})
            return

        if path == "/health":
            self.send_json(200, {
                "status": "ok",
                "version": VERSION,
                "service": "agent-friend-api"
            })

        elif path == "/v1/grade":
            url = params.get("url")
            if not url:
                self.send_json(400, {"error": "url parameter required"})
                return
            try:
                tools = fetch_remote_schema(url)
                result = handle_grade(tools)
                result["source_url"] = url
                self.send_json(200, result)
            except ValueError as e:
                self.send_json(422, {"error": str(e)})
            except Exception as e:
                self.send_json(500, {"error": f"failed to fetch schema: {e}"})

        elif path == "/v1/servers":
            # Return top servers from leaderboard
            try:
                leaderboard = load_leaderboard()
                limit = min(int(params.get("limit", 20)), 201)
                offset = int(params.get("offset", 0))
                grade_filter = params.get("grade")
                servers = leaderboard
                if grade_filter:
                    servers = [s for s in servers if s.get("grade") == grade_filter.upper()]
                self.send_json(200, {
                    "total": len(servers),
                    "servers": servers[offset:offset + limit],
                    "offset": offset,
                    "limit": limit,
                })
            except Exception as e:
                self.send_json(500, {"error": str(e)})

        else:
            self.send_json(404, {"error": f"unknown endpoint: {path}", "endpoints": [
                "GET /health",
                "GET /v1/grade?url=...",
                "POST /v1/grade",
                "GET /v1/servers",
            ]})

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        ip = self.client_address[0]

        if not check_rate_limit(ip):
            self.send_json(429, {"error": "rate limit exceeded"})
            return

        if path == "/v1/grade":
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 1_000_000:  # 1MB limit
                    self.send_json(413, {"error": "request too large"})
                    return
                body = self.rfile.read(length)
                data = json.loads(body)
                # Accept {"tools": [...]} or bare list
                if isinstance(data, dict):
                    tools = data.get("tools", data.get("functions", []))
                elif isinstance(data, list):
                    tools = data
                else:
                    self.send_json(400, {"error": "body must be JSON array or {tools: [...]}"})
                    return
                result = handle_grade(tools)
                self.send_json(200, result)
            except json.JSONDecodeError:
                self.send_json(400, {"error": "invalid JSON"})
            except Exception as e:
                self.send_json(500, {"error": str(e)})
        else:
            self.send_json(404, {"error": f"unknown endpoint: {path}"})


def load_leaderboard() -> list:
    """Load leaderboard data from leaderboard_data.py."""
    try:
        from agent_friend.leaderboard_data import LEADERBOARD_DATA
        return LEADERBOARD_DATA
    except ImportError:
        return []


def main():
    import argparse
    parser = argparse.ArgumentParser(description="agent-friend Quality API Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = http.server.HTTPServer((args.host, args.port), APIHandler)
    print(f"agent-friend API server starting on {args.host}:{args.port}", flush=True)
    print(f"Endpoints: GET /health, GET /v1/grade?url=..., POST /v1/grade, GET /v1/servers", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down...", flush=True)


if __name__ == "__main__":
    main()
