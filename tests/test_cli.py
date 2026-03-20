"""Tests for CLI utility functions — URL resolution in _resolve_file_or_example."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(file="-", example=None):
    """Create a minimal args namespace."""
    args = MagicMock()
    args.file = file
    args.example = example
    return args


SAMPLE_TOOLS = [
    {
        "name": "ping",
        "description": "Ping a host.",
        "inputSchema": {
            "type": "object",
            "properties": {"host": {"type": "string", "description": "Hostname"}},
            "required": ["host"],
        },
    }
]


# ---------------------------------------------------------------------------
# TestResolveFileOrExample — URL handling
# ---------------------------------------------------------------------------

class TestResolveFileOrExampleURL:
    """Tests for URL fetching in _resolve_file_or_example."""

    def _call(self, args):
        from agent_friend.cli import _resolve_file_or_example
        return _resolve_file_or_example(args)

    def test_plain_file_passthrough(self):
        """Non-URL file path is returned as-is."""
        args = _make_args(file="/tmp/some_file.json")
        result = self._call(args)
        assert result == "/tmp/some_file.json"

    def test_stdin_passthrough(self):
        """'-' (stdin) is returned as-is."""
        args = _make_args(file="-")
        result = self._call(args)
        assert result == "-"

    def test_http_url_fetches_to_temp_file(self):
        """http:// URL is fetched and written to a temp file."""
        raw = json.dumps(SAMPLE_TOOLS).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            args = _make_args(file="http://example.com/tools.json")
            result = self._call(args)

        assert os.path.exists(result)
        with open(result) as f:
            data = json.load(f)
        assert data == SAMPLE_TOOLS
        os.unlink(result)

    def test_https_url_fetches_to_temp_file(self):
        """https:// URL is fetched and written to a temp file."""
        raw = json.dumps(SAMPLE_TOOLS).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            args = _make_args(file="https://example.com/schema.json")
            result = self._call(args)

        assert os.path.exists(result)
        content = open(result, "rb").read()
        assert content == raw
        os.unlink(result)

    def test_http_error_exits(self):
        """HTTP 404 response causes sys.exit(1)."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            url="http://x.com", code=404, msg="Not Found", hdrs=None, fp=None
        )):
            args = _make_args(file="https://example.com/missing.json")
            with pytest.raises(SystemExit) as exc_info:
                self._call(args)
        assert exc_info.value.code == 1

    def test_url_error_exits(self):
        """Network error causes sys.exit(1)."""
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            args = _make_args(file="https://unreachable.example.com/tools.json")
            with pytest.raises(SystemExit) as exc_info:
                self._call(args)
        assert exc_info.value.code == 1

    def test_url_temp_file_prefix(self):
        """Temp file from URL fetch has expected prefix."""
        raw = b"[]"

        mock_resp = MagicMock()
        mock_resp.read.return_value = raw
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            args = _make_args(file="https://example.com/schema.json")
            result = self._call(args)

        assert os.path.basename(result).startswith("agent-friend-url-")
        os.unlink(result)

    def test_example_takes_precedence_over_url(self):
        """--example flag takes precedence over a URL file arg."""
        # If example is set, we should get an example path, not URL fetch
        args = _make_args(file="https://example.com/schema.json", example="notion")

        from agent_friend.examples import get_example
        try:
            get_example("notion")
        except ValueError:
            pytest.skip("notion example not available")

        result = self._call(args)
        # Should be a temp file from example, not a URL fetch
        assert os.path.exists(result)
        assert "example" in os.path.basename(result)
        os.unlink(result)
