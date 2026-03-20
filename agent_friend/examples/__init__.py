"""examples — Bundled MCP server tool schemas for quick demos.

Usage:
    from agent_friend.examples import get_example, list_examples

    tools = get_example("notion")      # returns parsed JSON (list of tool dicts)
    names = list_examples()            # returns ["filesystem", "github", "notion"]
"""

import json
import os
from typing import Any, Dict, List


_EXAMPLES_DIR = os.path.dirname(__file__)

# Map of example name -> JSON filename (without extension)
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "notion": {
        "file": "notion.json",
        "description": "Notion MCP server (22 tools) — official makenotion/notion-mcp-server",
    },
    "github": {
        "file": "github.json",
        "description": "GitHub MCP server (12 tools) — modelcontextprotocol/servers",
    },
    "filesystem": {
        "file": "filesystem.json",
        "description": "Filesystem MCP server (11 tools) — modelcontextprotocol/servers",
    },
    "slack": {
        "file": "slack.json",
        "description": "Slack MCP server (8 tools) — modelcontextprotocol/servers",
    },
    "puppeteer": {
        "file": "puppeteer.json",
        "description": "Puppeteer MCP server (7 tools) — modelcontextprotocol/servers",
    },
}


def list_examples() -> List[str]:
    """Return sorted list of available example names."""
    return sorted(_REGISTRY.keys())


def get_example_info() -> Dict[str, str]:
    """Return dict of {name: description} for all examples."""
    return {name: info["description"] for name, info in sorted(_REGISTRY.items())}


def get_example(name: str) -> List[Dict[str, Any]]:
    """Load and return the tool schema list for a named example.

    Parameters
    ----------
    name:
        Example name (e.g. "notion", "github", "filesystem").

    Returns
    -------
    List of tool definition dicts (MCP format).

    Raises
    ------
    ValueError:
        If the example name is not found.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            "Unknown example: '{name}'. Available: {available}".format(
                name=name, available=available,
            )
        )

    file_path = os.path.join(_EXAMPLES_DIR, _REGISTRY[name]["file"])
    with open(file_path, "r") as f:
        data = json.load(f)

    return data
