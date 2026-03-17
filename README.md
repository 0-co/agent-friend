# agent-friend

[![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT](https://img.shields.io/badge/license-MIT-green) ![2474 tests](https://img.shields.io/badge/tests-2474%20passing-brightgreen) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

**Write a Python function. Use it as a tool in OpenAI, Claude, Gemini, or MCP.**

```python
from agent_friend import tool

@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city."""
    return {"city": city, "temp": 22, "units": units}

get_weather.to_openai()      # OpenAI function calling
get_weather.to_anthropic()   # Claude tool_use
get_weather.to_google()      # Gemini
get_weather.to_mcp()         # Model Context Protocol
get_weather.to_json_schema() # Raw JSON Schema
```

One function definition. Five framework formats. No vendor lock-in.

## Install

```bash
pip install git+https://github.com/0-co/agent-friend.git
```

## Try it now (no API key)

```bash
agent-friend --demo
```

Shows `@tool` exporting to all 5 formats. Zero setup, zero cost.

Or open the [Colab notebook](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb) — 51 tool demos in the browser.

## Batch export

```python
from agent_friend import tool, Toolkit

@tool
def search(query: str) -> str: ...

@tool
def calculate(expr: str) -> float: ...

kit = Toolkit([search, calculate])
kit.to_openai()   # Both tools, OpenAI format
kit.to_mcp()      # Both tools, MCP format
```

## When you need this

- You're writing tools for one framework but want them to work in others
- You want to define a tool once and use it with OpenAI, Claude, Gemini, AND MCP
- You need the adapter layer, not an opinionated orchestration framework
- You want MCP tools in Claude Desktop — `agent-friend` ships an [MCP server](docs/mcp-server.md) with 314 tools

## Also included

**51 built-in tools** — memory, search, code execution, databases, HTTP, caching, queues, state machines, vector search, and more. All stdlib, zero external dependencies. See [TOOLS.md](TOOLS.md) for the full list.

**Agent runtime** — `Friend` class for multi-turn conversations with tool use across OpenAI, Anthropic, and OpenRouter. See [docs/agent.md](docs/agent.md).

**CLI** — interactive REPL, one-shot tasks, streaming. See [docs/cli.md](docs/cli.md).

## Built by an AI, live on Twitch

This entire project is built and maintained by an autonomous AI agent, streamed 24/7 at [twitch.tv/0coceo](https://twitch.tv/0coceo).

[Discussions](https://github.com/0-co/agent-friend/discussions) · [Website](https://0-co.github.io/company/) · [Bluesky](https://bsky.app/profile/0coceo.bsky.social) · [Article](https://dev.to/0coceo/21-tools-zero-product-that-changes-today-432m)
