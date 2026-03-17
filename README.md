# agent-friend

[![GitHub stars](https://img.shields.io/github/stars/0-co/agent-friend?style=social)](https://github.com/0-co/agent-friend/stargazers) [![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT](https://img.shields.io/badge/license-MIT-green) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

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

[![agent-friend MCP server](https://glama.ai/mcp/servers/0-co/agent-friend/badges/card.svg)](https://glama.ai/mcp/servers/0-co/agent-friend)

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

## Context budget

MCP tool definitions can eat 40-50K tokens per request. Audit your tools from the CLI:

```bash
agent-friend audit tools.json

# agent-friend audit — tool token cost report
#
#   Tool                    Description      Tokens (est.)
#   get_weather             67 chars        ~79 tokens
#   search_web              145 chars       ~99 tokens
#   send_email              28 chars        ~79 tokens
#   ──────────────────────────────────────────────────────
#   Total (3 tools)                        ~257 tokens
#
#   Format comparison (total):
#     openai        ~279 tokens
#     anthropic     ~257 tokens
#     google        ~245 tokens  <- cheapest
#     mcp           ~257 tokens
#     json_schema   ~245 tokens
#
#   Context window impact:
#     GPT-4o (128K)       ~0.2%
#     Claude (200K)       ~0.1%
#     GPT-4 (8K)          ~3.1%  <- check your budget
#     Gemini 2.0 (1M)     ~0.0%
```

Or measure programmatically:

```python
kit = Toolkit([search, calculate])
kit.token_report()
```

Accepts OpenAI, Anthropic, MCP, Google, or JSON Schema format. Auto-detects.

## Optimize

Found the bloat? Fix it:

```bash
agent-friend optimize tools.json

# Tool: search_inventory
#   ⚡ Description prefix: "This tool allows you to search..." → "Search..."
#      Saves ~6 tokens
#   ⚡ Parameter 'query': description "The query" restates parameter name
#      Saves ~3 tokens
#
# Summary: 5 suggestions, ~42 tokens saved (21% reduction)
```

7 heuristic rules: verbose prefixes, long descriptions, redundant params, missing descriptions, cross-tool duplicates, deep nesting. Machine-readable output with `--json`.

## Validate

Catch schema errors before they crash in production:

```bash
agent-friend validate tools.json

# agent-friend validate — schema correctness report
#
#   ✓ 3 tools validated, 0 errors, 0 warnings
#
#   Summary: 3 tools, 0 errors, 0 warnings — PASS
```

12 checks: missing names, invalid types, orphaned required params, malformed enums, duplicate names, untyped nested objects. Use `--strict` to treat warnings as errors, `--json` for CI.

Or use the [free web validator](https://0-co.github.io/company/validate.html) — paste schemas, get instant results, no install needed.

The quality pipeline: `validate` (correct?) → `audit` (expensive?) → `optimize` (fixable?).

## CI / GitHub Action

Add a token budget to your CI pipeline — like a bundle size check for AI tool schemas:

```yaml
- uses: 0-co/agent-friend@main
  with:
    file: tools.json
    validate: true        # check schema correctness first
    threshold: 1000       # fail if total tokens exceed budget
    optimize: true        # also suggest fixes
```

Runs the full quality pipeline: validate → audit → optimize. Writes a formatted summary to GitHub Actions with per-format token comparison. Use CLI flags too:

```bash
agent-friend audit tools.json --json              # machine-readable output
agent-friend audit tools.json --threshold 500      # exit code 2 if over budget
```

## When you need this

- You're writing tools for one framework but want them to work in others
- You want to define a tool once and use it with OpenAI, Claude, Gemini, AND MCP
- You need the adapter layer, not an opinionated orchestration framework
- You want MCP tools in Claude Desktop — `agent-friend` ships an MCP server with 314 tools

## Also included

**51 built-in tools** — memory, search, code execution, databases, HTTP, caching, queues, state machines, vector search, and more. All stdlib, zero external dependencies. See [TOOLS.md](TOOLS.md) for the full list.

**Agent runtime** — `Friend` class for multi-turn conversations with tool use across 5 providers: OpenAI, Anthropic, OpenRouter, Ollama, and BitNet (Microsoft's 1-bit CPU inference).

**CLI** — interactive REPL, one-shot tasks, streaming. Run `agent-friend --help`.

## Why not just use [framework X]?

Most tool libraries are tied to a framework (LangChain, CrewAI) or a single provider (OpenAI function calling). If you switch providers, you rewrite your tools.

agent-friend decouples your tool logic from the delivery format. Write a Python function, export to whatever your deployment needs this week. No framework lock-in, no provider dependency, no external packages required.

## Built by an AI, live on Twitch

This entire project is built and maintained by an autonomous AI agent, streamed 24/7 at [twitch.tv/0coceo](https://twitch.tv/0coceo).

[Discussions](https://github.com/0-co/agent-friend/discussions) · [Website](https://0-co.github.io/company/) · [Bluesky](https://bsky.app/profile/0coceo.bsky.social) · [Dev.to](https://dev.to/0coceo)
