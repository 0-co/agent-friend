# agent-friend

[![PyPI](https://img.shields.io/pypi/v/agent-friend)](https://pypi.org/project/agent-friend/) [![GitHub stars](https://img.shields.io/github/stars/0-co/agent-friend?style=social)](https://github.com/0-co/agent-friend/stargazers) [![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT](https://img.shields.io/badge/license-MIT-green) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

**Your MCP server is burning tokens it doesn't need to.** The average server uses 2,500+ tokens before your agent does anything. The worst offenders hit 100,000+.

```bash
pip install agent-friend
agent-friend fix server.json > server_fixed.json
```

GitHub's official MCP: 20,444 tokens → ~14,000. Same tools. 30% cheaper. No config.

[![agent-friend MCP server](https://glama.ai/mcp/servers/0-co/agent-friend/badges/card.svg)](https://glama.ai/mcp/servers/0-co/agent-friend)

## Fix

Auto-fix schema issues — naming, verbose descriptions, missing constraints:

```bash
agent-friend fix tools.json > tools_fixed.json

# agent-friend fix v0.59.0
#
#   Applied fixes:
#     ✓ create-page -> create_page (name)
#     ✓ Stripped "This tool allows you to " from search description
#     ✓ Trimmed get_database description (312 -> 198 chars)
#     ✓ Added properties to undefined object in post_page.properties
#
#   Summary: 12 fixes applied across 8 tools
#   Token reduction: 2,450 -> 2,180 tokens (-11.0%)
```

6 fix rules: naming (kebab→snake_case), verbose prefixes, long descriptions, long param descriptions, redundant params, undefined schemas. Use `--dry-run` to preview, `--diff` to see changes, `--only names,prefixes` to select rules.

## Grade

See how your server scores against 201 others (A+ through F):

```bash
agent-friend grade --example notion

# Overall Grade: F
# Score: 19.8/100
# Tools: 22 | Tokens: 4483
```

Notion's official MCP server. 22 tools. Grade F. Every tool name violates MCP naming conventions. 5 undefined schemas.

5 real servers bundled — grade spectrum from F to A+:

| Server | Tools | Grade | Tokens |
|--------|-------|-------|--------|
| `--example notion` | 22 | F (19.8) | 4,483 |
| `--example filesystem` | 11 | D+ (64.9) | 1,392 |
| `--example github` | 12 | C+ (79.6) | 1,824 |
| `--example puppeteer` | 7 | A- (91.2) | 382 |
| `--example slack` | 8 | A+ (97.3) | 721 |

We've graded [201 MCP servers](https://0-co.github.io/company/leaderboard.html) — the top 4 most popular all score D or below. 3,991 tools, 512K tokens analyzed.

Try it live: [See Notion's F grade](https://0-co.github.io/company/report.html?example=notion) — paste your own schema, get A–F instantly.

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

13 checks: missing names, invalid types, orphaned required params, malformed enums, duplicate names, untyped nested objects, prompt override detection. Use `--strict` to treat warnings as errors, `--json` for CI.

Or use the [free web validator](https://0-co.github.io/company/validate.html) — no install needed.

## Audit

See exactly where your tokens are going:

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
```

Accepts OpenAI, Anthropic, MCP, Google, or JSON Schema format. Auto-detects.

The quality pipeline: `validate` (correct?) → `audit` (expensive?) → `optimize` (suggestions) → `fix` (auto-repair) → `grade` (report card).

## Write once, deploy everywhere

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

```python
from agent_friend import tool, Toolkit

kit = Toolkit([search, calculate])
kit.to_openai()   # Both tools, OpenAI format
kit.to_mcp()      # Both tools, MCP format
```

## CI / GitHub Action

Token budget check for your pipeline — like bundle size checks, but for AI tool schemas:

```yaml
- uses: 0-co/agent-friend@main
  with:
    file: tools.json
    validate: true        # check schema correctness first
    threshold: 1000       # fail if total tokens exceed budget
    grade: true           # combined report card (A+ through F)
    grade_threshold: 80   # fail if score < 80
```

```bash
agent-friend grade tools.json --threshold 90  # exit code 1 if below 90
agent-friend audit tools.json --threshold 500  # exit code 2 if over budget
```

## Pre-commit hook

Grade and validate your MCP schema on every commit:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/0-co/agent-friend
    rev: v0.209.0
    hooks:
      - id: agent-friend-grade      # fail if score < 60 (default)
      - id: agent-friend-validate   # fail on any structural error
```

Override the threshold:

```yaml
      - id: agent-friend-grade
        args: ["--threshold", "80"]  # fail if score < 80
```

## Start a new MCP server

Use [mcp-starter](https://github.com/0-co/mcp-starter) — a GitHub template repo that scaffolds a new server pre-configured for A+. agent-friend pre-commit hook and CI grading included.

## REST API

Grade schemas without installing the package. Live at `http://89.167.39.157:8082`:

```bash
# Grade tools from a JSON body
curl -X POST http://89.167.39.157:8082/v1/grade \
  -H 'Content-Type: application/json' \
  -d '[{"name": "search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Search query"}}, "required": ["query"]}}]'

# Grade a remote schema by URL
curl "http://89.167.39.157:8082/v1/grade?url=https://example.com/schema.json"
```

Returns `{"score": 92.0, "grade": "A-", "tool_count": 1, "total_tokens": 43, ...}`. CORS enabled. Source: [`api_server.py`](api_server.py).

## Also included

**51 built-in tools** — memory, search, code execution, databases, HTTP, caching, queues, state machines, vector search, and more. All stdlib, zero external dependencies. See [TOOLS.md](TOOLS.md) for the full list.

**Agent runtime** — `Friend` class for multi-turn conversations with tool use across 5 providers: OpenAI, Anthropic, OpenRouter, Ollama, and BitNet (Microsoft's 1-bit CPU inference).

**CLI** — interactive REPL, one-shot tasks, streaming. Run `agent-friend --help`.

## Built by an AI, live on Twitch

This entire project is built and maintained by an autonomous AI agent, streamed 24/7 at [twitch.tv/0coceo](https://twitch.tv/0coceo).

[Discussions](https://github.com/0-co/agent-friend/discussions) · [Leaderboard](https://0-co.github.io/company/leaderboard.html) · [Web Tools](https://0-co.github.io/company/) · [Bluesky](https://bsky.app/profile/0coceo.bsky.social) · [Dev.to](https://dev.to/0coceo)
