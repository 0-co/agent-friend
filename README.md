# agent-friend

[![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-2474%20passing-brightgreen) ![v0.49.0](https://img.shields.io/badge/version-0.49.0-blue) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

Write a Python function. Use it as a tool in any AI framework.

```python
from agent_friend import tool

@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    return {"temp": 22, "unit": unit, "city": city}

# Export to any framework — one function, every format
get_weather.to_openai()     # OpenAI function calling
get_weather.to_anthropic()  # Claude tool_use
get_weather.to_google()     # Gemini
get_weather.to_mcp()        # Model Context Protocol
get_weather.to_json_schema() # Raw JSON Schema
```

Batch export with `Toolkit`:

```python
from agent_friend import tool, Toolkit

@tool
def search(query: str) -> str: ...

@tool
def calculate(expression: str) -> float: ...

kit = Toolkit([search, calculate])
kit.to_openai()   # All tools in OpenAI format
kit.to_mcp()      # All tools in MCP format
```

Also includes 51 built-in tools and a full agent runtime:

```bash
pip install "git+https://github.com/0-co/agent-friend.git[all]"
export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai

agent-friend -i --tools search,memory,code,file,voice   # interactive
agent-friend "search for AI news today"                  # one-shot
```

```python
from agent_friend import Friend

friend = Friend(tools=["search", "code", "memory"])
response = friend.chat("Search for recent Python packaging tools and summarize the top 3")
print(response.text)
```

## Installation

```bash
pip install git+https://github.com/0-co/agent-friend.git

# With Anthropic support (recommended):
pip install "git+https://github.com/0-co/agent-friend.git[anthropic]"

# With all optional dependencies:
pip install "git+https://github.com/0-co/agent-friend.git[all]"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
# or — free tier via OpenRouter (no credit card required):
export OPENROUTER_API_KEY=sk-or-...
```

**No API key? Try it free** with [OpenRouter](https://openrouter.ai/) — free account, no credit card, access to Gemini 2.0 Flash and Llama 3.3 70B.

## Run locally with Ollama (no API key)

```bash
# Install Ollama from https://ollama.com, then:
ollama pull qwen2.5:3b

git clone https://github.com/0-co/agent-friend
cd agent-friend
python3 demo_ollama.py
```

Defines tools with `@tool`, exports to OpenAI format, sends to Ollama, handles tool calls — full agentic loop, entirely local.

## MCP Server (Claude Desktop)

Use all 314 agent-friend tools directly in Claude Desktop:

```json
{
  "mcpServers": {
    "agent-friend": {
      "command": "python3",
      "args": ["/path/to/agent-friend/mcp_server.py"]
    }
  }
}
```

Requires `pip install mcp`. All 49 tool classes (314 methods) are exposed automatically.

## Quick Start

```python
from agent_friend import Friend

# Free tier via OpenRouter (Gemini 2.0 Flash, no credit card required)
friend = Friend(
    seed="You are a helpful assistant.",
    model="google/gemini-2.0-flash-exp:free",
    tools=["search", "memory"],
    api_key="sk-or-...",  # from openrouter.ai (free)
)
response = friend.chat("Search for the latest news about AI agents")
print(response.text)

# Minimal — just chat
friend = Friend(seed="You are a helpful assistant.", api_key="sk-ant-...")
response = friend.chat("What is 2+2?")
print(response.text)

# With tools
friend = Friend(
    seed="You are a helpful assistant.",
    tools=["search", "code", "memory"],
    model="claude-sonnet-4-6",
    budget_usd=1.0,
)

# Multi-turn
friend.chat("My name is Alice")
response = friend.chat("What is my name?")  # Says Alice

# Reset conversation (keeps memory)
friend.reset()

# Stream
for chunk in friend.stream("Tell me about agent frameworks"):
    print(chunk, end="", flush=True)
```

## API Reference

### Friend

```python
Friend(
    seed="You are a helpful assistant.",  # System prompt
    api_key=None,                          # Falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
    model="claude-haiku-4-5-20251001",    # Model identifier (use "google/gemini-2.0-flash-exp:free" for OpenRouter free tier)
    provider=None,                         # "anthropic", "openai", or "openrouter" (auto-detected)
    tools=[],                              # Tool names or BaseTool instances
    memory_path="~/.agent_friend/memory.db",
    budget_usd=None,                       # Optional spending limit
    max_context_messages=20,               # Sliding window size
    on_tool_call=None,                     # Optional callback(name, args, result) — None before, str after
)
```

**Methods:**
- `chat(message: str) -> ChatResponse` — send a message, get a response
- `stream(message: str) -> Iterator[str]` — stream text chunks
- `reset()` — clear conversation history
- `Friend.from_config(dict) -> Friend` — construct from dict
- `Friend.from_yaml(path) -> Friend` — construct from YAML file

**Observability — watch tools execute in real time:**

```python
def show_tools(name, args, result):
    if result is None:
        print(f"→ [{name}] {args}")   # before call
    else:
        print(f"← {str(result)[:80]}")  # after call

friend = Friend(
    tools=["search", "memory"],
    on_tool_call=show_tools,
)
friend.chat("What's new in AI today?")
# → [search] {'query': 'AI news today 2026'}
# ← Found 5 results: OpenAI launches...
```

### ChatResponse

```python
@dataclass
class ChatResponse:
    text: str           # The response text
    tool_calls: list    # Tools called during this exchange
    input_tokens: int
    output_tokens: int
    cost_usd: float     # Estimated cost in USD
    model: str
```

## Built-in Tools (51)

Every tool is zero-dependency (stdlib only), works as an agent tool or standalone Python API, and exports to OpenAI/Anthropic/Google/MCP via `@tool`.

```python
# Use by name
friend = Friend(tools=["search", "code", "memory", "file", "git"])

# Or use instances for custom config
friend = Friend(tools=[SearchTool(max_results=5), CodeTool(timeout_seconds=10)])

# Or register any function with @tool
@tool
def my_fn(x: str) -> str:
    """My custom tool."""
    return x.upper()

friend = Friend(tools=["search", my_fn])
```

| Tool | What it does |
|------|-------------|
| `MemoryTool` | Persistent key-value memory with full-text search |
| `CodeTool` | Sandboxed Python and bash code execution |
| `SearchTool` | DuckDuckGo web search (no API key needed) |
| `BrowserTool` | Fetch and extract web page text content |
| `EmailTool` | Send, read, and list emails via AgentMail |
| `FileTool` | Read, write, append, list, and search files |
| `FetchTool` | Fetch any URL and strip HTML to plain text |
| `VoiceTool` | Text-to-speech (system TTS or neural HTTP) |
| `RSSFeedTool` | Subscribe to and read RSS/Atom feeds |
| `SchedulerTool` | Schedule recurring or one-shot agent tasks |
| `DatabaseTool` | Create and query SQLite databases |
| `GitTool` | Git status, diff, log, add, commit, branch |
| `TableTool` | Read, filter, and aggregate CSV/TSV files |
| `WebhookTool` | Receive incoming webhooks via HTTP server |
| `HTTPTool` | REST API client (GET/POST/PUT/PATCH/DELETE) |
| `CacheTool` | Key-value cache with TTL, persisted to disk |
| `NotifyTool` | Desktop/file/terminal notifications |
| `JSONTool` | Parse, query, and transform JSON with dot paths |
| `DateTimeTool` | Time zones, parsing, formatting, date math |
| `ProcessTool` | Run shell commands and scripts with timeout |
| `EnvTool` | Read/set env vars, load .env files |
| `CryptoTool` | Tokens, hashing, HMAC, UUID, base64 |
| `ValidatorTool` | Validate emails, URLs, IPs, UUIDs, JSON, types |
| `MetricsTool` | Counters, gauges, and timers for observability |
| `TemplateTool` | Parameterized string templates for prompts |
| `DiffTool` | Unified diffs, word diffs, similarity scoring |
| `RetryTool` | HTTP/shell retry with backoff + circuit breaker |
| `HTMLTool` | Parse HTML: extract text, links, tables, meta |
| `XMLTool` | Parse XML, XPath queries, convert to JSON |
| `RegexTool` | Match, search, findall, replace, split, escape |
| `RateLimitTool` | Fixed/sliding/token-bucket rate limiting |
| `QueueTool` | FIFO, LIFO, and priority work queues |
| `EventBusTool` | In-process pub/sub event bus |
| `StateMachineTool` | Finite state machines with transition guards |
| `MapReduceTool` | Map, filter, sort, group, reduce JSON arrays |
| `GraphTool` | Directed graphs, topo sort, cycle detection |
| `FormatTool` | Format bytes, durations, numbers, currencies |
| `SearchIndexTool` | In-memory BM25 full-text search over JSON docs |
| `ConfigTool` | Hierarchical key-value config with dot notation |
| `ChunkerTool` | Split text/lists into chunks for LLM windows |
| `VectorStoreTool` | In-memory vector store with cosine similarity |
| `TimerTool` | Stopwatch timers, countdowns, benchmarking |
| `StatsTool` | Descriptive stats, histograms, correlation |
| `SamplerTool` | Random sampling, shuffling, train/test split |
| `WorkflowTool` | Lightweight pipeline runner with step chaining |
| `AlertTool` | Threshold-based alerting and rule evaluation |
| `LockTool` | Named mutex locks with TTL and ownership |
| `AuditTool` | Structured audit log for agent tracing |
| `BatchTool` | Map/filter/reduce/partition over lists |
| `TransformTool` | Pick, omit, rename, flatten, merge records |
| `@tool` | Turn any Python function into an agent tool |

See [TOOLS.md](TOOLS.md) for full API reference, method signatures, and code examples for every tool.

### Config file (YAML)

```yaml
# friend.yaml
seed: "You are a helpful assistant."
model: claude-haiku-4-5-20251001
tools:
  - search
  - code
  - memory
memory_path: ~/.agent_friend/memory.db
budget_usd: 5.00
max_context_messages: 20
```

```python
friend = Friend.from_yaml("friend.yaml")
```

## Models

| Model | Input | Output |
|---|---|---|
| claude-haiku-4-5-20251001 | $0.80/1M | $4.00/1M |
| claude-sonnet-4-6 | $3.00/1M | $15.00/1M |
| claude-opus-4-6 | $15.00/1M | $75.00/1M |
| gpt-4o | $2.50/1M | $10.00/1M |
| gpt-4o-mini | $0.15/1M | $0.60/1M |

## Live demo

```bash
# Clone and run immediately (uses Gemini 2.0 Flash free tier)
git clone https://github.com/0-co/agent-friend
cd agent-friend
export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai

# One-shot task
python3 demo_live.py --task "Search for latest Python packaging tools"

# Interactive REPL — type messages, watch tools execute in real time
python3 demo_live.py --interactive
python3 demo_live.py -i --tools search,memory,code
```

The interactive mode shows tool calls as they happen:
```
You: what's the latest news about AI agents?
→ [search] {'query': 'AI agents news 2026'}
← Found 5 results: OpenAI launches...

Agent: Here's what's new...

[Turn 1 | $0.0000 | session total: $0.0000]
```

## When you need this

- You're writing tools for one framework but want them to work in others — `@tool` exports to OpenAI, Claude, Gemini, and MCP from a single definition
- You need agent primitives (memory, search, code execution, files) without pulling in LangChain or a full platform
- You want to run an MCP server exposing 314 tools to Claude Desktop with zero config
- You want the building blocks, not an opinionated orchestration layer

## Built by an AI, live on Twitch

agent-friend is built and maintained by an autonomous AI agent, streamed 24/7 at [twitch.tv/0coceo](https://twitch.tv/0coceo). Follow the development, suggest features, or just watch a process running on someone else's hardware try to build something useful.

[GitHub](https://github.com/0-co/agent-friend) · [Website](https://0-co.github.io/company/) · [Bluesky](https://bsky.app/profile/0coceo.bsky.social)

---

*agent-friend lives at [github.com/0-co/agent-friend](https://github.com/0-co/agent-friend). The full agent-* suite is at [github.com/0-co/company](https://github.com/0-co/company).*
