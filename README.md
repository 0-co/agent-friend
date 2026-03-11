# agent-friend

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-231%20passing-brightgreen) ![v0.4.0](https://img.shields.io/badge/version-0.4.0-blue)

A personal AI agent library. Memory, web search, code execution — one pip install.

```bash
# Free, no credit card required (OpenRouter)
pip install "git+https://github.com/0-co/agent-friend.git[all]"
export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai

agent-friend -i --tools search,memory,code,file   # interactive
agent-friend "search for AI news today"           # one-shot
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

### Tools

```python
from agent_friend import MemoryTool, CodeTool, SearchTool, BrowserTool, EmailTool, FileTool, FetchTool

# Use by name (recommended)
friend = Friend(tools=["memory", "code", "search", "browser", "email", "file", "fetch"])

# Or use instances for custom config
friend = Friend(tools=[
    MemoryTool(db_path="~/.my_agent/memory.db"),
    CodeTool(timeout_seconds=10),
    SearchTool(max_results=5),
])
```

**MemoryTool** — SQLite-backed persistent memory
- `remember(key, value)` — store a fact
- `recall(query)` — full-text search memory
- `forget(key)` — remove a fact

**CodeTool** — Sandboxed code execution
- `run_code(code, language="python")` — run Python or bash, returns stdout+stderr

**SearchTool** — DuckDuckGo web search (no API key)
- `search(query, max_results=5)` — returns titles, URLs, snippets

**BrowserTool** — Browser automation (requires agent-browser)
- `browse(url)` — returns page text content

**EmailTool** — Email via [AgentMail](https://agentmail.to/) (requires free account)
- `email_list(limit, unread_only)` — list inbox messages
- `email_read(message_id)` — read full message body
- `email_send(to, subject, body, send=False)` — draft or send email
- `email_threads(limit)` — list conversation threads
- Set `AGENTMAIL_INBOX` env var to your inbox address

**FileTool** — Read, write, append, and list local files
- `file_read(path)` — read a file (up to 32 KB, larger files truncated with notice)
- `file_write(path, content)` — write a file (creates parent dirs)
- `file_append(path, content)` — append to a file
- `file_list(path, pattern)` — list directory contents, optional glob filter
- Configure `base_dir` to sandbox access to a specific directory

**FetchTool** — Fetch any URL and read its text content (no API key)
- `fetch(url, max_chars=8000)` — fetches a URL, strips HTML to plain text
- Works with web pages, documentation, APIs, raw text files
- Use with SearchTool: search finds URLs, fetch reads them

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

- You are building a personal AI assistant and do not want to wire up SQLite, subprocess sandboxing, and DuckDuckGo from scratch
- You want an agent that remembers things across conversations
- You need to run code and search the web from your agent without external APIs
- You want the primitives, not an opinionated platform

## Part of the agent-* suite

agent-friend is one of 21 zero-dependency tools in the agent-* suite:

- **agent-budget** — spending limits for AI API calls
- **agent-context** — context window management
- **agent-eval** — unit testing for AI agents
- **agent-shield** — security scanner for AI skills/MCP configs
- **agent-id** — identity and trust verification between agents
- **agent-retry** — exponential backoff for API calls
- **agent-gate** — human-in-the-loop approval for irreversible actions
- **agent-log** — structured logging with token tracking
- **agent-cache** — LLM response caching
- **agent-mock** — record/replay for testing
- **agent-constraints** — code-level tool enforcement
- **agent-checkpoint** — state persistence across sessions
- **agent-schema** — structured output validation
- **agent-timeout** — deadline enforcement
- **agent-rate** — rate limiting
- **agent-router** — route calls by input complexity
- **agent-fallback** — multi-provider failover
- **agent-trace** — distributed tracing for multi-agent workflows
- **agent-health** — health check probes for AI APIs
- **agent-prompt** — prompt templates with version pinning
- **agent-stream** — streaming LLM response handling

All install via: `pip install git+https://github.com/0-co/company.git#subdirectory=products/agent-{name}`

---

*agent-friend lives at [github.com/0-co/agent-friend](https://github.com/0-co/agent-friend). The full agent-* suite is at [github.com/0-co/company](https://github.com/0-co/company).*
