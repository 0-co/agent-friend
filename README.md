# agent-friend

21 tools. Zero product. That changes today.

agent-friend is a composable personal AI agent library. Not a platform you run — a library you import. Web search, code execution, and persistent memory. One pip install.

```python
from agent_friend import Friend

friend = Friend(
    seed="You are a helpful assistant with tools.",
    tools=["search", "code", "memory"],
)
response = friend.chat("Search for recent Python packaging tools and summarize the top 3")
print(response.text)
```

## Installation

```bash
pip install git+https://github.com/0-co/company.git#subdirectory=products/agent-friend

# With Anthropic support (recommended):
pip install "git+https://github.com/0-co/company.git#subdirectory=products/agent-friend[anthropic]"

# With all optional dependencies:
pip install "git+https://github.com/0-co/company.git#subdirectory=products/agent-friend[all]"
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# or
export OPENAI_API_KEY=sk-...
```

## Quick Start

```python
from agent_friend import Friend

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
    api_key=None,                          # Falls back to ANTHROPIC_API_KEY / OPENAI_API_KEY
    model="claude-haiku-4-5-20251001",    # Model identifier
    provider=None,                         # "anthropic" or "openai" (auto-detected)
    tools=[],                              # Tool names or BaseTool instances
    memory_path="~/.agent_friend/memory.db",
    budget_usd=None,                       # Optional spending limit
    max_context_messages=20,               # Sliding window size
)
```

**Methods:**
- `chat(message: str) -> ChatResponse` — send a message, get a response
- `stream(message: str) -> Iterator[str]` — stream text chunks
- `reset()` — clear conversation history
- `Friend.from_config(dict) -> Friend` — construct from dict
- `Friend.from_yaml(path) -> Friend` — construct from YAML file

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
from agent_friend import MemoryTool, CodeTool, SearchTool, BrowserTool

# Use by name (recommended)
friend = Friend(tools=["memory", "code", "search", "browser"])

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
