# agent-friend

[![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-640%20passing-brightgreen) ![v0.16.0](https://img.shields.io/badge/version-0.16.0-blue) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

A personal AI agent library. Memory, web search, code execution, scheduled tasks, SQLite databases — one pip install.

```bash
# Free, no credit card required (OpenRouter)
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
from agent_friend import MemoryTool, CodeTool, SearchTool, BrowserTool, EmailTool, FileTool, FetchTool, VoiceTool, RSSFeedTool, SchedulerTool, DatabaseTool, GitTool, TableTool, WebhookTool, HTTPTool, CacheTool, NotifyTool, JSONTool, tool

# Use by name (recommended)
friend = Friend(tools=["memory", "code", "search", "browser", "email", "file", "fetch", "voice", "rss", "scheduler", "database", "git", "table", "webhook", "http", "cache", "notify", "json"])

# Or use instances for custom config
friend = Friend(tools=[
    MemoryTool(db_path="~/.my_agent/memory.db"),
    CodeTool(timeout_seconds=10),
    SearchTool(max_results=5),
])

# Or register any function as a tool with @tool
@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny in {city}, 22°C"  # replace with real API call

friend = Friend(tools=["search", get_weather])
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

**VoiceTool** — Text-to-speech for your agent (zero required dependencies)
- `speak(text, voice=None)` — speaks text aloud or saves to MP3 file
- System TTS: espeak/espeak-ng (Linux), `say` (macOS), PowerShell (Windows)
- Neural TTS: set `AGENT_FRIEND_TTS_URL` to use any HTTP TTS server for high-quality voices
- Saves audio to `~/.agent_friend/voice/` when using HTTP backend
- Lets your agent narrate its responses, read documents aloud, or generate audio files

**RSSFeedTool** — Subscribe to and read RSS/Atom feeds (zero required dependencies)
- `subscribe(url, name)` — save a feed by name for quick access
- `list_feeds()` — list subscribed feeds
- `read_feed(name, count=5)` — get latest items from a subscribed feed
- `fetch_feed(url, count=5)` — fetch any RSS/Atom URL directly
- `unsubscribe(name)` — remove a subscribed feed
- Supports RSS 2.0, Atom, and RSS 1.0. Strips HTML from summaries automatically.

**SchedulerTool** — Schedule tasks for your agent to run on a timer or at a specific time
- `schedule(task_id, prompt, interval_minutes=None, run_at=None)` — create a recurring or one-shot task
- `run_pending()` — check and return tasks that are due (use with `agent-friend schedule` CLI)
- `list_scheduled()` — see all scheduled tasks and their next run times
- `cancel(task_id)` — remove a scheduled task
- `clear_all()` — remove all tasks
- Stores schedule in `~/.agent_friend/scheduler.json`. Zero dependencies.

**DatabaseTool** — Create and query SQLite databases (zero dependencies)
- `db_execute(sql, params=[])` — CREATE TABLE, INSERT, UPDATE, DELETE
- `db_query(sql, params=[])` — SELECT and return results as a formatted table
- `db_tables()` — list all tables in the database
- `db_schema(table)` — get the CREATE TABLE statement for any table
- Python API: `create_table()`, `insert()`, `query()`, `run()`, `list_tables()`, `get_schema()`
- Backed by `~/.agent_friend/agent.db`. Your agent can store and query structured data persistently.

**GitTool** — read and commit to git repositories (requires git installed)
- `git_status(repo_dir)` — working tree status
- `git_diff(staged, path, repo_dir)` — unstaged or staged diff
- `git_log(n, oneline, repo_dir)` — commit history
- `git_add(paths, repo_dir)` — stage files for commit
- `git_commit(message, repo_dir)` — commit staged changes
- `git_branch_list(repo_dir)` — list all local branches
- `git_branch_create(name, checkout, repo_dir)` — create a new branch
- Python API: `git.status()`, `git.diff()`, `git.log()`, `git.add()`, `git.commit()`, `git.branch_list()`, `git.branch_create()`

```python
from agent_friend import Friend, GitTool

# Point at a specific repo
git = GitTool(repo_dir="/path/to/repo")
friend = Friend(tools=["search", "code", "file", git])
friend.chat("Show me the git status and recent commits")
friend.chat("Stage all changes to src/ and commit with message 'Refactor auth flow'")

# Default: uses current working directory
friend = Friend(tools=["git"])
friend.chat("What changed in the last 5 commits?")
```

**TableTool** — read, filter, and aggregate CSV/TSV files (no pandas)
- `table_read(filepath)` — read CSV/TSV, return rows as JSON
- `table_columns(filepath)` — list column names
- `table_filter(filepath, column, operator, value)` — filter rows (eq/ne/gt/lt/gte/lte/contains/startswith)
- `table_aggregate(filepath, column, operation)` — count/sum/avg/min/max/unique over a column
- `table_write(filepath, rows, delimiter)` — write rows to CSV
- Python API: `read()`, `write()`, `columns()`, `filter_rows()`, `aggregate()`, `append_row()`
- Auto-detects delimiter (comma vs tab). Zero dependencies.

```python
from agent_friend import Friend, TableTool

table = TableTool()
friend = Friend(tools=["search", "code", table])
friend.chat("Read sales.csv and tell me the average revenue by region")
friend.chat("Filter transactions.csv to rows where amount > 1000")
```

**WebhookTool** — receive incoming webhooks (payment callbacks, GitHub events, form submissions)
- `wait_for_webhook(path, timeout)` — start HTTP server and wait for a POST request
- Returns: path, headers, body (str), json (parsed dict or None), received_at timestamp
- Port 0 = auto-assign random available port. Server shuts down after receiving one request.

```python
from agent_friend import Friend, WebhookTool

# Agent waits for a payment webhook, then reacts
hook = WebhookTool(port=8765)
friend = Friend(tools=["code", "memory", hook])
response = friend.chat(
    "Wait for a webhook at /payment with 60 second timeout. "
    "When it arrives, log the amount to memory."
)
# In another terminal: curl -X POST http://localhost:8765/payment -d '{"amount": 99.99}'
```

**HTTPTool** — generic REST API client (GET/POST/PUT/PATCH/DELETE with auth headers)
- `http_request(method, url, headers, body, body_text)` — make any HTTP request
- Returns: status code, response headers, body (str), json (parsed dict if JSON response)
- `default_headers` constructor param for auth headers shared across all requests
- No requests library required — stdlib only

```python
from agent_friend import Friend, HTTPTool

# API client with auth headers baked in
http = HTTPTool(default_headers={"Authorization": "Bearer sk-..."})
friend = Friend(tools=["memory", http])
response = friend.chat(
    "POST to https://api.example.com/orders with body {\"item\": \"widget\", \"qty\": 5}"
)

# One-off requests without config
friend = Friend(tools=["search", "http"])
friend.chat("GET https://api.github.com/repos/0-co/agent-friend and summarize the stats")
```

**CacheTool** — key-value cache with TTL expiry, persisted to disk
- `cache_get(key)` — retrieve a cached value (returns `null` if missing or expired)
- `cache_set(key, value, ttl_seconds=3600)` — store a value with optional TTL
- `cache_delete(key)` — remove one entry
- `cache_clear()` — remove all entries
- `cache_stats()` — JSON with entry count, hit/miss counts

```python
from agent_friend import Friend, CacheTool

friend = Friend(tools=["http", "cache"])
response = friend.chat(
    "Fetch the GitHub stars for 0-co/agent-friend. "
    "Cache the result under 'gh_stars' for 1 hour. "
    "If it's already cached, use the cached value."
)

# Python API
cache = CacheTool()
cache.cache_set("weather_nyc", '{"temp": 72, "sky": "clear"}', ttl_seconds=3600)
result = cache.cache_get("weather_nyc")  # returns value within 1 hour, else None
print(cache.cache_stats())  # {"entries": 1, "session_hits": 1, "session_misses": 0, ...}
```

**NotifyTool** — send notifications when tasks complete (desktop, file log, or terminal bell)
- `notify(title, message)` — best available channel (desktop → file fallback)
- `notify_desktop(title, message)` — system notification (notify-send / osascript)
- `notify_file(title, message, path=None)` — append to JSONL log file
- `bell()` — terminal bell character
- `read_notifications(n=10)` — read last N notifications from log

```python
from agent_friend import Friend

# Agent that notifies you when a long task is done
friend = Friend(
    seed="Run the report, then notify the user when complete.",
    tools=["scheduler", "notify"],
)
friend.chat("Run the daily news summary at 8:00 UTC and notify me when it's done")

# Python API — useful in scripts
from agent_friend import NotifyTool
notifier = NotifyTool()
notifier.notify("Report ready", "Daily news summary complete")       # desktop or file
notifier.notify_file("Error", "API timeout after 30s retry")        # always works
entries = notifier.read_notifications(n=5)                           # last 5 entries
```

**JSONTool** — parse, query, and transform JSON data with dot-notation paths
- `json_get(data, path)` — extract value at path (`"user.name"`, `"items[0].id"`, `"users[*].email"`)
- `json_set(data, path, value)` — return modified JSON with value set at path
- `json_keys(data)` — list top-level keys
- `json_filter(data, key, value)` — filter array by key=value
- `json_format(data, indent=2)` — pretty-print
- `json_merge(base, patch)` — merge two objects (patch overrides base)

```python
from agent_friend import Friend, JSONTool

friend = Friend(tools=["http", "json"])
response = friend.chat(
    "GET https://pypi.org/pypi/requests/json and extract the latest version from info.version"
)

# Python API
from agent_friend import JSONTool
jt = JSONTool()
data = '{"user": {"name": "Alice"}, "tags": ["ai", "python"]}'
jt.json_get(data, "user.name")              # '"Alice"'
jt.json_get(data, "tags[0]")               # '"ai"'
jt.json_set(data, "user.email", '"a@b.com"')  # modified JSON
jt.json_filter('[{"role":"admin"},{"role":"user"}]', "role", '"admin"')
```

**Custom Tools via `@tool`** — register any Python function as an agent tool
- Reads type hints to auto-generate the JSON schema
- Optional parameters (with defaults or `Optional[X]`) are not required
- The decorated function remains callable normally
- Mix with built-in tools: `Friend(tools=["search", my_fn])`

```python
from agent_friend import Friend, tool

@tool
def stock_price(ticker: str) -> str:
    """Get current stock price for a ticker symbol."""
    # call your actual API here
    return f"{ticker}: $182.50"

@tool(name="convert_temp", description="Convert Celsius to Fahrenheit")
def to_fahrenheit(celsius: float) -> str:
    return f"{celsius * 9/5 + 32:.1f}°F"

friend = Friend(tools=["search", stock_price, to_fahrenheit])
friend.chat("What's AAPL stock price and convert 22°C to Fahrenheit?")

# Functions still work normally
print(stock_price("AAPL"))    # "AAPL: $182.50"
print(to_fahrenheit(22.0))    # "71.6°F"
```

```python
# System TTS (zero config, works everywhere)
friend = Friend(tools=["voice"])
friend.chat("Read this article summary aloud")

# Neural TTS via HTTP server
from agent_friend import VoiceTool
friend = Friend(tools=[VoiceTool(tts_url="http://your-tts-server:8081")])
```

```python
# Agent with a real database — create tables, insert rows, run queries
from agent_friend import DatabaseTool

friend = Friend(tools=["database"])
friend.chat("Create a tasks table with title and done columns, then add 3 tasks")
friend.chat("Show me all incomplete tasks")

# Python API for scripting
db = DatabaseTool()
db.create_table("notes", "id INTEGER PRIMARY KEY, content TEXT, tag TEXT")
db.insert("notes", {"content": "Ship agent-friend v0.8", "tag": "work"})
rows = db.query("SELECT * FROM notes WHERE tag = ?", ["work"])
```

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
