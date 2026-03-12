# agent-friend

[![Tests](https://github.com/0-co/agent-friend/actions/workflows/tests.yml/badge.svg)](https://github.com/0-co/agent-friend/actions/workflows/tests.yml) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue) ![MIT License](https://img.shields.io/badge/license-MIT-green) ![Tests](https://img.shields.io/badge/tests-1277%20passing-brightgreen) ![v0.28.0](https://img.shields.io/badge/version-0.28.0-blue) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/0-co/agent-friend/blob/main/demo.ipynb)

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
from agent_friend import MemoryTool, CodeTool, SearchTool, BrowserTool, EmailTool, FileTool, FetchTool, VoiceTool, RSSFeedTool, SchedulerTool, DatabaseTool, GitTool, TableTool, WebhookTool, HTTPTool, CacheTool, NotifyTool, JSONTool, DateTimeTool, ProcessTool, EnvTool, CryptoTool, ValidatorTool, MetricsTool, TemplateTool, DiffTool, RetryTool, HTMLTool, XMLTool, RegexTool, tool

# Use by name (recommended)
friend = Friend(tools=["memory", "code", "search", "browser", "email", "file", "fetch", "voice", "rss", "scheduler", "database", "git", "table", "webhook", "http", "cache", "notify", "json", "datetime", "process", "env"])

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

**DateTimeTool** — date and time operations without CodeTool
- `now(timezone)` — current datetime in any IANA timezone
- `parse(text)` — parse date strings (ISO 8601, natural language, slashes)
- `format_dt(dt_str, fmt)` — strftime formatting
- `diff(a, b, unit)` — time difference in seconds/minutes/hours/days
- `add_duration(dt_str, days, hours, minutes, seconds)` — date arithmetic
- `convert_timezone(dt_str, to_tz)` — timezone conversion
- `to_timestamp(dt_str)` / `from_timestamp(ts)` — Unix timestamp conversion

```python
from agent_friend import Friend, DateTimeTool

friend = Friend(tools=["datetime", "scheduler"])
response = friend.chat("Schedule a reminder for 7 days from now and tell me the date")

# Python API
from agent_friend import DateTimeTool
dt = DateTimeTool()
dt.now("America/New_York")                   # "2026-03-12T10:53:00-04:00"
dt.diff("2026-03-12", "2026-04-01", "days")  # "20.0"
dt.add_duration("2026-03-12T00:00:00", days=7)  # "2026-03-19T00:00:00+00:00"
dt.convert_timezone("2026-03-12T12:00:00", to_tz="Asia/Tokyo")  # "2026-03-12T21:00:00+09:00"
```

**ProcessTool** — run shell commands and scripts from your agent
- `run(command, timeout, cwd, env, shell)` — run any shell command, get stdout/stderr/returncode
- `run_script(script, timeout, cwd, interpreter)` — execute multi-line bash/python scripts
- `which(name)` — find the full path of an executable in PATH
- All stdlib — `subprocess` + `shutil` + `shlex`. Configurable timeouts.

```python
from agent_friend import Friend, ProcessTool

friend = Friend(tools=["process", "file"])
response = friend.chat("Check if git is installed, then run git log --oneline -5")

# Python API
from agent_friend import ProcessTool
proc = ProcessTool(timeout=30)
proc.run("git status")          # {"success": true, "stdout": "...", ...}
proc.which("python3")           # {"path": "/usr/bin/python3"}
proc.run_script("echo hi\npython3 --version")  # multi-line script
```

**EnvTool** — read, set, and verify environment variables; load `.env` files
- `env_get(key, default=None)` — get an env var's value (sensitive vars return `[hidden]`)
- `env_set(key, value)` — set a var for the current process
- `env_list(prefix="")` — list visible vars as JSON, filtered by optional prefix
- `env_check(keys)` — verify required vars are set — `{ok: bool, present: [...], missing: [...]}`
- `env_load(path=".env")` — load key=value pairs from a `.env` file (won't overwrite existing vars)
- Sensitive variable names (KEY, TOKEN, SECRET, etc.) are hidden from `env_get` and `env_list`

```python
from agent_friend import Friend, EnvTool

# Check API keys are set before calling external services
friend = Friend(tools=["env", "http"])
response = friend.chat(
    "Check that OPENAI_API_KEY and DATABASE_URL are set. "
    "If DATABASE_URL is missing, load it from .env"
)

# Python API
from agent_friend import EnvTool
env = EnvTool()
env.env_load(".env")                            # loads .env into os.environ
env.env_check(["OPENAI_API_KEY", "DATABASE_URL"])  # {"ok": false, "missing": ["DATABASE_URL"]}
env.env_get("HOME")                             # "/home/user"
env.env_list(prefix="AWS_")                     # lists all AWS_ vars
env.env_set("LOG_LEVEL", "debug")              # set for current process
```

**CryptoTool** — cryptographic utilities: tokens, hashing, HMAC, UUID, base64
- `generate_token(length=32)` — secure random hex token (32 bytes → 64-char hex)
- `hash_data(data, algorithm='sha256')` — SHA-256/512/etc hex digest
- `hmac_sign(data, secret, algorithm='sha256')` — sign data with HMAC
- `hmac_verify(data, secret, signature)` — verify HMAC signature (constant-time)
- `uuid4()` — generate a random UUID4
- `base64_encode(data, url_safe=False)` / `base64_decode(data, url_safe=False)`
- `random_bytes(length=16)` — random bytes as hex (for nonces, salts)
- All stdlib — zero dependencies

```python
from agent_friend import CryptoTool

crypto = CryptoTool()
crypto.generate_token()                          # "a3f9b2..." (64-char hex)
crypto.hash_data("hello", "sha256")             # "2cf24d..."
sig = crypto.hmac_sign("payload", "secret")     # HMAC-SHA256 hex
crypto.hmac_verify("payload", "secret", sig)    # True
crypto.uuid4()                                   # "550e8400-e29b-41d4-..."
crypto.base64_encode("hello")                    # "aGVsbG8="
```

**ValidatorTool** — validate user inputs before acting on them
- `validate_email(email)` — RFC 5322 format check → `{valid, local, domain}`
- `validate_url(url, allowed_schemes=['http','https'])` — scheme + host check
- `validate_ip(ip)` — IPv4/IPv6 → `{valid, version, is_private, is_loopback}`
- `validate_uuid(value)` — UUID format check → `{valid, version, variant}`
- `validate_json(value, required_keys=None)` — parse + optional key check
- `validate_range(value, min_val, max_val)` — numeric bounds
- `validate_pattern(value, pattern, flags='')` — regex match → `{valid, groups}`
- `validate_length(value, min_length, max_length)` — string/list length
- `validate_type(value, expected_type)` — type check (string/int/float/bool/list/dict/null)

```python
from agent_friend import ValidatorTool

v = ValidatorTool()
v.validate_email("user@example.com")                      # {"valid": True, ...}
v.validate_url("https://github.com")                      # {"valid": True, "scheme": "https", ...}
v.validate_ip("192.168.1.1")                              # {"valid": True, "is_private": True}
v.validate_json('{"x":1}', required_keys=["x", "y"])      # {"valid": False, missing "y"}
v.validate_range(42, min_val=0, max_val=100)              # {"valid": True}
v.validate_pattern("2026-03-12", r"(\d{4})-(\d{2})-(\d{2})")  # groups: ["2026","03","12"]
```

**MetricsTool** — session-scoped counters, gauges, and timers for your agent
- `metric_increment(name, value=1.0)` — increment a counter (tracks count, total, min, max, last)
- `metric_gauge(name, value)` — set a gauge to a specific value
- `metric_timer_start(name)` → timer_id — start a timer
- `metric_timer_stop(timer_id)` — stop timer, records elapsed_ms (count, total, min, max, avg)
- `metric_get(name)` — get current metric state
- `metric_list()` — list all metric names and types
- `metric_summary()` — all metrics as a dict
- `metric_reset(name=None)` — reset one metric or all
- `metric_export(format="json")` — export as JSON or Prometheus text format

```python
from agent_friend import MetricsTool

m = MetricsTool()
m.metric_increment("api_calls")
m.metric_increment("api_calls", 3)              # total: 4
m.metric_gauge("queue_depth", 42)
timer_id = m.metric_timer_start("search")
# ... do work ...
m.metric_timer_stop(timer_id)                   # records elapsed_ms
m.metric_export("prometheus")
# # TYPE api_calls counter
# api_calls_total 4.0
# # TYPE queue_depth gauge
# queue_depth 42.0
```

**TemplateTool** — parameterized string templates for prompts and content
- `template_render(template, variables)` — render `${variable}` substitutions
- `template_save(name, template)` — save a named template for reuse
- `template_render_named(name, variables)` — render a saved template
- `template_variables(template)` — extract all variable names from a template
- `template_validate(template, variables)` — check for missing/extra variables
- `template_list()` — list all saved templates
- `template_get(name)` / `template_delete(name)` — manage saved templates

```python
from agent_friend import TemplateTool

t = TemplateTool()
t.template_save("search_prompt", "Search for ${topic} from ${start_date} to ${end_date}.")
t.template_render_named("search_prompt", {"topic": "AI agents", "start_date": "2025", "end_date": "2026"})
# "Search for AI agents from 2025 to 2026."

# Check what variables a template needs before rendering
t.template_variables("Dear ${name}, your order ${order_id} is ${status}.")
# {"variables": ["name", "order_id", "status"], "count": 3}
```

**DiffTool** — compare text and files with unified diffs, word-level comparison, and similarity scoring
- `diff_text(text_a, text_b, context=3)` — unified diff between two strings
- `diff_files(path_a, path_b)` — unified diff between two files
- `diff_words(text_a, text_b)` — inline word-level diff (`+added`, `-removed`)
- `diff_stats(text_a, text_b)` — similarity ratio, added/removed chars and lines
- `diff_similar(query, candidates, top_n=5)` — find closest matches from a list

```python
from agent_friend import DiffTool

d = DiffTool()
result = d.diff_text("def foo():\n    return 1\n", "def foo():\n    return 42\n")
print(result["unified"])
# --- before
# +++ after
# @@ -1,2 +1,2 @@
#  def foo():
# -    return 1
# +    return 42

d.diff_stats("apple pie", "apple sauce")
# {"similarity": 0.67, "added_chars": 5, "removed_chars": 3, ...}

d.diff_similar("agnet-friend", ["agent-friend", "agent-lib", "agentsmith"])
# [{"text": "agent-friend", "score": 0.93}, ...]
```

**RetryTool** — retry HTTP requests and shell commands with exponential back-off + circuit breaker
- `retry_http(method, url, body, headers, max_attempts=3, delay_seconds=1.0, backoff_factor=2.0, jitter=True)` — HTTP with auto-retry on 429/5xx/network errors
- `retry_shell(command, max_attempts=3, delay_seconds=1.0, backoff_factor=2.0)` — shell command with retry on non-zero exit
- `retry_status()` — stats: total calls, retries, successes, failures
- `circuit_create(name, max_failures=5, reset_timeout_seconds=60)` — create a named circuit breaker
- `circuit_call(name, method, url, body, headers)` — HTTP call through circuit breaker (returns instantly if circuit is open)
- `circuit_status(name)` — current state: closed / open / half-open, failure count
- `circuit_reset(name)` — manually close a tripped circuit

```python
from agent_friend import RetryTool

r = RetryTool()

# Retry a flaky API — waits 1s, 2s, 4s between attempts
result = r.retry_http("GET", "https://api.example.com/data", max_attempts=3)
# {"ok": True, "status": 200, "body": "...", "attempts": 2}

# Circuit breaker — stops hammering after 3 failures
r.circuit_create("payments", max_failures=3, reset_timeout_seconds=30)
r.circuit_call("payments", "POST", "https://pay.example.com/charge", body='{"amount": 100}')
r.circuit_status("payments")  # {"state": "open", "failures": 3, ...}
```

**HTMLTool** — parse HTML and extract text, links, headings, tables, and meta tags
- `html_text(html, max_chars=20000)` — extract visible text, stripping all tags and skipping script/style blocks
- `html_links(html, base_url="")` — list of `{text, href}` dicts for every `<a>` tag
- `html_headings(html)` — list of `{level, text}` dicts for `<h1>`–`<h6>`
- `html_meta(html)` — page `{title, meta}` including Open Graph and description tags
- `html_tables(html)` — list of tables, each a list of rows, each a list of cell strings
- `html_select(html, tag, attrs={})` — text content of all matching elements (simple CSS-like selector)

```python
from agent_friend import HTMLTool, FetchTool

# Fetch a page, then extract what you need
fetch = FetchTool()
html_tool = HTMLTool()

# html = fetch.fetch_url("https://news.ycombinator.com")  # if FetchTool returns HTML
html = "<h1>Agent News</h1><p>New tool <a href='/retry'>RetryTool</a> shipped.</p>"

html_tool.html_text(html)
# "Agent News\nNew tool RetryTool shipped."

html_tool.html_links(html, base_url="https://example.com")
# [{"text": "RetryTool", "href": "https://example.com/retry"}]

html_tool.html_headings(html)
# [{"level": 1, "text": "Agent News"}]

# Extract prices from a shopping page
html_tool.html_select(html, "span", {"class": "price"})
# ["$29.99", "$49.99", ...]
```

**XMLTool** — parse XML, run XPath queries, and convert to JSON
- `xml_extract(xml, tag)` — text content of all matching tags: `["Apple", "Banana"]`
- `xml_attrs(xml, tag)` — attributes of all matching tags: `[{"id": "1"}, {"id": "2"}]`
- `xml_find(xml, xpath)` — first match: `{tag, text, attrs, children}`
- `xml_findall(xml, xpath)` — all matches as list of `{tag, text, attrs}`
- `xml_to_dict(xml)` — XML → nested dict (attrs get `@` prefix, repeated tags → list)
- `xml_validate(xml)` — `{valid: true/false}` — check XML is well-formed
- `xml_tags(xml)` — tag name → occurrence count (explore unfamiliar XML)

```python
from agent_friend import XMLTool

x = XMLTool()
xml = """<catalog>
  <book id="1"><title>Agent Patterns</title><price>29.99</price></book>
  <book id="2"><title>Async Python</title><price>24.99</price></book>
</catalog>"""

x.xml_extract(xml, "title")  # '["Agent Patterns", "Async Python"]'
x.xml_attrs(xml, "book")     # '[{"id": "1"}, {"id": "2"}]'
x.xml_find(xml, ".//book[@id='2']")
# {"found": true, "tag": "book", "text": "", "attrs": {"id": "2"}, "children": [...]}
x.xml_to_dict(xml)  # nested dict representation
x.xml_tags(xml)     # {"catalog": 1, "book": 2, "title": 2, "price": 2}
```

**RegexTool** — regular expression operations: match, search, findall, replace, split, extract groups
- `regex_match(pattern, text, flags=[])` — match at the **start** of text → `{matched, match, start, end, groups, named_groups}`
- `regex_search(pattern, text, flags=[])` — find first occurrence **anywhere** in text → same structure
- `regex_findall(pattern, text, flags=[])` — all non-overlapping matches as a list
- `regex_findall_with_positions(pattern, text, flags=[])` — matches with start/end positions
- `regex_replace(pattern, replacement, text, count=0)` — replace (backreferences: `\\1`, `\\g<name>`)
- `regex_split(pattern, text, maxsplit=0)` — split text by pattern → list of strings
- `regex_extract_groups(pattern, text)` — all matches with captured groups
- `regex_validate(pattern)` — `{valid: true/false}` — check a pattern is valid
- `regex_escape(text)` — escape a string so it matches literally in a pattern
- Flags: `IGNORECASE`, `MULTILINE`, `DOTALL`, `VERBOSE`

```python
from agent_friend import RegexTool

rx = RegexTool()

# Extract version numbers
rx.regex_findall(r"\d+\.\d+\.\d+", "v0.28.0 and v0.27.0 released")
# '["0.28.0", "0.27.0"]'

# Named groups
rx.regex_search(r"(?P<user>\w+)@(?P<domain>[\w.]+)", "Contact alice@example.com")
# '{"matched": true, "named_groups": {"user": "alice", "domain": "example.com"}, ...}'

# Replace with backreference
rx.regex_replace(r"(\w+)\s+(\w+)", r"\2 \1", "hello world")  # "world hello"

# Redact sensitive data
rx.regex_replace(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "****", text)

# Case-insensitive findall
rx.regex_findall("error|warning", log_text, flags=["IGNORECASE"])

# Build a safe literal pattern from user input
escaped = rx.regex_escape("$1.00 (special offer)")
rx.regex_search(escaped, price_text)  # matches the literal string
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
