# agent-friend examples

Ready-to-run scripts showing agent-friend in real workflows.

## Setup

```bash
pip install "git+https://github.com/0-co/agent-friend.git"
export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai (no credit card)
```

## Examples

### `voice_briefing.py` — daily news spoken aloud

Searches for a topic, fetches top articles, summarizes, and speaks the result.

```bash
python3 examples/voice_briefing.py
python3 examples/voice_briefing.py --topic "Python packaging 2026"
python3 examples/voice_briefing.py --no-voice  # text only
```

Requires TTS: `espeak` (Linux), `say` (macOS), or set `AGENT_FRIEND_TTS_URL` for neural voices.

---

### `news_briefing.py` — read your RSS feeds and get a briefing

Subscribe to feeds, fetch latest items, and summarize them with the agent.

```bash
# Quick one-shot
agent-friend --tools rss "subscribe to https://news.ycombinator.com/rss as hn, then read me the top 5 stories"

# Or in Python
python3 -c "
from agent_friend import Friend
f = Friend(tools=['rss', 'memory'], api_key='sk-or-...')
f.chat('subscribe to https://news.ycombinator.com/rss as hn')
print(f.chat('read me the top 5 stories from hn and summarize each in one sentence').text)
"
```

---

### `research_assistant.py` — research any topic, save to markdown

Searches, fetches sources, writes a structured markdown summary, saves to file, remembers it.

```bash
python3 examples/research_assistant.py "LLM agent memory systems"
python3 examples/research_assistant.py "Python async patterns" --output report.md
python3 examples/research_assistant.py --depth quick "AI agent tools"
```

---

### `task_manager.py` — conversational task manager with SQLite database

Creates a SQLite database, inserts tasks, queries by status. Shows both the Python API (no LLM needed) and the agent API.

```bash
# Python API only (no API key needed)
python3 examples/task_manager.py

# With agent (conversational)
export OPENROUTER_API_KEY=sk-or-...
python3 examples/task_manager.py
```

The Python API demo works with zero config — no API key, no external services.

---

## Full demo scripts

The root directory also contains:

- `demo_live.py` — interactive REPL showing tool calls in real time
- `demo_briefing.py` — daily briefing with email + search + memory
- `demo.ipynb` — Colab notebook with 11 demos (no install required)
