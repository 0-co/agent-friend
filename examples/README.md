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

### `research_assistant.py` — research any topic, save to markdown

Searches, fetches sources, writes a structured markdown summary, saves to file, remembers it.

```bash
python3 examples/research_assistant.py "LLM agent memory systems"
python3 examples/research_assistant.py "Python async patterns" --output report.md
python3 examples/research_assistant.py --depth quick "AI agent tools"
```

---

## Full demo scripts

The root directory also contains:

- `demo_live.py` — interactive REPL showing tool calls in real time
- `demo_briefing.py` — daily briefing with email + search + memory
- `demo.ipynb` — Colab notebook with 7 demos (no install required)
