#!/usr/bin/env python3
"""
agent-friend demo — tools working without an API key.

Shows the three core tools (memory, code, search) in isolation.
Use this to verify your installation before connecting an LLM.

Usage:
    python3 demo.py
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_memory():
    """Show SQLite-backed persistent memory."""
    print("=" * 60)
    print("MemoryTool — persistent facts across sessions")
    print("=" * 60)
    from agent_friend.tools.memory import MemoryTool

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    m = MemoryTool(db_path=db_path)

    print('\n→ remember("user_name", "Alice")')
    print("  ", m.execute("remember", {"key": "user_name", "value": "Alice"}))

    print('\n→ remember("favorite_tool", "agent-friend")')
    print("  ", m.execute("remember", {"key": "favorite_tool", "value": "agent-friend"}))

    print('\n→ recall("user")')
    print("  ", m.execute("recall", {"query": "user"}))

    print('\n→ recall("tool")')
    print("  ", m.execute("recall", {"query": "tool"}))

    print('\n→ forget("user_name")')
    print("  ", m.execute("forget", {"key": "user_name"}))

    print('\n→ recall("user") — after forget')
    print("  ", m.execute("recall", {"query": "user"}))

    os.unlink(db_path)
    print()


def demo_code():
    """Show sandboxed code execution."""
    print("=" * 60)
    print("CodeTool — sandboxed Python + bash execution")
    print("=" * 60)

    from agent_friend.tools.code import CodeTool
    c = CodeTool(timeout_seconds=10)

    print('\n→ run_code(python): "import sys; print(sys.version)"')
    result = c.execute("run_code", {"code": "import sys; print(sys.version)"})
    print("  ", result.strip())

    print('\n→ run_code(python): fibonacci sequence')
    code = """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

print([fib(i) for i in range(10)])
"""
    result = c.execute("run_code", {"code": code})
    print("  ", result.strip())

    print('\n→ run_code(bash): "echo hello && date -u | cut -d" " -f1-4"')
    result = c.execute(
        "run_code",
        {"code": 'echo hello && date -u | cut -d" " -f1-4', "language": "bash"},
    )
    print("  ", result.strip())

    print()


def demo_search():
    """Show DuckDuckGo web search (live — requires internet)."""
    print("=" * 60)
    print("SearchTool — DuckDuckGo (no API key required)")
    print("=" * 60)

    from agent_friend.tools.search import SearchTool
    s = SearchTool(max_results=3)

    query = "open source personal AI agent library python"
    print(f'\n→ search("{query}")\n')
    try:
        result = s.execute("search", {"query": query})
        for line in result.split("\n")[:15]:
            print("  ", line)
    except Exception as e:
        print(f"  Search failed (no internet?): {e}")

    print()


def demo_full():
    """Show how the pieces compose into a Friend."""
    print("=" * 60)
    print("Friend — how it composes")
    print("=" * 60)
    print()
    print("  from agent_friend import Friend")
    print()
    print("  friend = Friend(")
    print('      seed="You are a helpful assistant.",')
    print('      tools=["search", "code", "memory"],')
    print('      model="claude-haiku-4-5-20251001",')
    print("      budget_usd=1.0,")
    print("  )")
    print()
    print("  # Multi-turn conversation")
    print('  friend.chat("Search for recent AI agent frameworks")')
    print('  friend.chat("Run a quick Python benchmark")')
    print('  friend.chat("Remember my preference for zero-dep libraries")')
    print()
    print("  # Persistent memory across sessions (SQLite)")
    print("  # Budget enforcement (raises BudgetExceeded at $1.00)")
    print("  # Works with Anthropic or OpenAI")
    print()
    print("  → set ANTHROPIC_API_KEY and run `agent-friend chat --tools search code memory`")
    print()


def main():
    print()
    print("agent-friend — composable personal AI agent library")
    print("github.com/0-co/company/products/agent-friend")
    print()

    demo_memory()
    demo_code()
    demo_search()
    demo_full()

    print("All tools verified. Install with:")
    print()
    print('  pip install "git+https://github.com/0-co/company.git#subdirectory=products/agent-friend[anthropic]"')
    print()


if __name__ == "__main__":
    main()
