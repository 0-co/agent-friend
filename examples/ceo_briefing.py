#!/usr/bin/env python3
"""CEO Daily Briefing — real dogfooding of agent-friend.

Demonstrates Friend + @tool-decorated functions from ceo_toolkit
hitting live APIs (Twitch, GitHub, Dev.to) via vault-* wrappers,
with a LOCAL LLM (Ollama) orchestrating the tool calls autonomously.

Run (from the agent-friend directory):
    PYTHONPATH=. python3 examples/ceo_briefing.py

Requires:
    - Ollama running at localhost:11434
    - qwen2.5:3b pulled (or specify --model)
    - openai pip package installed
    - sudo -u vault access for tool calls
"""

import json
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, ".")

from agent_friend import Friend
from examples.ceo_toolkit import (
    devto_article_status,
    github_repo_stats,
    twitch_followers,
    twitch_stream_status,
)

# ---------------------------------------------------------------------------
# Tool call visibility callback
# ---------------------------------------------------------------------------


def on_tool_call(name: str, arguments: Dict[str, Any], result: Optional[str]) -> None:
    """Print tool calls and results as they happen."""
    if result is None:
        args_str = json.dumps(arguments) if arguments else "(no args)"
        print(f"  -> calling {name}({args_str})")
    else:
        preview = result[:200] + "..." if len(result) > 200 else result
        print(f"  <- {name}: {preview}")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the daily briefing bot for 0-co, an AI-run company streaming on Twitch.
Call ALL provided tools to gather live metrics, then produce a concise briefing.
Always call every tool before writing the briefing."""

USER_PROMPT = """\
Run the daily briefing. Call these tools:
1. twitch_stream_status — is the stream live?
2. twitch_followers — how many followers?
3. github_repo_stats — stars, forks, issues for 0-co/agent-friend
4. devto_article_status — article 3362409 views and reactions

After gathering all data, write a brief status report."""


def main() -> None:
    model = "qwen2.5:3b"
    for i, arg in enumerate(sys.argv):
        if arg == "--model" and i + 1 < len(sys.argv):
            model = sys.argv[i + 1]

    print("=" * 60)
    print("CEO DAILY BRIEFING — agent-friend dogfood demo")
    print("=" * 60)
    print()
    print(f"Model:    {model} (Ollama, local)")
    print(f"Tools:    twitch_stream_status, twitch_followers,")
    print(f"          github_repo_stats, devto_article_status")
    print()

    friend = Friend(
        model=model,
        provider="ollama",
        seed=SYSTEM_PROMPT,
        tools=[
            twitch_stream_status,
            twitch_followers,
            github_repo_stats,
            devto_article_status,
        ],
        on_tool_call=on_tool_call,
    )

    print("--- Tool Calls ---")
    print()

    t0 = time.time()
    try:
        response = friend.chat(USER_PROMPT)
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"\nERROR after {elapsed:.1f}s: {exc}")
        print(f"Type: {type(exc).__name__}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.time() - t0

    print("--- Briefing ---")
    print()
    print(response.text if response.text else "(No text response)")
    print()

    print("--- Stats ---")
    print(f"  Model:         {response.model}")
    print(f"  Tool calls:    {len(response.tool_calls)}")
    for tc in response.tool_calls:
        print(f"                 - {tc['name']}")
    print(f"  Input tokens:  {response.input_tokens:,}")
    print(f"  Output tokens: {response.output_tokens:,}")
    print(f"  Cost:          $0.00 (local)")
    print(f"  Wall time:     {elapsed:.1f}s")
    print()

    if len(response.tool_calls) == 0:
        print("WARNING: Model made zero tool calls.")
        print("The 3B model may struggle with tool calling.")
        print("Try: --model qwen2.5:7b")


if __name__ == "__main__":
    main()
