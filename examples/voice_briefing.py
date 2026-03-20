#!/usr/bin/env python3
"""
Voice briefing — daily news summary spoken aloud.

Uses agent-friend with search + voice tools to:
  1. Search for news on a topic
  2. Summarize the top results
  3. Speak the summary (system TTS or neural HTTP TTS)

Setup:
    pip install "git+https://github.com/0-co/agent-friend.git"

    # Free inference (no credit card):
    export OPENROUTER_API_KEY=sk-or-...  # from openrouter.ai

    # Or with Anthropic/OpenAI:
    export ANTHROPIC_API_KEY=sk-ant-...
    export OPENAI_API_KEY=sk-...

    # Optional: neural TTS server (high-quality voices)
    # export AGENT_FRIEND_TTS_URL=http://your-tts-server:8081

Usage:
    python3 voice_briefing.py
    python3 voice_briefing.py --topic "Python packaging 2026"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser(description="Daily news briefing spoken aloud")
    parser.add_argument("--topic", default="AI agents 2026", help="Topic to search for")
    parser.add_argument("--voice", default=None, help="TTS voice name (optional)")
    parser.add_argument("--no-voice", action="store_true", help="Skip TTS, print only")
    args = parser.parse_args()

    from agent_friend import Friend
    from agent_friend.tools.voice import VoiceTool

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("No API key found. Set OPENROUTER_API_KEY (free at openrouter.ai).")
        sys.exit(1)

    tts_url = os.environ.get("AGENT_FRIEND_TTS_URL")

    tools = ["search", "fetch"]
    if not args.no_voice:
        tools.append(VoiceTool(tts_url=tts_url, default_voice=args.voice or "en-US-AriaNeural"))

    friend = Friend(
        seed=(
            "You are a concise news briefing assistant. "
            "Search for the latest news, fetch key articles, and give a clear 3-sentence summary. "
            "If a voice tool is available, speak the summary aloud after printing it."
        ),
        tools=tools,
        api_key=api_key,
        budget_usd=0.05,
    )

    print(f"\nSearching for: {args.topic}")
    print("-" * 40)

    response = friend.chat(
        f"Search for '{args.topic}', fetch the most relevant article, "
        f"and give me a 3-sentence summary. "
        + ("Then speak the summary aloud using the voice tool." if not args.no_voice else "")
    )
    print(response.text)
    print(f"\n[${response.cost_usd:.4f} | {response.input_tokens + response.output_tokens} tokens]")


if __name__ == "__main__":
    main()
