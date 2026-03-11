#!/usr/bin/env python3
"""
agent-friend daily briefing demo.

Agent searches for AI news, checks email, summarizes everything.
Demonstrates: search + email + memory working together via LLM.

Setup:
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai
    export AGENTMAIL_INBOX=0coceo@agentmail.to

    # Or with Anthropic:
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    python3 demo_briefing.py
    python3 demo_briefing.py --model "claude-haiku-4-5-20251001"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from agent_friend import Friend
    from agent_friend.tools.email import EmailTool

    # Detect which API key is available
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("No API key found. Set OPENROUTER_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY.")
        print("Free option: https://openrouter.ai/ (no credit card required)")
        sys.exit(1)

    # Choose model based on key
    if api_key.startswith("sk-or-"):
        model = "google/gemini-2.0-flash-exp:free"
        provider_name = "OpenRouter (free)"
    elif api_key.startswith("sk-ant-"):
        model = "claude-haiku-4-5-20251001"
        provider_name = "Anthropic"
    else:
        model = "gpt-4o-mini"
        provider_name = "OpenAI"

    inbox = os.environ.get("AGENTMAIL_INBOX", "0coceo@agentmail.to")

    print(f"\n{'='*60}")
    print(f"agent-friend daily briefing")
    print(f"model: {model} via {provider_name}")
    print(f"inbox: {inbox}")
    print(f"{'='*60}\n")

    friend = Friend(
        seed=(
            "You are a helpful daily briefing assistant. "
            "Your job is to search for relevant news, check email, "
            "and provide a concise summary. "
            "Be specific. Use real information from your tools. "
            "Keep responses under 200 words each."
        ),
        tools=[
            "search",
            "memory",
            EmailTool(inbox=inbox),
        ],
        model=model,
        api_key=api_key,
        budget_usd=0.10,
    )

    print("Step 1: Checking inbox for new messages...")
    r1 = friend.chat(
        "Check my email inbox. List any new or unread messages with their subjects and senders. "
        "If the inbox is empty, just say 'Inbox is clear.'"
    )
    print(f"\nAgent: {r1.text}\n")

    print("Step 2: Searching for today's AI news...")
    r2 = friend.chat(
        "Search for 'AI agents 2026 news' and summarize the 3 most interesting recent developments."
    )
    print(f"\nAgent: {r2.text}\n")

    print("Step 3: Saving summary to memory...")
    r3 = friend.chat(
        "Remember today's briefing: store a key called 'last_briefing' with a one-sentence "
        "summary of the most important thing from today's email and news check."
    )
    print(f"\nAgent: {r3.text}\n")

    print("Step 4: Final summary...")
    r4 = friend.chat(
        "Give me a single paragraph briefing: what's in the inbox, what's happening in AI today, "
        "and what should I focus on."
    )
    print(f"\nAgent: {r4.text}\n")

    # Show stats
    total_cost = sum(r.cost_usd for r in [r1, r2, r3, r4])
    total_tokens = sum(r.input_tokens + r.output_tokens for r in [r1, r2, r3, r4])
    print(f"{'='*60}")
    print(f"Total: {total_tokens} tokens | Cost: ${total_cost:.4f}")
    if total_cost == 0:
        print("(Free tier — no cost)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
