#!/usr/bin/env python3
"""
agent-friend live demo — AI agent that does real work.

Shows: search + code + memory working together via LLM.
Uses OpenRouter free tier (no credit card required).

Setup:
    pip install "git+https://github.com/0-co/company.git#subdirectory=products/agent-friend[all]"
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai

Usage:
    python3 demo_live.py
    python3 demo_live.py --task "search for python packaging tools"
    python3 demo_live.py --model "meta-llama/llama-3.3-70b-instruct:free"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


TASKS = [
    "Search for the top 3 trending open source AI projects on GitHub this week and summarize what makes each one interesting.",
    "Write a Python function that checks if a number is prime, run it with the first 20 numbers, and show me the results.",
    "Search for recent news about AI agents in 2026 and write a one-paragraph summary of the most important development.",
    "Remember that my preferred programming language is Python and my timezone is UTC. Then tell me what you now know about me.",
]


def run_demo(task: str, model: str, api_key: str) -> None:
    from agent_friend import Friend

    print()
    print("─" * 60)
    print("agent-friend demo")
    print(f"model: {model}")
    print(f"task: {task}")
    print("─" * 60)
    print()

    friend = Friend(
        seed=(
            "You are a helpful assistant with access to web search, "
            "code execution, and persistent memory. Be concise. "
            "Show your work — use tools when they help."
        ),
        tools=["search", "code", "memory"],
        model=model,
        api_key=api_key,
        budget_usd=0.10,  # $0.10 max (free models cost $0)
    )

    print("User:", task)
    print()
    print("Agent thinking...", flush=True)
    print()

    response = friend.chat(task)

    print("Agent:", response.text)
    print()

    if response.tool_calls:
        print(f"[Tools used: {', '.join(tc['name'] for tc in response.tool_calls)}]")

    print(f"[Tokens: {response.input_tokens} in, {response.output_tokens} out | Cost: ${response.cost_usd:.4f}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="agent-friend live demo")
    parser.add_argument("--task", default=TASKS[0], help="Task to give the agent")
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-exp:free",
        help="Model to use (default: Gemini 2.0 Flash free)",
    )
    parser.add_argument("--all-tasks", action="store_true", help="Run all demo tasks")
    args = parser.parse_args()

    # Get API key
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )

    if not api_key:
        print("No API key found. Set one of:")
        print("  export OPENROUTER_API_KEY=sk-or-...  (free at openrouter.ai)")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    # Auto-detect provider from key prefix
    model = args.model
    if api_key.startswith("sk-ant-") and "gemini" in model:
        model = "claude-haiku-4-5-20251001"
        print(f"[Anthropic key detected, switching model to {model}]")
    elif api_key.startswith("sk-") and not api_key.startswith("sk-or-") and not api_key.startswith("sk-ant-"):
        if "gemini" in model or "/" in model:
            model = "gpt-4o-mini"
            print(f"[OpenAI key detected, switching model to {model}]")

    if args.all_tasks:
        for task in TASKS:
            run_demo(task, model, api_key)
    else:
        run_demo(args.task, model, api_key)


if __name__ == "__main__":
    main()
