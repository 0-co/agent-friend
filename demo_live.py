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
    python3 demo_live.py --interactive
    python3 demo_live.py --interactive --tools search,memory,code
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI colors
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
GRAY = "\033[90m"
BOLD = "\033[1m"
RESET = "\033[0m"

TASKS = [
    "Search for the top 3 trending open source AI projects on GitHub this week and summarize what makes each one interesting.",
    "Write a Python function that checks if a number is prime, run it with the first 20 numbers, and show me the results.",
    "Search for recent news about AI agents in 2026 and write a one-paragraph summary of the most important development.",
    "Remember that my preferred programming language is Python and my timezone is UTC. Then tell me what you now know about me.",
]


def tool_callback(name: str, args: dict, result) -> None:
    """Show tool calls in real time."""
    if result is None:
        # Before call
        args_short = str(args)[:80]
        print(f"{CYAN}→ [{name}]{RESET} {GRAY}{args_short}{RESET}", flush=True)
    else:
        # After call
        result_short = str(result)[:100].replace("\n", " ")
        print(f"{GREEN}← {result_short}{RESET}", flush=True)


def run_demo(task: str, model: str, api_key: str, tools_list: list[str]) -> None:
    from agent_friend import Friend

    print()
    print(f"{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}agent-friend demo{RESET}")
    print(f"  model: {GRAY}{model}{RESET}")
    print(f"  tools: {GRAY}{', '.join(tools_list)}{RESET}")
    print(f"  task:  {task[:70]}")
    print(f"{BOLD}{'─' * 60}{RESET}")
    print()

    friend = Friend(
        seed=(
            "You are a helpful assistant with access to web search, "
            "code execution, and persistent memory. Be concise. "
            "Show your work — use tools when they help."
        ),
        tools=tools_list,
        model=model,
        api_key=api_key,
        budget_usd=0.10,
        on_tool_call=tool_callback,
    )

    print(f"{BOLD}User:{RESET} {task}")
    print()

    response = friend.chat(task)

    print()
    print(f"{BOLD}Agent:{RESET} {response.text}")
    print()
    print(f"{GRAY}[Tokens: {response.input_tokens} in, {response.output_tokens} out | Cost: ${response.cost_usd:.4f}]{RESET}")
    print()


def run_interactive(model: str, api_key: str, tools_list: list[str]) -> None:
    from agent_friend import Friend

    print()
    print(f"{BOLD}{'─' * 60}{RESET}")
    print(f"{BOLD}agent-friend — interactive mode{RESET}")
    print(f"  model: {GRAY}{model}{RESET}")
    print(f"  tools: {GRAY}{', '.join(tools_list)}{RESET}")
    print(f"  {GRAY}Type your message. 'quit' or Ctrl-C to exit. 'reset' to clear history.{RESET}")
    print(f"{BOLD}{'─' * 60}{RESET}")
    print()

    friend = Friend(
        seed=(
            "You are a helpful assistant with access to web search, "
            "code execution, and persistent memory. Be concise and direct."
        ),
        tools=tools_list,
        model=model,
        api_key=api_key,
        budget_usd=1.00,
        on_tool_call=tool_callback,
    )

    session_cost = 0.0
    turn = 0

    try:
        while True:
            try:
                user_input = input(f"{BOLD}You:{RESET} ").strip()
            except EOFError:
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if user_input.lower() == "reset":
                friend.reset()
                print(f"{GRAY}[Conversation reset. Memory persists.]{RESET}\n")
                continue

            print()

            response = friend.chat(user_input)
            turn += 1
            session_cost += response.cost_usd

            print()
            print(f"{BOLD}Agent:{RESET} {response.text}")
            print()
            print(
                f"{GRAY}[Turn {turn} | ${response.cost_usd:.4f} | session total: ${session_cost:.4f}]{RESET}"
            )
            print()

    except KeyboardInterrupt:
        pass

    print(f"\n{GRAY}Session: {turn} turns, ${session_cost:.4f} total cost{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="agent-friend live demo")
    parser.add_argument("--task", default=TASKS[0], help="Task to give the agent")
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-exp:free",
        help="Model to use (default: Gemini 2.0 Flash free)",
    )
    parser.add_argument("--all-tasks", action="store_true", help="Run all demo tasks")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument(
        "--tools",
        default="search,code,memory",
        help="Comma-separated tools to enable (default: search,code,memory)",
    )
    args = parser.parse_args()

    tools_list = [t.strip() for t in args.tools.split(",") if t.strip()]

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
    if api_key.startswith("sk-ant-") and ("gemini" in model or "/" in model):
        model = "claude-haiku-4-5-20251001"
        print(f"{GRAY}[Anthropic key detected, switching model to {model}]{RESET}")
    elif api_key.startswith("sk-") and not api_key.startswith("sk-or-") and not api_key.startswith("sk-ant-"):
        if "gemini" in model or "/" in model:
            model = "gpt-4o-mini"
            print(f"{GRAY}[OpenAI key detected, switching model to {model}]{RESET}")

    if args.interactive:
        run_interactive(model, api_key, tools_list)
    elif args.all_tasks:
        for task in TASKS:
            run_demo(task, model, api_key, tools_list)
    else:
        run_demo(args.task, model, api_key, tools_list)


if __name__ == "__main__":
    main()
