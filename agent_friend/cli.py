"""cli.py — CLI entry point for agent-friend."""

import sys
import os
import argparse

# ANSI colors (disabled if not a TTY)
_TTY = sys.stderr.isatty()
CYAN = "\033[36m" if _TTY else ""
GREEN = "\033[32m" if _TTY else ""
YELLOW = "\033[33m" if _TTY else ""
GRAY = "\033[90m" if _TTY else ""
BOLD = "\033[1m" if _TTY else ""
RESET = "\033[0m" if _TTY else ""


def _tool_callback(name: str, args: dict, result) -> None:
    if result is None:
        args_short = str(args)[:80]
        print(f"{CYAN}→ [{name}]{RESET} {GRAY}{args_short}{RESET}", file=sys.stderr, flush=True)
    else:
        result_short = str(result)[:100].replace("\n", " ")
        print(f"{GREEN}← {result_short}{RESET}", file=sys.stderr, flush=True)


def _auto_model(api_key: str | None, requested: str) -> str:
    """Pick a sensible default model based on available API key."""
    if api_key is None:
        return requested
    if api_key.startswith("sk-ant-"):
        return "claude-haiku-4-5-20251001"
    if api_key.startswith("sk-or-"):
        return "google/gemini-2.0-flash-exp:free"
    return requested  # OpenAI or unknown


def _get_api_key() -> str | None:
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def main() -> None:
    """Main entry point for the agent-friend CLI."""
    parser = argparse.ArgumentParser(
        prog="agent-friend",
        description=(
            "agent-friend — a personal AI agent with memory, search, and code execution.\n\n"
            "Quick start (free, no credit card):\n"
            "  export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai\n"
            "  agent-friend -i                       # interactive\n"
            '  agent-friend "search for AI news"     # one-shot'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "message",
        nargs="?",
        help="Send a single message and exit",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start an interactive multi-turn chat session",
    )
    parser.add_argument(
        "--seed",
        default="You are a helpful personal AI assistant.",
        help="System prompt (default: helpful assistant)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model to use. Auto-detected from API key if not set.\n"
            "  Anthropic key → claude-haiku-4-5-20251001\n"
            "  OpenRouter key → google/gemini-2.0-flash-exp:free (free!)\n"
            "  OpenAI key → gpt-4o-mini"
        ),
    )
    parser.add_argument(
        "--tools",
        default="",
        help="Comma-separated tools: search,code,memory,browser,email (default: none)",
    )
    parser.add_argument(
        "--config",
        help="Path to a YAML config file",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Spending limit in USD (free models cost $0)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    if args.no_color:
        global CYAN, GREEN, YELLOW, GRAY, BOLD, RESET
        CYAN = GREEN = YELLOW = GRAY = BOLD = RESET = ""

    if not args.message and not args.interactive:
        parser.print_help()
        sys.exit(0)

    api_key = _get_api_key()
    if not api_key:
        print("No API key found. Set one of:", file=sys.stderr)
        print("  export OPENROUTER_API_KEY=sk-or-...  (free at openrouter.ai)", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  export OPENAI_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    model = args.model or _auto_model(api_key, "claude-haiku-4-5-20251001")
    tools_list = [t.strip() for t in args.tools.split(",") if t.strip()]

    if args.interactive:
        _run_interactive(args, model, api_key, tools_list)
    else:
        _run_single(args, model, api_key, tools_list)


def _build_friend(args, model: str, api_key: str, tools_list: list):
    from .friend import Friend

    if getattr(args, "config", None):
        return Friend.from_yaml(args.config)

    return Friend(
        seed=args.seed,
        model=model,
        tools=tools_list,
        budget_usd=getattr(args, "budget", None),
        on_tool_call=_tool_callback,
    )


def _run_single(args, model: str, api_key: str, tools_list: list) -> None:
    friend = _build_friend(args, model, api_key, tools_list)
    try:
        response = friend.chat(args.message)
        print(response.text)
        print(
            f"{GRAY}[tokens: {response.input_tokens}+{response.output_tokens}, cost: ${response.cost_usd:.4f}]{RESET}",
            file=sys.stderr,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def _run_interactive(args, model: str, api_key: str, tools_list: list) -> None:
    friend = _build_friend(args, model, api_key, tools_list)

    print(f"{BOLD}agent-friend{RESET}", file=sys.stderr)
    print(f"  model: {GRAY}{model}{RESET}", file=sys.stderr)
    if tools_list:
        print(f"  tools: {GRAY}{', '.join(tools_list)}{RESET}", file=sys.stderr)
    print(f"  {GRAY}Type messages. 'reset' to clear history. Ctrl-C to exit.{RESET}\n", file=sys.stderr)

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
            if user_input.lower() in ("exit", "quit", "bye", "q"):
                break
            if user_input.lower() == "reset":
                friend.reset()
                print(f"{GRAY}[Conversation reset. Memory persists.]{RESET}\n", file=sys.stderr)
                continue

            try:
                response = friend.chat(user_input)
                turn += 1
                session_cost += response.cost_usd
                print(f"\n{BOLD}Friend:{RESET} {response.text}\n")
                print(
                    f"{GRAY}[Turn {turn} | ${response.cost_usd:.4f} | total: ${session_cost:.4f}]{RESET}\n",
                    file=sys.stderr,
                )
            except Exception as error:
                print(f"Error: {error}", file=sys.stderr)

    except KeyboardInterrupt:
        pass

    print(f"\n{GRAY}Session ended: {turn} turns, ${session_cost:.4f} total{RESET}", file=sys.stderr)
