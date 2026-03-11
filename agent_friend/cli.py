"""cli.py — CLI entry point for agent-friend."""

import sys
import os
import argparse


def main() -> None:
    """Main entry point for the agent-friend CLI."""
    parser = argparse.ArgumentParser(
        prog="agent-friend",
        description="agent-friend — a composable personal AI agent library",
    )
    subparsers = parser.add_subparsers(dest="command")

    # agent-friend chat
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--seed", default="You are a helpful personal AI assistant.",
        help="System prompt for the agent"
    )
    chat_parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Model to use (default: claude-haiku-4-5-20251001)"
    )
    chat_parser.add_argument(
        "--tools", nargs="*", default=[],
        help="Tools to enable: search, code, memory, browser"
    )
    chat_parser.add_argument(
        "--config", help="Path to a YAML config file"
    )
    chat_parser.add_argument(
        "--budget", type=float, help="Spending limit in USD"
    )

    # agent-friend run
    run_parser = subparsers.add_parser("run", help="Send a single message and exit")
    run_parser.add_argument("message", help="Message to send to the agent")
    run_parser.add_argument(
        "--seed", default="You are a helpful personal AI assistant.",
        help="System prompt"
    )
    run_parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001", help="Model to use"
    )
    run_parser.add_argument(
        "--tools", nargs="*", default=[], help="Tools to enable"
    )
    run_parser.add_argument(
        "--config", help="Path to a YAML config file"
    )

    args = parser.parse_args()

    if args.command == "run":
        _run_single(args)
    elif args.command == "chat":
        _run_interactive(args)
    else:
        parser.print_help()
        sys.exit(0)


def _build_friend(args):
    """Build a Friend instance from CLI args."""
    from .friend import Friend

    if getattr(args, "config", None):
        return Friend.from_yaml(args.config)

    return Friend(
        seed=args.seed,
        model=args.model,
        tools=getattr(args, "tools", []) or [],
        budget_usd=getattr(args, "budget", None),
    )


def _run_single(args) -> None:
    """Send one message and print the response."""
    friend = _build_friend(args)
    try:
        response = friend.chat(args.message)
        print(response.text)
        print(
            f"\n[tokens: {response.input_tokens}+{response.output_tokens},"
            f" cost: ${response.cost_usd:.4f}]",
            file=sys.stderr,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


def _run_interactive(args) -> None:
    """Run an interactive multi-turn chat loop."""
    friend = _build_friend(args)
    print(f"agent-friend — model: {friend._config.model}", file=sys.stderr)
    print("Type 'exit' or Ctrl-C to quit.\n", file=sys.stderr)

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.", file=sys.stderr)
            break

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye.", file=sys.stderr)
            break

        if not user_input:
            continue

        try:
            response = friend.chat(user_input)
            print(f"Friend: {response.text}\n")
        except Exception as error:
            print(f"Error: {error}", file=sys.stderr)
