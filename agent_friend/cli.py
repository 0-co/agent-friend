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


def _auto_model(api_key, requested: str) -> str:
    """Pick a sensible default model based on available API key."""
    if api_key is None:
        return requested
    if api_key.startswith("sk-ant-"):
        return "claude-haiku-4-5-20251001"
    if api_key.startswith("sk-or-"):
        return "google/gemini-2.0-flash-exp:free"
    return requested  # OpenAI or unknown


def _get_api_key():
    return (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )


def main() -> None:
    """Main entry point for the agent-friend CLI."""
    # Route subcommands before argparse (which uses a flat positional arg)
    if len(sys.argv) > 1 and sys.argv[1] == "audit":
        _run_audit_command(sys.argv[2:])
        return

    if len(sys.argv) > 1 and sys.argv[1] == "optimize":
        _run_optimize_command(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(
        prog="agent-friend",
        description=(
            "agent-friend — universal AI tool adapter. Write once, export everywhere.\n\n"
            "Quick start:\n"
            "  agent-friend --demo                   # see @tool exports (no API key)\n"
            "  agent-friend --version                # show version\n"
            "  agent-friend -i                       # interactive chat (needs API key)\n"
            "  agent-friend audit <file.json>        # token cost report for tool defs\n"
            "  agent-friend optimize <file.json>     # suggest token-saving rewrites"
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo of @tool exports (no API key needed)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__import__('agent_friend').__version__}",
    )

    args = parser.parse_args()

    if args.no_color:
        global CYAN, GREEN, YELLOW, GRAY, BOLD, RESET
        CYAN = GREEN = YELLOW = GRAY = BOLD = RESET = ""

    if args.demo:
        _run_demo()
        return

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


def _run_audit_command(argv: list) -> None:
    """Handle `agent-friend audit <file.json>` subcommand."""
    audit_parser = argparse.ArgumentParser(
        prog="agent-friend audit",
        description="Analyze tool definitions and report token cost across all 5 formats.",
    )
    audit_parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help='Path to a JSON file with tool definitions, or "-" for stdin (default: stdin)',
    )
    audit_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    audit_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON (machine-readable)",
    )
    audit_parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Fail (exit 2) if total tokens exceed this value",
    )
    audit_args = audit_parser.parse_args(argv)

    from .audit import run_audit

    use_color = not audit_args.no_color
    exit_code = run_audit(
        audit_args.file,
        use_color=use_color,
        json_output=audit_args.json_output,
        threshold=audit_args.threshold,
    )
    sys.exit(exit_code)


def _run_optimize_command(argv: list) -> None:
    """Handle `agent-friend optimize <file.json>` subcommand."""
    optimize_parser = argparse.ArgumentParser(
        prog="agent-friend optimize",
        description="Suggest token-saving rewrites for tool definitions.",
    )
    optimize_parser.add_argument(
        "file",
        nargs="?",
        default="-",
        help='Path to a JSON file with tool definitions, or "-" for stdin (default: stdin)',
    )
    optimize_parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    optimize_parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output suggestions as JSON (machine-readable)",
    )
    optimize_args = optimize_parser.parse_args(argv)

    from .optimize import run_optimize

    use_color = not optimize_args.no_color
    exit_code = run_optimize(
        optimize_args.file,
        use_color=use_color,
        json_output=optimize_args.json_output,
    )
    sys.exit(exit_code)


def _run_demo() -> None:
    """Show @tool decorator in action — no API key needed."""
    import json
    from .tools.function_tool import tool

    print(f"\n{BOLD}agent-friend @tool demo{RESET}")
    print(f"{GRAY}No API key required. This shows the universal export feature.{RESET}\n")

    # Define the example function
    print(f"{YELLOW}Step 1:{RESET} Define a function with @tool\n")
    print(f"""{CYAN}from agent_friend import tool

@tool
def get_weather(city: str, units: str = "celsius") -> str:
    \"\"\"Get current weather for a city.

    Args:
        city: City name (e.g. London, Tokyo)
        units: Temperature units (celsius or fahrenheit)
    \"\"\"
    return f"Weather in {{city}}: 22°C, partly cloudy"{RESET}
""")

    @tool
    def get_weather(city: str, units: str = "celsius") -> str:
        """Get current weather for a city.

        Args:
            city: City name (e.g. London, Tokyo)
            units: Temperature units (celsius or fahrenheit)
        """
        return f"Weather in {city}: 22°C, partly cloudy"

    # Show exports
    formats = [
        ("OpenAI", "to_openai", "client.chat.completions.create(tools=...)"),
        ("Anthropic", "to_anthropic", "client.messages.create(tools=...)"),
        ("Google", "to_google", "genai.GenerativeModel(tools=...)"),
        ("MCP", "to_mcp", "Model Context Protocol servers"),
        ("JSON Schema", "to_json_schema", "Any framework"),
    ]

    print(f"{YELLOW}Step 2:{RESET} Export to any AI framework\n")

    for name, method, usage in formats:
        result = getattr(get_weather, method)()
        compact = json.dumps(result[0] if isinstance(result, list) else result, indent=2)
        lines = compact.split("\n")
        preview = "\n".join(lines[:8])
        if len(lines) > 8:
            preview += f"\n  ... ({len(lines) - 8} more lines)"
        print(f"{GREEN}get_weather.{method}(){RESET}  {GRAY}# {usage}{RESET}")
        print(f"{GRAY}{preview}{RESET}\n")

    # Call the function
    print(f"{YELLOW}Step 3:{RESET} Call it like a normal function\n")
    result = get_weather("Tokyo")
    print(f"  get_weather(\"Tokyo\") → {GREEN}{result}{RESET}\n")

    # Batch export
    print(f"{YELLOW}Bonus:{RESET} Batch export with Toolkit\n")
    print(f"""{CYAN}from agent_friend import Toolkit

tk = Toolkit(tools=[get_weather, stock_price, send_email])
tk.to_openai()     # all tools in OpenAI format
tk.to_anthropic()  # all tools in Anthropic format{RESET}
""")

    # Token estimation
    print(f"{YELLOW}Bonus:{RESET} Context budget — how many tokens do your tools cost?\n")
    from .toolkit import Toolkit as _Tk
    _demo_kit = _Tk([get_weather])
    _report = _demo_kit.token_report()
    print(f"  {CYAN}tk.token_report(){RESET}")
    for fmt, tokens in _report["estimates"].items():
        bar = "\u2588" * max(1, tokens // 5)
        print(f"  {fmt:13s} ~{tokens:>3d} tokens {GRAY}{bar}{RESET}")
    print(f"  {GRAY}most expensive: {_report['most_expensive']}, least: {_report['least_expensive']}{RESET}\n")

    print(f"{BOLD}Try it:{RESET}")
    print(f"  pip install \"git+https://github.com/0-co/agent-friend.git[all]\"")
    print(f"  {GRAY}# Then use @tool in your own code{RESET}\n")


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
