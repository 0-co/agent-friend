"""git_commit_agent.py — An AI agent that can read and commit to git repos.

Combines GitTool with CodeTool and FileTool for a developer assistant that can:
- Read the current git status and recent history
- View diffs and understand what changed
- Write or edit files
- Stage changes and commit with a descriptive message

Usage:
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai
    python3 examples/git_commit_agent.py
    python3 examples/git_commit_agent.py --repo /path/to/repo
"""

import argparse
import os
import sys
from agent_friend import Friend, GitTool


def main():
    parser = argparse.ArgumentParser(description="Git commit agent")
    parser.add_argument("--repo", default=".", help="Path to git repo (default: current dir)")
    args = parser.parse_args()

    if not any(
        os.environ.get(k)
        for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    ):
        print("No API key found. Set OPENROUTER_API_KEY (free at openrouter.ai) and re-run.")
        sys.exit(1)

    git = GitTool(repo_dir=args.repo)

    # Quick sanity check — Python API, no LLM needed
    status = git.status()
    if "fatal" in status.lower():
        print(f"Not a git repo: {args.repo}")
        sys.exit(1)

    print(f"Git repo: {os.path.abspath(args.repo)}")
    print(f"Status:\n{status}\n")
    print(f"Recent commits:\n{git.log(n=3)}\n")

    # Build agent with git + code + file tools
    agent = Friend(
        seed="""You are a developer assistant with git access.

You can read the repo status, view diffs, understand the code, and commit changes.

Before committing:
1. Use git_status to see what changed
2. Use git_diff to understand the changes
3. Write a clear, specific commit message that explains WHY the change was made
4. Stage the specific files that belong together
5. Commit with the descriptive message

Never commit all changes blindly — understand what you're committing first.""",
        tools=[git, "code", "file"],
        on_tool_call=lambda name, args, result: (
            print(f"  → [{name}] {args}") if result is None
            else print(f"  ← {str(result)[:100]}")
        ),
    )

    print("Agent ready. Ask me to review commits, understand changes, or commit code.\n")
    print("Examples:")
    print("  'What's the current git status and what changed in the last 3 commits?'")
    print("  'Show me what changed in src/ since the last commit'")
    print("  'Stage all changes to tests/ and commit with a good message'\n")

    # Interactive REPL
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDone.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break

        response = agent.chat(user_input)
        print(f"\nAgent: {response.text}\n")
        if response.cost_usd > 0:
            print(f"[cost: ${response.cost_usd:.4f}]\n")


if __name__ == "__main__":
    main()
