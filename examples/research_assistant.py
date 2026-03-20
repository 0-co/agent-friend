#!/usr/bin/env python3
"""
Research assistant — multi-step deep dive on any topic.

Uses agent-friend with search + fetch + memory + file tools to:
  1. Search for a topic
  2. Fetch and read the most relevant sources
  3. Generate a structured research summary
  4. Save the summary to a markdown file
  5. Remember that this research was done

Setup:
    pip install "git+https://github.com/0-co/agent-friend.git"
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai

Usage:
    python3 research_assistant.py "LLM agent memory systems"
    python3 research_assistant.py "Python async patterns" --output report.md
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    parser = argparse.ArgumentParser(description="Research a topic and save a summary")
    parser.add_argument("topic", nargs="?", default="AI agent memory systems",
                        help="Topic to research")
    parser.add_argument("--output", default=None, help="Output file (default: auto-named)")
    parser.add_argument("--depth", choices=["quick", "thorough"], default="thorough",
                        help="Research depth")
    args = parser.parse_args()

    from agent_friend import Friend

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("No API key found. Set OPENROUTER_API_KEY (free at openrouter.ai).")
        sys.exit(1)

    output_file = args.output or f"research_{args.topic[:30].replace(' ', '_')}.md"

    from agent_friend.tools.file import FileTool
    friend = Friend(
        seed=(
            "You are a thorough research assistant. "
            "When given a topic: search for it, fetch the top 2-3 sources, "
            "then produce a structured markdown summary with: "
            "## Overview, ## Key Findings (3-5 bullets), ## Sources. "
            "Be specific and cite actual URLs."
        ),
        tools=["search", "fetch", "memory", FileTool(base_dir=".")],
        api_key=api_key,
        budget_usd=0.15 if args.depth == "thorough" else 0.05,
    )

    print(f"\nResearching: {args.topic}")
    print(f"Depth: {args.depth} | Output: {output_file}")
    print("-" * 50)

    # Step 1: Research
    queries = (
        [f"'{args.topic}'", f"'{args.topic}' best practices 2026"]
        if args.depth == "thorough"
        else [f"'{args.topic}'"]
    )

    query_str = " and ".join(queries)
    r1 = friend.chat(
        f"Search for {query_str}. Fetch the 2 most relevant results. "
        f"Give me a comprehensive markdown research summary."
    )
    print(r1.text)

    # Step 2: Save to file
    r2 = friend.chat(
        f"Write the research summary to a file called '{output_file}'. "
        f"Include a header '# Research: {args.topic}' and today's date."
    )
    print(f"\n{r2.text}")

    # Step 3: Remember it was done
    friend.chat(
        f"Remember that I researched '{args.topic}'. "
        f"Key: 'research_{args.topic[:20]}'. Value: summary saved to {output_file}."
    )

    total_cost = sum(r.cost_usd for r in [r1, r2])
    print(f"\n[${total_cost:.4f} | Output: {output_file}]")


if __name__ == "__main__":
    main()
