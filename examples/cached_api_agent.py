"""cached_api_agent.py — Cache expensive API calls to avoid redundant requests.

Demonstrates CacheTool: store results with TTL so the agent doesn't
re-fetch the same data on every run.
"""

import json
import os
from agent_friend import Friend, CacheTool, HTTPTool


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY")
        return

    # Build tools
    cache = CacheTool()
    http = HTTPTool()

    agent = Friend(
        seed=(
            "You are a developer assistant. "
            "Before fetching data, always check the cache. "
            "Cache results with a 1-hour TTL to avoid redundant requests."
        ),
        api_key=api_key,
        model="google/gemini-2.0-flash-exp:free",
        tools=[cache, http],
        budget_usd=0.05,
    )

    print("=== Cached API Agent ===\n")

    # First request — will fetch and cache
    r1 = agent.chat(
        "Check if 'pypi_requests_version' is cached. "
        "If not, fetch https://pypi.org/pypi/requests/json and extract the latest version. "
        "Cache the version string under 'pypi_requests_version' for 3600 seconds."
    )
    print("First request:", r1.text)
    print(f"  Cost: ${r1.cost_usd:.4f}")
    print()

    # Second request — should use cache
    r2 = agent.chat(
        "What is the latest version of requests? Use the cache."
    )
    print("Second request (from cache):", r2.text)
    print(f"  Cost: ${r2.cost_usd:.4f}")
    print()

    # Show stats
    stats = json.loads(cache.cache_stats())
    print(f"Cache stats: {stats}")
    print(f"  Entries: {stats['entries']}")
    print(f"  Session hits: {stats['session_hits']}")
    print(f"  Session misses: {stats['session_misses']}")


if __name__ == "__main__":
    main()
