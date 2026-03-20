"""api_agent.py — Agent that interacts with REST APIs.

Demonstrates HTTPTool: an agent that can GET data from public APIs,
POST updates, and handle authenticated endpoints via default_headers.

Run:
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai
    python examples/api_agent.py
"""

import os
from agent_friend import Friend, HTTPTool

api_key = (
    os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)
if not api_key:
    print("Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY")
    exit(1)

# HTTPTool with auth headers for all requests
# Replace with your actual auth token:
# http = HTTPTool(default_headers={"Authorization": "Bearer YOUR_TOKEN"})

# Public demo — no auth needed
http = HTTPTool()

friend = Friend(
    seed=(
        "You are a helpful assistant with access to HTTP APIs. "
        "When asked to fetch data or make API calls, use the http_request tool. "
        "Return results clearly."
    ),
    tools=["memory", http],
    api_key=api_key,
    budget_usd=0.10,
)

# Example 1: Fetch public API data
print("=== Example 1: Fetch public data ===")
response = friend.chat(
    "GET https://api.github.com/repos/0-co/agent-friend and tell me "
    "the star count, open issues, and primary language."
)
print(response.text)
print()

# Example 2: POST to test endpoint
print("=== Example 2: POST request ===")
response = friend.chat(
    "POST to https://httpbin.org/post with a JSON body containing "
    "{\"tool\": \"agent-friend\", \"version\": \"0.13.0\"}. "
    "Tell me what the server echoed back."
)
print(response.text)
print()

# Example 3: Save result to memory
print("=== Example 3: Fetch + remember ===")
response = friend.chat(
    "GET https://pypi.org/pypi/requests/json to find the latest version of the "
    "'requests' Python package. Save it to memory as 'requests_version'."
)
print(response.text)
