"""custom_tools.py — Register any Python function as an agent tool with @tool.

The @tool decorator wraps a plain function in a FunctionTool that agent-friend
can use. Type hints become the JSON schema automatically.

Usage:
    export OPENROUTER_API_KEY=sk-or-...  # free at openrouter.ai
    python3 examples/custom_tools.py
"""

from typing import Optional
from agent_friend import Friend, tool


# ── Define custom tools ───────────────────────────────────────────────────────


@tool
def stock_price(ticker: str) -> str:
    """Get current stock price for a ticker symbol."""
    # In production, call a real stock API here
    prices = {"AAPL": "182.50", "GOOG": "175.20", "MSFT": "415.30", "NVDA": "875.00"}
    price = prices.get(ticker.upper())
    if price is None:
        return f"No data for {ticker}."
    return f"{ticker.upper()}: ${price}"


@tool(name="convert_celsius", description="Convert a temperature from Celsius to Fahrenheit")
def celsius_to_fahrenheit(celsius: float) -> str:
    return f"{celsius:.1f}°C = {celsius * 9 / 5 + 32:.1f}°F"


@tool
def team_member(name: str, include_email: Optional[bool] = None) -> str:
    """Look up a team member's role and contact info."""
    team = {
        "alice": {"role": "Engineer", "email": "alice@example.com"},
        "bob": {"role": "Designer", "email": "bob@example.com"},
    }
    member = team.get(name.lower())
    if not member:
        return f"No team member named '{name}'."
    if include_email:
        return f"{name}: {member['role']} ({member['email']})"
    return f"{name}: {member['role']}"


# ── Sanity-check: functions still work normally ───────────────────────────────

assert stock_price("AAPL") == "AAPL: $182.50"
assert celsius_to_fahrenheit(0.0) == "0.0°C = 32.0°F"
assert team_member("alice") == "alice: Engineer"
print("Functions callable normally ✓")


# ── Wire them into an agent ───────────────────────────────────────────────────

import os

if not any(
    os.environ.get(k)
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
):
    print("\nNo API key found — skipping live agent demo.")
    print("Set OPENROUTER_API_KEY (free at openrouter.ai) and re-run.")
else:
    friend = Friend(
        seed="You are a helpful assistant with access to custom tools.",
        tools=["search", stock_price, celsius_to_fahrenheit, team_member],
        on_tool_call=lambda name, args, result: (
            print(f"→ [{name}] {args}") if result is None
            else print(f"← {str(result)[:80]}")
        ),
    )

    response = friend.chat(
        "What's the AAPL and NVDA stock price? Also convert 25°C to Fahrenheit. "
        "And who is Alice on the team?"
    )
    print("\n" + response.text)
    print(f"\n[cost: ${response.cost_usd:.4f}]")
