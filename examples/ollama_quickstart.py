#!/usr/bin/env python3
"""Ollama tool calling in 5 lines.

Prerequisites:
    1. pip install git+https://github.com/0-co/agent-friend.git
    2. ollama pull qwen2.5:3b
    3. Ollama running at localhost:11434

Usage:
    python ollama_quickstart.py
"""

from agent_friend import Friend, tool, Toolkit


# --- Define tools with @tool decorator ---
# Type hints become JSON Schema. Docstrings become descriptions.
# No manual dict schemas. No boilerplate.

@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: City name (e.g. Tokyo, London, New York)
        units: Temperature units — celsius or fahrenheit
    """
    # Stub data — replace with real API call
    temps = {"Tokyo": 22, "London": 14, "New York": 18, "Sydney": 28}
    return {"city": city, "temp": temps.get(city, 20), "units": units, "condition": "clear"}


@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression.

    Args:
        expression: Math expression to evaluate (e.g. '2 + 2', '100 * 0.15')
    """
    # Safe eval for demo — use a proper parser in production
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        return float(eval(expression))
    return 0.0


@tool
def search_facts(topic: str) -> str:
    """Look up facts about a topic.

    Args:
        topic: What to search for
    """
    facts = {
        "python": "Python 3.13 was released October 2024. Key features: free-threaded mode, JIT compiler.",
        "tokyo": "Tokyo metro population: ~14 million. Capital of Japan since 1868.",
        "ollama": "Ollama runs open-source LLMs locally. Supports tool calling since v0.4.",
    }
    return facts.get(topic.lower(), f"No facts found for '{topic}'.")


def main():
    # --- The 5-line version ---
    print("=" * 60)
    print("agent-friend + Ollama: tool calling in 5 lines")
    print("=" * 60)

    # Live tool calling (requires: pip install openai, ollama running)
    print("\n--- Live tool call via Ollama ---")
    try:
        friend = Friend(model="qwen2.5:3b", tools=[get_weather, calculate, search_facts])
        response = friend.chat("What's the weather in Tokyo?")
        print(f"Response: {response}\n")
    except ImportError:
        print("Skipping live demo (install openai: pip install openai)")
        print("The schema generation below works without it.\n")

    # Show what the model sees (the auto-generated schemas)
    print("--- What Ollama receives (auto-generated) ---")
    kit = Toolkit(tools=[get_weather, calculate, search_facts])
    report = kit.token_report()
    print(f"3 tools = ~{report['estimates']['openai']} tokens")
    print(f"Cheapest format: {report['least_expensive']}")
    print(f"Most expensive: {report['most_expensive']}")

    # Cross-format export — same tools, any provider
    print("\n--- Same @tool, every format ---")
    for fmt in ["openai", "anthropic", "google", "mcp", "json_schema"]:
        schema = getattr(get_weather, f"to_{fmt}")()
        # Each format returns a list; grab the first item's name
        item = schema[0] if isinstance(schema, list) else schema
        name = item.get("name", item.get("title", item.get("function", {}).get("name", "?")))
        print(f"  .to_{fmt}() -> {name}")

    print("\n" + "=" * 60)
    print("Done. Same tools work with OpenAI, Anthropic, Gemini, MCP.")
    print("No vendor lock-in. No boilerplate. No API costs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
