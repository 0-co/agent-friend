#!/usr/bin/env python3
"""End-to-end demo: agent-friend @tool decorator with Ollama (local LLM).

Proves the full pipeline:
  1. Define Python functions with @tool
  2. Export to OpenAI tool format via .to_openai()
  3. Send to Ollama (OpenAI-compatible API)
  4. LLM calls the tool
  5. Execute tool, return result
  6. LLM produces final answer

Requirements: ollama running locally with qwen2.5:3b model.
  ollama pull qwen2.5:3b
"""

import json
import urllib.request

from agent_friend import tool, Toolkit

# ---------------------------------------------------------------------------
# Define tools with @tool decorator
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name
        unit: Temperature unit (celsius or fahrenheit)
    """
    # Simulated weather data
    data = {
        "new york": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "cloudy"},
        "tokyo": {"temp": 28, "condition": "humid"},
    }
    info = data.get(city.lower(), {"temp": 20, "condition": "unknown"})
    if unit == "fahrenheit":
        info["temp"] = int(info["temp"] * 9 / 5 + 32)
    return {"city": city, "unit": unit, **info}


@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression safely.

    Args:
        expression: A math expression like '2 + 3 * 4'
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    try:
        result = eval(expression)  # safe: only digits and operators
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Ollama OpenAI-compatible chat
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://127.0.0.1:11434/v1/chat/completions"
MODEL = "qwen2.5:3b"


def ollama_chat(messages, tools=None):
    """Send a chat request to Ollama's OpenAI-compatible endpoint."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


def execute_tool(name, arguments):
    """Execute a tool by name with the given arguments."""
    fn = TOOL_MAP.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    # @tool-decorated functions are callable directly
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return fn(**arguments)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run_demo(user_message):
    print(f"\n{'='*60}")
    print(f"USER: {user_message}")
    print(f"{'='*60}")

    # Export tools to OpenAI format
    kit = Toolkit([get_weather, calculate])
    openai_tools = kit.to_openai()

    print(f"\nTools exported ({len(openai_tools)} tools in OpenAI format)")
    for t in openai_tools:
        print(f"  - {t['function']['name']}: {t['function']['description'][:60]}")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": user_message},
    ]

    # First LLM call
    print("\n[1] Sending to Ollama...")
    response = ollama_chat(messages, tools=openai_tools)
    choice = response["choices"][0]
    msg = choice["message"]

    if msg.get("tool_calls"):
        # LLM wants to call tools
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            fn_args = tc["function"]["arguments"]
            if isinstance(fn_args, str):
                fn_args_parsed = json.loads(fn_args)
            else:
                fn_args_parsed = fn_args
            print(f"\n[2] LLM called tool: {fn_name}({fn_args_parsed})")

            result = execute_tool(fn_name, fn_args_parsed)
            print(f"    Result: {result}")

            # Add assistant message and tool result
            messages.append(msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })

        # Second LLM call with tool results
        print("\n[3] Sending tool results back to Ollama...")
        response2 = ollama_chat(messages)
        final = response2["choices"][0]["message"]["content"]
        print(f"\nASSISTANT: {final}")
    else:
        # No tool call — direct answer
        print(f"\nASSISTANT: {msg['content']}")

    return msg


if __name__ == "__main__":
    print("agent-friend + Ollama End-to-End Demo")
    print(f"Model: {MODEL}")
    print(f"Tools: @tool decorator → .to_openai() → Ollama")

    # Demo 1: Weather tool
    run_demo("What's the weather like in Tokyo?")

    # Demo 2: Calculator tool
    run_demo("What is 42 * 17 + 3.14?")

    # Demo 3: No tool needed
    run_demo("What is the capital of France?")

    print(f"\n{'='*60}")
    print("Demo complete. All tools defined with @tool, exported via .to_openai().")
