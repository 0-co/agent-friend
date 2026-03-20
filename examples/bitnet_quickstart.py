#!/usr/bin/env python3
"""BitNet 1-bit LLM quickstart.

Prerequisites:
    1. pip install git+https://github.com/0-co/agent-friend.git
    2. BitNet llama-server running at localhost:8080
       Setup: https://github.com/microsoft/BitNet

Usage:
    python bitnet_quickstart.py
"""

from agent_friend import Friend

# BitNet: 1-bit LLM inference on CPU
# Requires: BitNet llama-server running at localhost:8080
# Setup: https://github.com/microsoft/BitNet
friend = Friend(model="bitnet-b1.58-2B-4T")  # auto-detects BitNet provider
response = friend.chat("What is the capital of France?")
print(response.text)
print(f"Tokens: {response.input_tokens} in, {response.output_tokens} out")
print(f"Cost: ${response.cost_usd:.4f}")  # Always $0 — runs locally
