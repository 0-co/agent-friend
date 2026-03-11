"""config.py — Configuration loading for agent-friend."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class FriendConfig:
    """All configuration options for a Friend instance.

    Parameters
    ----------
    seed:                  System prompt / persona for the agent.
    model:                 LLM model identifier.
    api_key:               API key (Anthropic or OpenAI). Falls back to env vars.
    provider:              "anthropic" or "openai" (auto-detected from model if not set).
    tools:                 List of tool names ("memory", "code", "search", "browser")
                           or BaseTool instances.
    memory_path:           Path for SQLite memory database.
    budget_usd:            Optional spending limit in USD.
    max_context_messages:  Maximum messages to keep in conversation history.
    """

    seed: str = "You are a helpful personal AI assistant."
    model: str = "claude-haiku-4-5-20251001"
    api_key: Optional[str] = None
    provider: Optional[str] = None
    tools: List[Any] = field(default_factory=list)
    memory_path: str = "~/.agent_friend/memory.db"
    budget_usd: Optional[float] = None
    max_context_messages: int = 20

    def resolve_provider(self) -> str:
        """Determine provider from explicit config, model name, or api_key prefix."""
        if self.provider:
            return self.provider.lower()
        model_lower = self.model.lower()
        # OpenRouter models use slash notation (e.g. "google/gemini-2.0-flash-exp:free")
        if "/" in model_lower or ":free" in model_lower or ":nitro" in model_lower:
            return "openrouter"
        if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
            return "openai"
        # Infer from explicit api_key prefix (e.g. Friend(api_key="sk-or-...") without setting model)
        if self.api_key and self.api_key.startswith("sk-or-"):
            return "openrouter"
        if self.api_key and self.api_key.startswith("sk-") and not self.api_key.startswith("sk-ant-"):
            return "openai"
        return "anthropic"

    def resolve_api_key(self) -> Optional[str]:
        """Return explicit api_key or look up the right environment variable."""
        if self.api_key:
            return self.api_key
        provider = self.resolve_provider()
        if provider == "openai":
            return os.environ.get("OPENAI_API_KEY")
        if provider == "openrouter":
            return os.environ.get("OPENROUTER_API_KEY")
        return os.environ.get("ANTHROPIC_API_KEY")


def load_from_dict(data: Dict[str, Any]) -> FriendConfig:
    """Build a FriendConfig from a plain dictionary.

    Unknown keys are silently ignored.
    """
    known = {
        "seed",
        "model",
        "api_key",
        "provider",
        "tools",
        "memory_path",
        "budget_usd",
        "max_context_messages",
    }
    filtered = {key: value for key, value in data.items() if key in known}
    return FriendConfig(**filtered)


def load_from_yaml(path: Union[str, Path]) -> FriendConfig:
    """Load a FriendConfig from a YAML file.

    Requires PyYAML if available, falls back to a minimal YAML parser
    for simple key: value files (no nested structures needed for config).
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    content = path.read_text(encoding="utf-8")

    try:
        import yaml
        data = yaml.safe_load(content)
    except ImportError:
        data = _parse_simple_yaml(content)

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping, got: {type(data).__name__}")

    return load_from_dict(data)


def _parse_simple_yaml(content: str) -> Dict[str, Any]:
    """Minimal YAML parser for simple flat key: value config files.

    Handles:
    - key: value
    - Lists (- item per line)
    - Numbers and booleans
    - Quoted strings

    Does not handle nested mappings (not needed for FriendConfig).
    """
    result: Dict[str, Any] = {}
    current_list_key: Optional[str] = None
    current_list: List[Any] = []

    for line in content.splitlines():
        stripped = line.strip()

        # Skip comments and blank lines
        if not stripped or stripped.startswith("#"):
            if current_list_key:
                result[current_list_key] = current_list
                current_list_key = None
                current_list = []
            continue

        # List item
        if stripped.startswith("- "):
            value = stripped[2:].strip()
            current_list.append(_coerce_value(value))
            continue

        # Flush pending list if we hit a non-list line
        if current_list_key:
            result[current_list_key] = current_list
            current_list_key = None
            current_list = []

        # Key: value pair
        if ":" in stripped:
            key, _, raw_value = stripped.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()

            if not raw_value:
                # Might be the start of a list
                current_list_key = key
                current_list = []
            else:
                result[key] = _coerce_value(raw_value)

    # Flush any trailing list
    if current_list_key and current_list:
        result[current_list_key] = current_list

    return result


def _coerce_value(raw: str) -> Any:
    """Convert a raw YAML scalar string to a Python type."""
    # Remove surrounding quotes
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]

    lower = raw.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    if lower in ("null", "~", "none"):
        return None

    # Try numeric
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    return raw
