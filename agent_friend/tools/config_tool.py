"""config_tool.py — ConfigTool for agent-friend (stdlib only).

Hierarchical key-value configuration management for agents.
Supports dot-notation namespaces, defaults, type coercion, and
environment variable overrides.

Features:
* config_set / config_get with dot-notation keys (e.g. "db.host")
* config_defaults — set multiple defaults at once
* config_load_env — populate config from os.environ with a prefix
* config_list — list all keys (optionally filtered by prefix)
* config_delete — remove a key
* config_dump — export all config as JSON
* config_require — assert required keys are present
* Multiple named configs per tool instance

Usage::

    tool = ConfigTool()

    tool.config_set("app", "db.host", "localhost")
    tool.config_set("app", "db.port", 5432)
    tool.config_set("app", "debug", True)

    tool.config_get("app", "db.host")      # "localhost"
    tool.config_get("app", "db.port", as_type="int")   # 5432
"""

import json
import os
from typing import Any, Dict, List, Optional

from .base import BaseTool


def _coerce(value: Any, as_type: Optional[str]) -> Any:
    if as_type is None:
        return value
    if as_type == "int":
        return int(value)
    if as_type == "float":
        return float(value)
    if as_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if as_type == "str":
        return str(value)
    if as_type == "json":
        if isinstance(value, str):
            return json.loads(value)
        return value
    raise ValueError(f"Unknown type '{as_type}'. Valid: int, float, bool, str, json.")


class _Config:
    """A single named configuration store."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def list_keys(self, prefix: str = "") -> List[str]:
        if prefix:
            return sorted(k for k in self._data if k.startswith(prefix))
        return sorted(self._data.keys())

    def dump(self) -> Dict[str, Any]:
        return dict(self._data)


class ConfigTool(BaseTool):
    """Hierarchical key-value configuration for agent workflows.

    Supports dot-notation keys, type coercion, env-var loading, and
    required-key validation.  Multiple named configs per instance.

    Parameters
    ----------
    max_configs:
        Maximum named config stores (default 20).
    max_keys:
        Maximum keys per config (default 1_000).
    """

    def __init__(self, max_configs: int = 20, max_keys: int = 1_000) -> None:
        self.max_configs = max_configs
        self.max_keys = max_keys
        self._configs: Dict[str, _Config] = {}

    def _get_or_create(self, name: str) -> _Config:
        if name not in self._configs:
            if len(self._configs) >= self.max_configs:
                raise RuntimeError(f"Max configs ({self.max_configs}) reached.")
            self._configs[name] = _Config()
        return self._configs[name]

    # ── public API ────────────────────────────────────────────────────

    def config_set(self, name: str, key: str, value: Any) -> str:
        """Set a configuration key.

        *key* may use dot-notation (e.g. ``"db.host"``).

        Returns ``{ok: true, key: "...", value: ...}``
        """
        try:
            cfg = self._get_or_create(name)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        if key not in cfg._data and len(cfg._data) >= self.max_keys:
            return json.dumps({"error": f"Max keys ({self.max_keys}) reached."})

        cfg.set(key, value)
        return json.dumps({"ok": True, "key": key, "value": value})

    def config_get(
        self,
        name: str,
        key: str,
        default: Any = None,
        as_type: Optional[str] = None,
    ) -> str:
        """Get a configuration value.

        If the key does not exist, *default* is returned.
        *as_type* coerces the value: ``int``, ``float``, ``bool``, ``str``, ``json``.

        Returns ``{key: "...", value: ..., found: bool}``
        """
        cfg = self._configs.get(name)
        found = cfg is not None and key in cfg._data
        raw = cfg.get(key, default) if cfg else default

        if as_type and raw is not None:
            try:
                raw = _coerce(raw, as_type)
            except (ValueError, json.JSONDecodeError) as exc:
                return json.dumps({"error": f"Type coercion failed: {exc}"})

        return json.dumps({"key": key, "value": raw, "found": found})

    def config_defaults(self, name: str, defaults: Dict[str, Any]) -> str:
        """Set multiple default values (only sets if key not already present).

        Returns ``{set: N}`` — the number of keys that were actually set.
        """
        try:
            cfg = self._get_or_create(name)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        count = 0
        for k, v in defaults.items():
            if k not in cfg._data:
                if len(cfg._data) >= self.max_keys:
                    break
                cfg.set(k, v)
                count += 1
        return json.dumps({"set": count, "name": name})

    def config_load_env(
        self,
        name: str,
        prefix: str = "",
        strip_prefix: bool = True,
        lowercase: bool = True,
    ) -> str:
        """Populate config from environment variables.

        Only variables starting with *prefix* are imported.  If
        *strip_prefix* is true the prefix is removed from the key name.
        If *lowercase* is true keys are lowercased (and ``__`` → ``.``
        for dot-notation).

        Returns ``{loaded: N, keys: [...]}``
        """
        try:
            cfg = self._get_or_create(name)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        loaded = []
        for env_key, env_val in os.environ.items():
            if prefix and not env_key.startswith(prefix):
                continue
            key = env_key
            if strip_prefix and prefix:
                key = key[len(prefix):]
            if lowercase:
                key = key.lower().replace("__", ".")
            if key not in cfg._data and len(cfg._data) >= self.max_keys:
                continue
            cfg.set(key, env_val)
            loaded.append(key)
        return json.dumps({"loaded": len(loaded), "keys": sorted(loaded)})

    def config_list(self, name: str, prefix: str = "") -> str:
        """List all keys in a config, optionally filtered by *prefix*.

        Returns a JSON array of keys (sorted).
        """
        cfg = self._configs.get(name)
        if cfg is None:
            return json.dumps([])
        return json.dumps(cfg.list_keys(prefix))

    def config_delete(self, name: str, key: str) -> str:
        """Delete a specific key.

        Returns ``{deleted: true/false, key: "..."}``
        """
        cfg = self._configs.get(name)
        if cfg is None:
            return json.dumps({"deleted": False, "key": key})
        removed = cfg.delete(key)
        return json.dumps({"deleted": removed, "key": key})

    def config_dump(self, name: str) -> str:
        """Export all config as a JSON object."""
        cfg = self._configs.get(name)
        if cfg is None:
            return json.dumps({})
        return json.dumps(cfg.dump())

    def config_require(self, name: str, keys: List[str]) -> str:
        """Assert that required keys are all present and non-None.

        Returns ``{ok: true}`` if all present, else ``{ok: false, missing: [...]}``
        """
        cfg = self._configs.get(name)
        missing = []
        for k in keys:
            if cfg is None or cfg.get(k) is None:
                missing.append(k)
        if missing:
            return json.dumps({"ok": False, "missing": missing})
        return json.dumps({"ok": True, "missing": []})

    def config_drop(self, name: str) -> str:
        """Delete a named config store."""
        if name not in self._configs:
            return json.dumps({"error": f"No config named '{name}'"})
        del self._configs[name]
        return json.dumps({"dropped": True, "name": name})

    def config_list_stores(self) -> str:
        """List all named config stores with their key counts."""
        result = [
            {"name": n, "keys": len(c._data)}
            for n, c in self._configs.items()
        ]
        return json.dumps(result)

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "config"

    @property
    def description(self) -> str:
        return (
            "Hierarchical key-value configuration management. Dot-notation keys, "
            "type coercion (int/float/bool/json), env-var loading, defaults, "
            "and required-key validation. Multiple named configs. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "config_set",
                "description": "Set a config key (dot-notation OK). Returns {ok, key, value}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "key": {"type": "string", "description": "Key (dot-notation OK: 'db.host')"},
                        "value": {"description": "Value (any JSON type)"},
                    },
                    "required": ["name", "key", "value"],
                },
            },
            {
                "name": "config_get",
                "description": "Get a config value. as_type coerces: int/float/bool/str/json.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "key": {"type": "string"},
                        "default": {"description": "Default if key missing"},
                        "as_type": {"type": "string", "description": "Coerce to int/float/bool/str/json"},
                    },
                    "required": ["name", "key"],
                },
            },
            {
                "name": "config_defaults",
                "description": "Set multiple defaults — only sets keys not already present.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "defaults": {"type": "object", "description": "Key-value pairs"},
                    },
                    "required": ["name", "defaults"],
                },
            },
            {
                "name": "config_load_env",
                "description": "Load environment variables into config. prefix filters by prefix.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "prefix": {"type": "string", "description": "Only load vars with this prefix"},
                        "strip_prefix": {"type": "boolean"},
                        "lowercase": {"type": "boolean"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "config_list",
                "description": "List all keys, optionally filtered by prefix.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "prefix": {"type": "string"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "config_delete",
                "description": "Delete a specific config key.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "key": {"type": "string"},
                    },
                    "required": ["name", "key"],
                },
            },
            {
                "name": "config_dump",
                "description": "Export all config as a JSON object.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "config_require",
                "description": "Assert required keys exist. Returns {ok: bool, missing: [...]}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "keys": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["name", "keys"],
                },
            },
            {
                "name": "config_drop",
                "description": "Delete an entire named config store.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "config_list_stores",
                "description": "List all named config stores with key counts.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "config_set":
            return self.config_set(**arguments)
        if tool_name == "config_get":
            return self.config_get(**arguments)
        if tool_name == "config_defaults":
            return self.config_defaults(**arguments)
        if tool_name == "config_load_env":
            return self.config_load_env(**arguments)
        if tool_name == "config_list":
            return self.config_list(**arguments)
        if tool_name == "config_delete":
            return self.config_delete(**arguments)
        if tool_name == "config_dump":
            return self.config_dump(**arguments)
        if tool_name == "config_require":
            return self.config_require(**arguments)
        if tool_name == "config_drop":
            return self.config_drop(**arguments)
        if tool_name == "config_list_stores":
            return self.config_list_stores()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
