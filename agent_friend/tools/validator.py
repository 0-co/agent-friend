"""validator.py — ValidatorTool for agent-friend (stdlib only).

Agents processing user input need to validate it before acting on it.
This tool covers the common cases: emails, URLs, IPs, UUIDs, phone numbers,
JSON schema, numeric ranges, and string patterns — all without deps.

Usage::

    tool = ValidatorTool()
    tool.validate_email("user@example.com")     # {"valid": True}
    tool.validate_url("https://github.com")      # {"valid": True, "scheme": "https", ...}
    tool.validate_ip("192.168.1.1")             # {"valid": True, "version": 4}
    tool.validate_uuid("550e8400-...")           # {"valid": True, "version": 4}
    tool.validate_json('{"key": "value"}')       # {"valid": True, "parsed": {...}}
    tool.validate_range(42, min_val=0, max_val=100)  # {"valid": True}
    tool.validate_pattern("abc123", r"^[a-z0-9]+$")  # {"valid": True}
"""

import ipaddress
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .base import BaseTool


class ValidatorTool(BaseTool):
    """Input validation for agents: email, URL, IP, UUID, JSON, range, regex.

    All operations are stdlib-only — zero dependencies.
    """

    @property
    def name(self) -> str:
        return "validator"

    @property
    def description(self) -> str:
        return (
            "Validate user inputs and data: email addresses, URLs, IP addresses, "
            "UUIDs, JSON strings, numeric ranges, and regex patterns."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "validate_email",
                "description": (
                    "Validate an email address. Checks format only (RFC 5322 simplified). "
                    "Returns {valid: bool, reason: str}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "description": "Email address to validate."}
                    },
                    "required": ["email"],
                },
            },
            {
                "name": "validate_url",
                "description": (
                    "Validate a URL. Checks that it has a valid scheme and host. "
                    "Returns {valid: bool, scheme: str, host: str, path: str, reason: str}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to validate."},
                        "allowed_schemes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Allowed schemes (default: ['http', 'https']).",
                        },
                    },
                    "required": ["url"],
                },
            },
            {
                "name": "validate_ip",
                "description": (
                    "Validate an IP address (IPv4 or IPv6). "
                    "Returns {valid: bool, version: int, is_private: bool, is_loopback: bool}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "ip": {"type": "string", "description": "IP address to validate."}
                    },
                    "required": ["ip"],
                },
            },
            {
                "name": "validate_uuid",
                "description": (
                    "Validate a UUID string. "
                    "Returns {valid: bool, version: int, variant: str}. "
                    "Accepts with or without hyphens."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "UUID string to validate."}
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "validate_json",
                "description": (
                    "Validate a JSON string. "
                    "Returns {valid: bool, parsed: any, type: str, reason: str}. "
                    "Optionally check that specific keys exist."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "JSON string to validate."},
                        "required_keys": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Keys that must exist at the top level (optional).",
                        },
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "validate_range",
                "description": (
                    "Validate that a number is within a range. "
                    "Returns {valid: bool, value: number, reason: str}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "description": "Number to validate."},
                        "min_val": {"type": "number", "description": "Minimum value (inclusive)."},
                        "max_val": {"type": "number", "description": "Maximum value (inclusive)."},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "validate_pattern",
                "description": (
                    "Validate a string against a regex pattern. "
                    "Returns {valid: bool, groups: list, reason: str}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "String to test."},
                        "pattern": {"type": "string", "description": "Regex pattern."},
                        "flags": {
                            "type": "string",
                            "description": "Regex flags: 'i' (ignore case), 'm' (multiline), 's' (dotall). Default empty.",
                        },
                    },
                    "required": ["value", "pattern"],
                },
            },
            {
                "name": "validate_length",
                "description": (
                    "Validate string or list length. "
                    "Returns {valid: bool, length: int, reason: str}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"description": "String or list to measure."},
                        "min_length": {"type": "integer", "description": "Minimum length (inclusive)."},
                        "max_length": {"type": "integer", "description": "Maximum length (inclusive)."},
                    },
                    "required": ["value"],
                },
            },
            {
                "name": "validate_type",
                "description": (
                    "Check that a value is of the expected type. "
                    "Returns {valid: bool, actual_type: str, expected_type: str}. "
                    "Supported types: string, int, float, number, bool, list, dict, null."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "value": {"description": "Value to check."},
                        "expected_type": {
                            "type": "string",
                            "description": "Expected type name: string, int, float, number, bool, list, dict, null.",
                        },
                    },
                    "required": ["value", "expected_type"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    # Simplified RFC 5322 — catches the obvious bad cases
    _EMAIL_RE = re.compile(
        r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
        r"\.[a-zA-Z]{2,}$"
    )

    def validate_email(self, email: str) -> Dict[str, Any]:
        email = email.strip()
        if not email:
            return {"valid": False, "reason": "Empty string"}
        if len(email) > 320:
            return {"valid": False, "reason": "Email too long (max 320 chars)"}
        if email.count("@") != 1:
            return {"valid": False, "reason": f"Expected exactly one '@', found {email.count('@')}"}
        local, domain = email.rsplit("@", 1)
        if len(local) > 64:
            return {"valid": False, "reason": "Local part too long (max 64 chars)"}
        if ".." in email:
            return {"valid": False, "reason": "Consecutive dots not allowed"}
        if not self._EMAIL_RE.match(email):
            return {"valid": False, "reason": "Invalid format"}
        return {"valid": True, "local": local, "domain": domain}

    # ------------------------------------------------------------------
    # URL
    # ------------------------------------------------------------------

    def validate_url(
        self, url: str, allowed_schemes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if allowed_schemes is None:
            allowed_schemes = ["http", "https"]
        try:
            parsed = urlparse(url.strip())
        except Exception as exc:
            return {"valid": False, "reason": str(exc)}

        if not parsed.scheme:
            return {"valid": False, "reason": "Missing scheme (e.g. https://)"}
        if parsed.scheme.lower() not in [s.lower() for s in allowed_schemes]:
            return {
                "valid": False,
                "reason": f"Scheme '{parsed.scheme}' not in allowed: {allowed_schemes}",
            }
        if not parsed.netloc:
            return {"valid": False, "reason": "Missing host"}

        return {
            "valid": True,
            "scheme": parsed.scheme,
            "host": parsed.hostname or "",
            "port": parsed.port,
            "path": parsed.path,
            "query": parsed.query,
        }

    # ------------------------------------------------------------------
    # IP
    # ------------------------------------------------------------------

    def validate_ip(self, ip: str) -> Dict[str, Any]:
        try:
            addr = ipaddress.ip_address(ip.strip())
            return {
                "valid": True,
                "version": addr.version,
                "is_private": addr.is_private,
                "is_loopback": addr.is_loopback,
                "is_global": addr.is_global,
                "compressed": str(addr),
            }
        except ValueError:
            return {"valid": False, "reason": f"'{ip}' is not a valid IP address"}

    # ------------------------------------------------------------------
    # UUID
    # ------------------------------------------------------------------

    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-?[0-9a-f]{4}-?([0-9a-f])[0-9a-f]{3}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    _UUID_VERSION_MAP = {
        "1": "time-based",
        "3": "name-based MD5",
        "4": "random",
        "5": "name-based SHA-1",
    }

    def validate_uuid(self, value: str) -> Dict[str, Any]:
        val = value.strip()
        m = self._UUID_RE.match(val)
        if not m:
            return {"valid": False, "reason": "Not a valid UUID format"}
        version = int(m.group(1))
        return {
            "valid": True,
            "version": version,
            "variant": self._UUID_VERSION_MAP.get(str(version), "unknown"),
        }

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def validate_json(
        self, value: str, required_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            return {"valid": False, "reason": str(exc), "parsed": None}

        result: Dict[str, Any] = {
            "valid": True,
            "type": type(parsed).__name__,
            "parsed": parsed,
        }

        if required_keys and isinstance(parsed, dict):
            missing = [k for k in required_keys if k not in parsed]
            if missing:
                result["valid"] = False
                result["reason"] = f"Missing required keys: {missing}"
        elif required_keys and not isinstance(parsed, dict):
            result["valid"] = False
            result["reason"] = "Cannot check required_keys: parsed value is not an object"

        return result

    # ------------------------------------------------------------------
    # Range
    # ------------------------------------------------------------------

    def validate_range(
        self,
        value: float,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Dict[str, Any]:
        if min_val is not None and value < min_val:
            return {"valid": False, "value": value, "reason": f"{value} < min {min_val}"}
        if max_val is not None and value > max_val:
            return {"valid": False, "value": value, "reason": f"{value} > max {max_val}"}
        return {"valid": True, "value": value}

    # ------------------------------------------------------------------
    # Pattern
    # ------------------------------------------------------------------

    _FLAG_MAP = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}

    def validate_pattern(
        self, value: str, pattern: str, flags: str = ""
    ) -> Dict[str, Any]:
        combined = 0
        for ch in flags.lower():
            if ch in self._FLAG_MAP:
                combined |= self._FLAG_MAP[ch]
            else:
                return {"valid": False, "reason": f"Unknown flag: '{ch}'"}
        try:
            m = re.search(pattern, value, combined)
        except re.error as exc:
            return {"valid": False, "reason": f"Invalid regex: {exc}"}

        if m is None:
            return {"valid": False, "reason": "Pattern did not match", "groups": []}
        return {"valid": True, "match": m.group(0), "groups": list(m.groups())}

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------

    def validate_length(
        self,
        value: Any,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            length = len(value)
        except TypeError:
            return {"valid": False, "reason": f"Cannot measure length of {type(value).__name__}"}

        if min_length is not None and length < min_length:
            return {
                "valid": False,
                "length": length,
                "reason": f"Length {length} < min {min_length}",
            }
        if max_length is not None and length > max_length:
            return {
                "valid": False,
                "length": length,
                "reason": f"Length {length} > max {max_length}",
            }
        return {"valid": True, "length": length}

    # ------------------------------------------------------------------
    # Type
    # ------------------------------------------------------------------

    _TYPE_CHECK: Dict[str, type] = {
        "string": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    def validate_type(self, value: Any, expected_type: str) -> Dict[str, Any]:
        actual = type(value).__name__
        et = expected_type.lower()

        if et == "null":
            valid = value is None
        elif et == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif et in self._TYPE_CHECK:
            cls = self._TYPE_CHECK[et]
            # Booleans are ints in Python — treat them as distinct
            if et == "int":
                valid = isinstance(value, int) and not isinstance(value, bool)
            else:
                valid = isinstance(value, cls)
        else:
            return {
                "valid": False,
                "actual_type": actual,
                "expected_type": expected_type,
                "reason": f"Unknown type '{expected_type}'",
            }

        return {"valid": valid, "actual_type": actual, "expected_type": expected_type}

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "validate_email":
                return json.dumps(self.validate_email(arguments["email"]))

            elif tool_name == "validate_url":
                return json.dumps(
                    self.validate_url(
                        arguments["url"],
                        arguments.get("allowed_schemes"),
                    )
                )

            elif tool_name == "validate_ip":
                return json.dumps(self.validate_ip(arguments["ip"]))

            elif tool_name == "validate_uuid":
                return json.dumps(self.validate_uuid(arguments["value"]))

            elif tool_name == "validate_json":
                return json.dumps(
                    self.validate_json(
                        arguments["value"],
                        arguments.get("required_keys"),
                    )
                )

            elif tool_name == "validate_range":
                return json.dumps(
                    self.validate_range(
                        float(arguments["value"]),
                        float(arguments["min_val"]) if "min_val" in arguments else None,
                        float(arguments["max_val"]) if "max_val" in arguments else None,
                    )
                )

            elif tool_name == "validate_pattern":
                return json.dumps(
                    self.validate_pattern(
                        arguments["value"],
                        arguments["pattern"],
                        arguments.get("flags", ""),
                    )
                )

            elif tool_name == "validate_length":
                return json.dumps(
                    self.validate_length(
                        arguments["value"],
                        int(arguments["min_length"]) if "min_length" in arguments else None,
                        int(arguments["max_length"]) if "max_length" in arguments else None,
                    )
                )

            elif tool_name == "validate_type":
                return json.dumps(
                    self.validate_type(arguments["value"], arguments["expected_type"])
                )

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except (KeyError, ValueError, TypeError) as exc:
            return json.dumps({"error": str(exc)})
