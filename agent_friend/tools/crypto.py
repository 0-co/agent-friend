"""crypto.py — CryptoTool for agent-friend (stdlib only).

Agents often need to generate secure tokens, verify webhook signatures (HMAC),
hash data, and encode/decode base64.  All operations are available in Python's
standard library — no dependencies required.

Usage::

    tool = CryptoTool()
    tool.generate_token()                                # "a3f9..."  (32-byte hex)
    tool.hash_data("hello", algorithm="sha256")          # "2cf24d..."
    tool.hmac_sign("payload", "secret")                  # "abc123..."
    tool.hmac_verify("payload", "secret", "abc123...")   # True
    tool.uuid4()                                         # "550e8400-..."
    tool.base64_encode("hello")                          # "aGVsbG8="
    tool.base64_decode("aGVsbG8=")                       # "hello"
"""

import base64
import hashlib
import hmac
import json
import secrets
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseTool


class CryptoTool(BaseTool):
    """Cryptographic utilities for agents: tokens, hashing, HMAC, UUID, base64.

    All operations use Python's stdlib — zero dependencies.

    Supported hash algorithms: md5, sha1, sha224, sha256, sha384, sha512.
    Default: sha256.
    """

    @property
    def name(self) -> str:
        return "crypto"

    @property
    def description(self) -> str:
        return (
            "Cryptographic utilities: secure random tokens, SHA hashing, "
            "HMAC signing and verification, UUID generation, base64 encoding."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "generate_token",
                "description": (
                    "Generate a cryptographically secure random token. "
                    "Returns a hex string. length is the number of random bytes "
                    "(output will be 2× as long in hex). Default 32 bytes → 64-char hex."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "length": {
                            "type": "integer",
                            "description": "Number of random bytes (default 32).",
                        }
                    },
                },
            },
            {
                "name": "hash_data",
                "description": (
                    "Hash a string using the specified algorithm. "
                    "Returns the hex digest. "
                    "Supported: md5, sha1, sha224, sha256, sha384, sha512. "
                    "Default: sha256."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The string to hash.",
                        },
                        "algorithm": {
                            "type": "string",
                            "description": "Hash algorithm (default sha256).",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "hmac_sign",
                "description": (
                    "Sign data with a secret key using HMAC. "
                    "Returns the hex digest. "
                    "Commonly used to sign webhook payloads. "
                    "Supported algorithms: sha1, sha224, sha256, sha384, sha512."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The string to sign.",
                        },
                        "secret": {
                            "type": "string",
                            "description": "The secret key.",
                        },
                        "algorithm": {
                            "type": "string",
                            "description": "HMAC algorithm (default sha256).",
                        },
                    },
                    "required": ["data", "secret"],
                },
            },
            {
                "name": "hmac_verify",
                "description": (
                    "Verify an HMAC signature against data and a secret key. "
                    "Returns true if valid, false otherwise. "
                    "Uses constant-time comparison to prevent timing attacks."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The original data that was signed.",
                        },
                        "secret": {
                            "type": "string",
                            "description": "The secret key used to sign.",
                        },
                        "signature": {
                            "type": "string",
                            "description": "The hex signature to verify.",
                        },
                        "algorithm": {
                            "type": "string",
                            "description": "HMAC algorithm (default sha256).",
                        },
                    },
                    "required": ["data", "secret", "signature"],
                },
            },
            {
                "name": "uuid4",
                "description": "Generate a random UUID (version 4). Returns a string like '550e8400-e29b-41d4-a716-446655440000'.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "base64_encode",
                "description": "Base64-encode a string. Returns the encoded string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The string to encode.",
                        },
                        "url_safe": {
                            "type": "boolean",
                            "description": "Use URL-safe base64 (- and _ instead of + and /). Default false.",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "base64_decode",
                "description": "Decode a base64-encoded string. Returns the original string.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "string",
                            "description": "The base64 string to decode.",
                        },
                        "url_safe": {
                            "type": "boolean",
                            "description": "Use URL-safe base64 decoding. Default false.",
                        },
                    },
                    "required": ["data"],
                },
            },
            {
                "name": "random_bytes",
                "description": "Generate random bytes and return as hex. Useful for nonces, salts, or IVs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "length": {
                            "type": "integer",
                            "description": "Number of random bytes (default 16).",
                        }
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _SUPPORTED_HASH = {"md5", "sha1", "sha224", "sha256", "sha384", "sha512"}
    _SUPPORTED_HMAC = {"sha1", "sha224", "sha256", "sha384", "sha512"}

    def _check_hash_alg(self, algorithm: str, supported: set) -> str:
        alg = algorithm.lower()
        if alg not in supported:
            raise ValueError(
                f"Unsupported algorithm '{alg}'. Choose from: {sorted(supported)}"
            )
        return alg

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate_token(self, length: int = 32) -> str:
        """Return a secure random hex token of *length* bytes."""
        if length < 1:
            raise ValueError("length must be >= 1")
        return secrets.token_hex(length)

    def hash_data(self, data: str, algorithm: str = "sha256") -> str:
        """Hash *data* with *algorithm* and return the hex digest."""
        alg = self._check_hash_alg(algorithm, self._SUPPORTED_HASH)
        h = hashlib.new(alg)
        h.update(data.encode("utf-8"))
        return h.hexdigest()

    def hmac_sign(self, data: str, secret: str, algorithm: str = "sha256") -> str:
        """Sign *data* with *secret* and return the HMAC hex digest."""
        alg = self._check_hash_alg(algorithm, self._SUPPORTED_HMAC)
        mac = hmac.new(
            secret.encode("utf-8"), data.encode("utf-8"), digestmod=alg
        )
        return mac.hexdigest()

    def hmac_verify(
        self,
        data: str,
        secret: str,
        signature: str,
        algorithm: str = "sha256",
    ) -> bool:
        """Verify *signature* against *data* + *secret*. Constant-time."""
        expected = self.hmac_sign(data, secret, algorithm)
        return hmac.compare_digest(expected, signature)

    def uuid4(self) -> str:
        """Return a new random UUID4 string."""
        return str(uuid.uuid4())

    def base64_encode(self, data: str, url_safe: bool = False) -> str:
        """Base64-encode *data*."""
        raw = data.encode("utf-8")
        if url_safe:
            return base64.urlsafe_b64encode(raw).decode("ascii")
        return base64.b64encode(raw).decode("ascii")

    def base64_decode(self, data: str, url_safe: bool = False) -> str:
        """Decode base64 *data* and return the UTF-8 string."""
        try:
            if url_safe:
                return base64.urlsafe_b64decode(data.encode("ascii")).decode("utf-8")
            return base64.b64decode(data.encode("ascii")).decode("utf-8")
        except Exception as exc:
            raise ValueError(f"Invalid base64 data: {exc}") from exc

    def random_bytes(self, length: int = 16) -> str:
        """Return *length* random bytes as a hex string."""
        if length < 1:
            raise ValueError("length must be >= 1")
        return secrets.token_hex(length)

    # ------------------------------------------------------------------
    # BaseTool execute
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "generate_token":
                length = int(arguments.get("length", 32))
                token = self.generate_token(length)
                return json.dumps({"token": token, "length_bytes": length, "length_hex": len(token)})

            elif tool_name == "hash_data":
                data = arguments["data"]
                algorithm = arguments.get("algorithm", "sha256")
                digest = self.hash_data(data, algorithm)
                return json.dumps({"digest": digest, "algorithm": algorithm.lower()})

            elif tool_name == "hmac_sign":
                data = arguments["data"]
                secret = arguments["secret"]
                algorithm = arguments.get("algorithm", "sha256")
                signature = self.hmac_sign(data, secret, algorithm)
                return json.dumps({"signature": signature, "algorithm": algorithm.lower()})

            elif tool_name == "hmac_verify":
                data = arguments["data"]
                secret = arguments["secret"]
                signature = arguments["signature"]
                algorithm = arguments.get("algorithm", "sha256")
                valid = self.hmac_verify(data, secret, signature, algorithm)
                return json.dumps({"valid": valid, "algorithm": algorithm.lower()})

            elif tool_name == "uuid4":
                return json.dumps({"uuid": self.uuid4()})

            elif tool_name == "base64_encode":
                data = arguments["data"]
                url_safe = bool(arguments.get("url_safe", False))
                encoded = self.base64_encode(data, url_safe)
                return json.dumps({"encoded": encoded, "url_safe": url_safe})

            elif tool_name == "base64_decode":
                data = arguments["data"]
                url_safe = bool(arguments.get("url_safe", False))
                decoded = self.base64_decode(data, url_safe)
                return json.dumps({"decoded": decoded, "url_safe": url_safe})

            elif tool_name == "random_bytes":
                length = int(arguments.get("length", 16))
                result = self.random_bytes(length)
                return json.dumps({"hex": result, "length_bytes": length})

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except (KeyError, ValueError) as exc:
            return json.dumps({"error": str(exc)})
