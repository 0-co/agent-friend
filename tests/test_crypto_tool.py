"""Tests for CryptoTool."""

import json
import re

import pytest

from agent_friend.tools.crypto import CryptoTool


@pytest.fixture()
def tool():
    return CryptoTool()


# ---------------------------------------------------------------------------
# BaseTool contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_name(self, tool):
        assert tool.name == "crypto"

    def test_description(self, tool):
        assert len(tool.description) > 10

    def test_definitions_list(self, tool):
        defs = tool.definitions()
        assert isinstance(defs, list)
        assert len(defs) >= 7

    def test_definitions_have_required_keys(self, tool):
        for d in tool.definitions():
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ---------------------------------------------------------------------------
# generate_token
# ---------------------------------------------------------------------------

class TestGenerateToken:
    def test_default_length(self, tool):
        token = tool.generate_token()
        assert len(token) == 64  # 32 bytes → 64 hex chars

    def test_custom_length(self, tool):
        token = tool.generate_token(16)
        assert len(token) == 32

    def test_is_hex(self, tool):
        token = tool.generate_token()
        assert re.fullmatch(r"[0-9a-f]+", token)

    def test_uniqueness(self, tool):
        tokens = {tool.generate_token() for _ in range(10)}
        assert len(tokens) == 10

    def test_invalid_length(self, tool):
        with pytest.raises(ValueError):
            tool.generate_token(0)

    def test_execute_generate_token(self, tool):
        result = json.loads(tool.execute("generate_token", {}))
        assert "token" in result
        assert result["length_bytes"] == 32
        assert result["length_hex"] == 64

    def test_execute_generate_token_custom(self, tool):
        result = json.loads(tool.execute("generate_token", {"length": 8}))
        assert len(result["token"]) == 16
        assert result["length_bytes"] == 8


# ---------------------------------------------------------------------------
# hash_data
# ---------------------------------------------------------------------------

class TestHashData:
    def test_sha256_known(self, tool):
        # echo -n "hello" | sha256sum
        assert tool.hash_data("hello") == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_md5_known(self, tool):
        import hashlib
        expected = hashlib.md5(b"hello").hexdigest()
        assert tool.hash_data("hello", "md5") == expected

    def test_sha512(self, tool):
        result = tool.hash_data("hello", "sha512")
        assert len(result) == 128

    def test_sha1(self, tool):
        result = tool.hash_data("hello", "sha1")
        assert len(result) == 40

    def test_sha384(self, tool):
        result = tool.hash_data("hello", "sha384")
        assert len(result) == 96

    def test_sha224(self, tool):
        result = tool.hash_data("hello", "sha224")
        assert len(result) == 56

    def test_invalid_algorithm(self, tool):
        with pytest.raises(ValueError):
            tool.hash_data("hello", "blake2b")

    def test_empty_string(self, tool):
        result = tool.hash_data("")
        assert len(result) == 64  # sha256 of empty string

    def test_execute_hash_data(self, tool):
        result = json.loads(tool.execute("hash_data", {"data": "hello"}))
        assert result["digest"] == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert result["algorithm"] == "sha256"

    def test_execute_hash_data_md5(self, tool):
        result = json.loads(tool.execute("hash_data", {"data": "hello", "algorithm": "md5"}))
        assert result["algorithm"] == "md5"
        assert len(result["digest"]) == 32

    def test_case_insensitive_algorithm(self, tool):
        result1 = tool.hash_data("hello", "SHA256")
        result2 = tool.hash_data("hello", "sha256")
        assert result1 == result2


# ---------------------------------------------------------------------------
# hmac_sign
# ---------------------------------------------------------------------------

class TestHmacSign:
    def test_known_value(self, tool):
        import hmac as _hmac
        import hashlib
        expected = _hmac.new(b"secret", b"payload", digestmod=hashlib.sha256).hexdigest()
        assert tool.hmac_sign("payload", "secret") == expected

    def test_different_secrets_differ(self, tool):
        s1 = tool.hmac_sign("data", "secret1")
        s2 = tool.hmac_sign("data", "secret2")
        assert s1 != s2

    def test_different_data_differ(self, tool):
        s1 = tool.hmac_sign("data1", "secret")
        s2 = tool.hmac_sign("data2", "secret")
        assert s1 != s2

    def test_sha1(self, tool):
        result = tool.hmac_sign("data", "secret", "sha1")
        assert len(result) == 40

    def test_sha512(self, tool):
        result = tool.hmac_sign("data", "secret", "sha512")
        assert len(result) == 128

    def test_invalid_algorithm(self, tool):
        with pytest.raises(ValueError):
            tool.hmac_sign("data", "secret", "md5")

    def test_execute_hmac_sign(self, tool):
        result = json.loads(tool.execute("hmac_sign", {"data": "payload", "secret": "key"}))
        assert "signature" in result
        assert result["algorithm"] == "sha256"
        assert len(result["signature"]) == 64


# ---------------------------------------------------------------------------
# hmac_verify
# ---------------------------------------------------------------------------

class TestHmacVerify:
    def test_valid_signature(self, tool):
        sig = tool.hmac_sign("payload", "secret")
        assert tool.hmac_verify("payload", "secret", sig) is True

    def test_invalid_signature(self, tool):
        assert tool.hmac_verify("payload", "secret", "deadbeef") is False

    def test_wrong_secret(self, tool):
        sig = tool.hmac_sign("payload", "secret")
        assert tool.hmac_verify("payload", "wrong", sig) is False

    def test_tampered_data(self, tool):
        sig = tool.hmac_sign("payload", "secret")
        assert tool.hmac_verify("tampered", "secret", sig) is False

    def test_with_sha1(self, tool):
        sig = tool.hmac_sign("data", "key", "sha1")
        assert tool.hmac_verify("data", "key", sig, "sha1") is True

    def test_execute_hmac_verify_valid(self, tool):
        sig = tool.hmac_sign("payload", "secret")
        result = json.loads(tool.execute("hmac_verify", {
            "data": "payload", "secret": "secret", "signature": sig
        }))
        assert result["valid"] is True

    def test_execute_hmac_verify_invalid(self, tool):
        result = json.loads(tool.execute("hmac_verify", {
            "data": "payload", "secret": "secret", "signature": "bad"
        }))
        assert result["valid"] is False

    def test_roundtrip_all_algorithms(self, tool):
        for alg in ["sha1", "sha224", "sha256", "sha384", "sha512"]:
            sig = tool.hmac_sign("test", "k", alg)
            assert tool.hmac_verify("test", "k", sig, alg) is True


# ---------------------------------------------------------------------------
# uuid4
# ---------------------------------------------------------------------------

class TestUUID4:
    UUID_RE = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    )

    def test_format(self, tool):
        result = tool.uuid4()
        assert self.UUID_RE.match(result), f"Bad UUID: {result}"

    def test_uniqueness(self, tool):
        uuids = {tool.uuid4() for _ in range(20)}
        assert len(uuids) == 20

    def test_execute_uuid4(self, tool):
        result = json.loads(tool.execute("uuid4", {}))
        assert "uuid" in result
        assert self.UUID_RE.match(result["uuid"])


# ---------------------------------------------------------------------------
# base64_encode / base64_decode
# ---------------------------------------------------------------------------

class TestBase64:
    def test_encode_known(self, tool):
        assert tool.base64_encode("hello") == "aGVsbG8="

    def test_decode_known(self, tool):
        assert tool.base64_decode("aGVsbG8=") == "hello"

    def test_roundtrip(self, tool):
        original = "The quick brown fox"
        assert tool.base64_decode(tool.base64_encode(original)) == original

    def test_url_safe_encode(self, tool):
        # Characters that differ: + → -, / → _
        data = "\xff\xfe"  # bytes that produce + or / in standard base64
        standard = tool.base64_encode(data)
        url_safe = tool.base64_encode(data, url_safe=True)
        # At minimum, no + or /
        assert "+" not in url_safe
        assert "/" not in url_safe

    def test_url_safe_roundtrip(self, tool):
        original = "url safe test \xff\xfe"
        encoded = tool.base64_encode(original, url_safe=True)
        decoded = tool.base64_decode(encoded, url_safe=True)
        assert decoded == original

    def test_empty_string(self, tool):
        assert tool.base64_encode("") == ""
        assert tool.base64_decode("") == ""

    def test_invalid_base64(self, tool):
        with pytest.raises(ValueError):
            tool.base64_decode("!!!notbase64!!!")

    def test_execute_encode(self, tool):
        result = json.loads(tool.execute("base64_encode", {"data": "hello"}))
        assert result["encoded"] == "aGVsbG8="
        assert result["url_safe"] is False

    def test_execute_decode(self, tool):
        result = json.loads(tool.execute("base64_decode", {"data": "aGVsbG8="}))
        assert result["decoded"] == "hello"

    def test_execute_encode_url_safe(self, tool):
        result = json.loads(tool.execute("base64_encode", {"data": "hello", "url_safe": True}))
        assert result["url_safe"] is True


# ---------------------------------------------------------------------------
# random_bytes
# ---------------------------------------------------------------------------

class TestRandomBytes:
    def test_default_length(self, tool):
        result = tool.random_bytes()
        assert len(result) == 32  # 16 bytes → 32 hex

    def test_custom_length(self, tool):
        result = tool.random_bytes(8)
        assert len(result) == 16

    def test_is_hex(self, tool):
        result = tool.random_bytes()
        assert re.fullmatch(r"[0-9a-f]+", result)

    def test_uniqueness(self, tool):
        results = {tool.random_bytes() for _ in range(10)}
        assert len(results) == 10

    def test_invalid_length(self, tool):
        with pytest.raises(ValueError):
            tool.random_bytes(0)

    def test_execute_random_bytes(self, tool):
        result = json.loads(tool.execute("random_bytes", {}))
        assert "hex" in result
        assert result["length_bytes"] == 16
        assert len(result["hex"]) == 32

    def test_execute_random_bytes_custom(self, tool):
        result = json.loads(tool.execute("random_bytes", {"length": 4}))
        assert len(result["hex"]) == 8


# ---------------------------------------------------------------------------
# execute — error handling
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    def test_unknown_tool(self, tool):
        result = json.loads(tool.execute("nonexistent", {}))
        assert "error" in result

    def test_missing_required_arg(self, tool):
        result = json.loads(tool.execute("hash_data", {}))
        assert "error" in result

    def test_invalid_algorithm_via_execute(self, tool):
        result = json.loads(tool.execute("hash_data", {"data": "x", "algorithm": "blowfish"}))
        assert "error" in result
