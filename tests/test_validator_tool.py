"""Tests for ValidatorTool."""

import json

import pytest

from agent_friend.tools.validator import ValidatorTool


@pytest.fixture()
def tool():
    return ValidatorTool()


# ---------------------------------------------------------------------------
# BaseTool contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_name(self, tool):
        assert tool.name == "validator"

    def test_description(self, tool):
        assert len(tool.description) > 10

    def test_definitions(self, tool):
        defs = tool.definitions()
        assert isinstance(defs, list)
        assert len(defs) >= 9

    def test_definitions_keys(self, tool):
        for d in tool.definitions():
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ---------------------------------------------------------------------------
# validate_email
# ---------------------------------------------------------------------------

class TestValidateEmail:
    def test_valid_simple(self, tool):
        r = tool.validate_email("user@example.com")
        assert r["valid"] is True
        assert r["local"] == "user"
        assert r["domain"] == "example.com"

    def test_valid_with_dots(self, tool):
        assert tool.validate_email("first.last@sub.domain.org")["valid"] is True

    def test_valid_plus_tag(self, tool):
        assert tool.validate_email("user+tag@example.com")["valid"] is True

    def test_invalid_no_at(self, tool):
        assert tool.validate_email("notanemail")["valid"] is False

    def test_invalid_double_at(self, tool):
        assert tool.validate_email("a@@b.com")["valid"] is False

    def test_invalid_no_domain(self, tool):
        assert tool.validate_email("user@")["valid"] is False

    def test_invalid_no_tld(self, tool):
        assert tool.validate_email("user@localhost")["valid"] is False

    def test_invalid_consecutive_dots(self, tool):
        assert tool.validate_email("user..name@example.com")["valid"] is False

    def test_empty(self, tool):
        assert tool.validate_email("")["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_email", {"email": "a@b.com"}))
        assert r["valid"] is True

    def test_execute_invalid(self, tool):
        r = json.loads(tool.execute("validate_email", {"email": "bad"}))
        assert r["valid"] is False
        assert "reason" in r


# ---------------------------------------------------------------------------
# validate_url
# ---------------------------------------------------------------------------

class TestValidateUrl:
    def test_valid_https(self, tool):
        r = tool.validate_url("https://github.com/0-co/agent-friend")
        assert r["valid"] is True
        assert r["scheme"] == "https"
        assert r["host"] == "github.com"

    def test_valid_http(self, tool):
        assert tool.validate_url("http://example.com")["valid"] is True

    def test_invalid_no_scheme(self, tool):
        r = tool.validate_url("github.com/foo")
        assert r["valid"] is False

    def test_invalid_ftp_by_default(self, tool):
        r = tool.validate_url("ftp://example.com")
        assert r["valid"] is False

    def test_ftp_allowed(self, tool):
        r = tool.validate_url("ftp://files.example.com", allowed_schemes=["ftp"])
        assert r["valid"] is True

    def test_with_query(self, tool):
        r = tool.validate_url("https://example.com/search?q=hello")
        assert r["valid"] is True
        assert r["query"] == "q=hello"

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_url", {"url": "https://example.com"}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# validate_ip
# ---------------------------------------------------------------------------

class TestValidateIp:
    def test_valid_ipv4(self, tool):
        r = tool.validate_ip("192.168.1.1")
        assert r["valid"] is True
        assert r["version"] == 4
        assert r["is_private"] is True

    def test_valid_ipv6(self, tool):
        r = tool.validate_ip("::1")
        assert r["valid"] is True
        assert r["version"] == 6
        assert r["is_loopback"] is True

    def test_loopback(self, tool):
        r = tool.validate_ip("127.0.0.1")
        assert r["is_loopback"] is True

    def test_global(self, tool):
        r = tool.validate_ip("8.8.8.8")
        assert r["is_global"] is True
        assert r["is_private"] is False

    def test_invalid(self, tool):
        r = tool.validate_ip("999.999.999.999")
        assert r["valid"] is False

    def test_invalid_text(self, tool):
        r = tool.validate_ip("not-an-ip")
        assert r["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_ip", {"ip": "8.8.8.8"}))
        assert r["valid"] is True
        assert r["version"] == 4


# ---------------------------------------------------------------------------
# validate_uuid
# ---------------------------------------------------------------------------

class TestValidateUuid:
    VALID_UUID4 = "550e8400-e29b-41d4-a716-446655440000"
    VALID_UUID1 = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

    def test_valid_uuid4(self, tool):
        r = tool.validate_uuid(self.VALID_UUID4)
        assert r["valid"] is True
        assert r["version"] == 4

    def test_valid_uuid1(self, tool):
        r = tool.validate_uuid(self.VALID_UUID1)
        assert r["valid"] is True
        assert r["version"] == 1

    def test_without_hyphens(self, tool):
        no_hyphens = self.VALID_UUID4.replace("-", "")
        r = tool.validate_uuid(no_hyphens)
        assert r["valid"] is True

    def test_invalid_short(self, tool):
        assert tool.validate_uuid("550e8400")["valid"] is False

    def test_invalid_text(self, tool):
        assert tool.validate_uuid("not-a-uuid-at-all-here")["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_uuid", {"value": self.VALID_UUID4}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# validate_json
# ---------------------------------------------------------------------------

class TestValidateJson:
    def test_valid_object(self, tool):
        r = tool.validate_json('{"key": "value"}')
        assert r["valid"] is True
        assert r["type"] == "dict"
        assert r["parsed"] == {"key": "value"}

    def test_valid_array(self, tool):
        r = tool.validate_json('[1, 2, 3]')
        assert r["valid"] is True
        assert r["type"] == "list"

    def test_valid_number(self, tool):
        r = tool.validate_json('42')
        assert r["valid"] is True

    def test_invalid_json(self, tool):
        r = tool.validate_json('{bad json}')
        assert r["valid"] is False
        assert "reason" in r

    def test_required_keys_present(self, tool):
        r = tool.validate_json('{"a": 1, "b": 2}', required_keys=["a", "b"])
        assert r["valid"] is True

    def test_required_keys_missing(self, tool):
        r = tool.validate_json('{"a": 1}', required_keys=["a", "b", "c"])
        assert r["valid"] is False
        assert "b" in r["reason"]

    def test_required_keys_non_dict(self, tool):
        r = tool.validate_json('[1,2,3]', required_keys=["key"])
        assert r["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_json", {"value": '{"x": 1}'}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# validate_range
# ---------------------------------------------------------------------------

class TestValidateRange:
    def test_in_range(self, tool):
        r = tool.validate_range(50, min_val=0, max_val=100)
        assert r["valid"] is True

    def test_at_min(self, tool):
        assert tool.validate_range(0, min_val=0)["valid"] is True

    def test_at_max(self, tool):
        assert tool.validate_range(100, max_val=100)["valid"] is True

    def test_below_min(self, tool):
        r = tool.validate_range(-1, min_val=0)
        assert r["valid"] is False

    def test_above_max(self, tool):
        r = tool.validate_range(101, max_val=100)
        assert r["valid"] is False

    def test_no_bounds(self, tool):
        assert tool.validate_range(999999)["valid"] is True

    def test_float(self, tool):
        assert tool.validate_range(3.14, min_val=0.0, max_val=10.0)["valid"] is True

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_range", {"value": 5, "min_val": 0, "max_val": 10}))
        assert r["valid"] is True

    def test_execute_out_of_range(self, tool):
        r = json.loads(tool.execute("validate_range", {"value": 15, "max_val": 10}))
        assert r["valid"] is False


# ---------------------------------------------------------------------------
# validate_pattern
# ---------------------------------------------------------------------------

class TestValidatePattern:
    def test_matching(self, tool):
        r = tool.validate_pattern("hello123", r"^[a-z]+\d+$")
        assert r["valid"] is True

    def test_not_matching(self, tool):
        r = tool.validate_pattern("hello", r"^\d+$")
        assert r["valid"] is False

    def test_case_insensitive(self, tool):
        r = tool.validate_pattern("HELLO", r"^hello$", flags="i")
        assert r["valid"] is True

    def test_capture_groups(self, tool):
        r = tool.validate_pattern("2026-03-12", r"(\d{4})-(\d{2})-(\d{2})")
        assert r["valid"] is True
        assert r["groups"] == ["2026", "03", "12"]

    def test_invalid_regex(self, tool):
        r = tool.validate_pattern("hello", r"[invalid")
        assert r["valid"] is False
        assert "regex" in r["reason"].lower()

    def test_unknown_flag(self, tool):
        r = tool.validate_pattern("hello", r"hello", flags="z")
        assert r["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_pattern", {"value": "abc", "pattern": r"^[a-z]+$"}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# validate_length
# ---------------------------------------------------------------------------

class TestValidateLength:
    def test_string_in_range(self, tool):
        r = tool.validate_length("hello", min_length=3, max_length=10)
        assert r["valid"] is True
        assert r["length"] == 5

    def test_string_too_short(self, tool):
        r = tool.validate_length("hi", min_length=5)
        assert r["valid"] is False

    def test_string_too_long(self, tool):
        r = tool.validate_length("too long string", max_length=5)
        assert r["valid"] is False

    def test_list_length(self, tool):
        r = tool.validate_length([1, 2, 3], min_length=2, max_length=5)
        assert r["valid"] is True

    def test_empty_string(self, tool):
        r = tool.validate_length("", min_length=1)
        assert r["valid"] is False

    def test_no_bounds(self, tool):
        assert tool.validate_length("anything")["valid"] is True

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_length", {"value": "hello", "min_length": 3}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# validate_type
# ---------------------------------------------------------------------------

class TestValidateType:
    def test_string(self, tool):
        r = tool.validate_type("hello", "string")
        assert r["valid"] is True

    def test_int(self, tool):
        r = tool.validate_type(42, "int")
        assert r["valid"] is True

    def test_bool_is_not_int(self, tool):
        # Python bools are ints but we treat them as distinct
        r = tool.validate_type(True, "int")
        assert r["valid"] is False

    def test_float(self, tool):
        r = tool.validate_type(3.14, "float")
        assert r["valid"] is True

    def test_number_accepts_int(self, tool):
        r = tool.validate_type(42, "number")
        assert r["valid"] is True

    def test_number_accepts_float(self, tool):
        r = tool.validate_type(1.5, "number")
        assert r["valid"] is True

    def test_number_rejects_bool(self, tool):
        r = tool.validate_type(True, "number")
        assert r["valid"] is False

    def test_bool(self, tool):
        assert tool.validate_type(True, "bool")["valid"] is True
        assert tool.validate_type(False, "bool")["valid"] is True

    def test_list(self, tool):
        assert tool.validate_type([1, 2], "list")["valid"] is True

    def test_dict(self, tool):
        assert tool.validate_type({"a": 1}, "dict")["valid"] is True

    def test_null(self, tool):
        assert tool.validate_type(None, "null")["valid"] is True

    def test_wrong_type(self, tool):
        r = tool.validate_type("hello", "int")
        assert r["valid"] is False
        assert r["actual_type"] == "str"

    def test_unknown_type(self, tool):
        r = tool.validate_type("hello", "banana")
        assert r["valid"] is False

    def test_execute(self, tool):
        r = json.loads(tool.execute("validate_type", {"value": "hello", "expected_type": "string"}))
        assert r["valid"] is True


# ---------------------------------------------------------------------------
# execute — error handling
# ---------------------------------------------------------------------------

class TestExecuteErrors:
    def test_unknown_tool(self, tool):
        r = json.loads(tool.execute("nonexistent", {}))
        assert "error" in r

    def test_missing_required_arg(self, tool):
        r = json.loads(tool.execute("validate_email", {}))
        assert "error" in r
