"""Tests for RegexTool."""

import json
import re
import pytest

from agent_friend.tools.regex_tool import RegexTool


@pytest.fixture
def tool():
    return RegexTool()


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "regex"


def test_description(tool):
    assert "regular expression" in tool.description.lower() or "regex" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 9


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "regex_match", "regex_search", "regex_findall",
        "regex_findall_with_positions", "regex_replace",
        "regex_split", "regex_extract_groups",
        "regex_validate", "regex_escape",
    }


# ── regex_match ────────────────────────────────────────────────────────────────


def test_match_success(tool):
    result = json.loads(tool.regex_match(r"\d+", "123abc"))
    assert result["matched"] is True
    assert result["match"] == "123"
    assert result["start"] == 0


def test_match_fail(tool):
    result = json.loads(tool.regex_match(r"\d+", "abc"))
    assert result["matched"] is False


def test_match_groups(tool):
    result = json.loads(tool.regex_match(r"(\w+)\s+(\w+)", "hello world"))
    assert result["groups"] == ["hello", "world"]


def test_match_named_groups(tool):
    result = json.loads(tool.regex_match(r"(?P<first>\w+)\s+(?P<second>\w+)", "hello world"))
    assert result["named_groups"] == {"first": "hello", "second": "world"}


def test_match_not_at_start_fails(tool):
    # regex_match only matches at start
    result = json.loads(tool.regex_match(r"\d+", "abc123"))
    assert result["matched"] is False


def test_match_bad_pattern(tool):
    result = json.loads(tool.regex_match(r"(\d+", "abc"))
    assert "error" in result


def test_match_ignorecase_flag(tool):
    result = json.loads(tool.regex_match(r"hello", "HELLO world", flags=["IGNORECASE"]))
    assert result["matched"] is True


# ── regex_search ───────────────────────────────────────────────────────────────


def test_search_finds_anywhere(tool):
    result = json.loads(tool.regex_search(r"\d+", "abc123def"))
    assert result["matched"] is True
    assert result["match"] == "123"


def test_search_first_match_only(tool):
    result = json.loads(tool.regex_search(r"\d+", "abc 1 and 2"))
    assert result["match"] == "1"


def test_search_no_match(tool):
    result = json.loads(tool.regex_search(r"\d+", "no numbers here"))
    assert result["matched"] is False


def test_search_version_string(tool):
    result = json.loads(tool.regex_search(r"\d+\.\d+\.\d+", "Version 0.27.0 released"))
    assert result["matched"] is True
    assert result["match"] == "0.27.0"


def test_search_email(tool):
    result = json.loads(tool.regex_search(r"\w+@\w+\.\w+", "Contact alice@example.com please"))
    assert result["match"] == "alice@example.com"


def test_search_bad_pattern(tool):
    result = json.loads(tool.regex_search(r"[invalid", "text"))
    assert "error" in result


# ── regex_findall ──────────────────────────────────────────────────────────────


def test_findall_numbers(tool):
    result = json.loads(tool.regex_findall(r"\d+", "1 apple, 2 bananas, 3 oranges"))
    assert result == ["1", "2", "3"]


def test_findall_emails(tool):
    result = json.loads(tool.regex_findall(r"\w+@\w+\.\w+", "a@x.com b@y.org c@z.net"))
    assert len(result) == 3


def test_findall_with_group_returns_groups(tool):
    result = json.loads(tool.regex_findall(r"(\w+)=(\w+)", "a=1 b=2 c=3"))
    # With 2 groups, returns list of tuples (as lists in JSON)
    assert len(result) == 3


def test_findall_no_matches(tool):
    result = json.loads(tool.regex_findall(r"\d+", "no digits"))
    assert result == []


def test_findall_max_results(tool):
    t = RegexTool(max_results=3)
    result = json.loads(t.regex_findall(r"\d", "1 2 3 4 5 6 7 8 9"))
    assert len(result) == 3


def test_findall_bad_pattern(tool):
    result = json.loads(tool.regex_findall(r"[bad", "text"))
    assert "error" in result


# ── regex_findall_with_positions ──────────────────────────────────────────────


def test_findall_with_positions_structure(tool):
    result = json.loads(tool.regex_findall_with_positions(r"\d+", "a1b22c333"))
    assert len(result) == 3
    assert result[0]["match"] == "1"
    assert result[1]["match"] == "22"
    assert result[2]["match"] == "333"


def test_findall_with_positions_start_end(tool):
    result = json.loads(tool.regex_findall_with_positions(r"\d+", "x12y"))
    assert result[0]["start"] == 1
    assert result[0]["end"] == 3


def test_findall_with_positions_empty(tool):
    result = json.loads(tool.regex_findall_with_positions(r"\d+", "no digits"))
    assert result == []


def test_findall_with_positions_groups(tool):
    result = json.loads(tool.regex_findall_with_positions(r"(\w+)=(\w+)", "a=1"))
    assert result[0]["groups"] == ["a", "1"]


# ── regex_replace ──────────────────────────────────────────────────────────────


def test_replace_all(tool):
    result = tool.regex_replace(r"\s+", " ", "too   many    spaces")
    assert result == "too many spaces"


def test_replace_count(tool):
    result = tool.regex_replace(r"x", "y", "x x x x", count=2)
    assert result == "y y x x"


def test_replace_backreference(tool):
    result = tool.regex_replace(r"(\w+)\s+(\w+)", r"\2 \1", "hello world")
    assert result == "world hello"


def test_replace_no_match(tool):
    result = tool.regex_replace(r"\d+", "N", "no digits here")
    assert result == "no digits here"


def test_replace_redact(tool):
    result = tool.regex_replace(r"\b\d{4}\b", "****", "card: 1234 and 5678")
    assert "****" in result
    assert "1234" not in result


def test_replace_bad_pattern(tool):
    result = json.loads(tool.regex_replace(r"[bad", "x", "text"))
    assert "error" in result


def test_replace_ignorecase(tool):
    result = tool.regex_replace(r"hello", "Hi", "Hello World hello", flags=["IGNORECASE"])
    assert "Hi" in result


# ── regex_split ────────────────────────────────────────────────────────────────


def test_split_whitespace(tool):
    result = json.loads(tool.regex_split(r"\s+", "one two   three"))
    assert result == ["one", "two", "three"]


def test_split_comma(tool):
    result = json.loads(tool.regex_split(r",\s*", "a, b, c"))
    assert result == ["a", "b", "c"]


def test_split_maxsplit(tool):
    result = json.loads(tool.regex_split(r",", "a,b,c,d", maxsplit=2))
    assert result == ["a", "b", "c,d"]


def test_split_no_match(tool):
    result = json.loads(tool.regex_split(r"X", "abc"))
    assert result == ["abc"]


def test_split_bad_pattern(tool):
    result = json.loads(tool.regex_split(r"[bad", "text"))
    assert "error" in result


# ── regex_extract_groups ──────────────────────────────────────────────────────


def test_extract_groups_basic(tool):
    result = json.loads(tool.regex_extract_groups(r"(\w+)@(\w+)\.(\w+)", "a@b.com c@d.org"))
    assert len(result) == 2
    assert result[0]["groups"] == ["a", "b", "com"]


def test_extract_groups_named(tool):
    result = json.loads(tool.regex_extract_groups(
        r"(?P<user>\w+)@(?P<domain>\w+)\.(?P<tld>\w+)",
        "alice@example.com"
    ))
    assert result[0]["named_groups"] == {"user": "alice", "domain": "example", "tld": "com"}


def test_extract_groups_no_match(tool):
    result = json.loads(tool.regex_extract_groups(r"(\d+)", "no digits"))
    assert result == []


def test_extract_groups_bad_pattern(tool):
    result = json.loads(tool.regex_extract_groups(r"[bad", "text"))
    assert "error" in result


# ── regex_validate ─────────────────────────────────────────────────────────────


def test_validate_good_pattern(tool):
    result = json.loads(tool.regex_validate(r"\d+\.\d+"))
    assert result["valid"] is True


def test_validate_bad_pattern(tool):
    result = json.loads(tool.regex_validate(r"(\d+"))
    assert result["valid"] is False
    assert "error" in result


def test_validate_complex_valid(tool):
    result = json.loads(tool.regex_validate(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})"))
    assert result["valid"] is True


def test_validate_empty_pattern(tool):
    result = json.loads(tool.regex_validate(r""))
    assert result["valid"] is True  # empty pattern is valid re


# ── regex_escape ───────────────────────────────────────────────────────────────


def test_escape_special_chars(tool):
    result = tool.regex_escape("1+1=2 (always)")
    # Special chars should be escaped
    assert "\\+" in result or re.escape("1+1=2 (always)") == result


def test_escape_plain_text(tool):
    result = tool.regex_escape("hello")
    assert "hello" in result


def test_escape_makes_literal_match(tool):
    text = "price: $1.00 (sale!)"
    escaped = tool.regex_escape("$1.00")
    found = json.loads(tool.regex_findall(escaped, text))
    assert "$1.00" in found


# ── flags ──────────────────────────────────────────────────────────────────────


def test_ignorecase_flag(tool):
    result = json.loads(tool.regex_findall("hello", "Hello HELLO hello", flags=["IGNORECASE"]))
    assert len(result) == 3


def test_multiline_flag(tool):
    text = "line1\nline2\nline3"
    result = json.loads(tool.regex_findall(r"^\w+", text, flags=["MULTILINE"]))
    assert len(result) == 3


def test_dotall_flag(tool):
    text = "start\nmiddle\nend"
    result = json.loads(tool.regex_search(r"start.*end", text, flags=["DOTALL"]))
    assert result["matched"] is True


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_regex_match(tool):
    result = json.loads(tool.execute("regex_match", {"pattern": r"\d+", "text": "123"}))
    assert result["matched"] is True


def test_execute_regex_search(tool):
    result = json.loads(tool.execute("regex_search", {"pattern": r"\d+", "text": "abc123"}))
    assert result["matched"] is True


def test_execute_regex_findall(tool):
    result = json.loads(tool.execute("regex_findall", {"pattern": r"\d+", "text": "1 2 3"}))
    assert result == ["1", "2", "3"]


def test_execute_regex_findall_with_positions(tool):
    result = json.loads(tool.execute("regex_findall_with_positions", {"pattern": r"\d", "text": "1a2"}))
    assert len(result) == 2


def test_execute_regex_replace(tool):
    result = tool.execute("regex_replace", {"pattern": r"\s+", "replacement": "-", "text": "a b c"})
    assert result == "a-b-c"


def test_execute_regex_split(tool):
    result = json.loads(tool.execute("regex_split", {"pattern": r",", "text": "a,b,c"}))
    assert result == ["a", "b", "c"]


def test_execute_regex_extract_groups(tool):
    result = json.loads(tool.execute("regex_extract_groups", {"pattern": r"(\w+)=(\w+)", "text": "x=1"}))
    assert result[0]["groups"] == ["x", "1"]


def test_execute_regex_validate(tool):
    result = json.loads(tool.execute("regex_validate", {"pattern": r"\d+"}))
    assert result["valid"] is True


def test_execute_regex_escape(tool):
    result = tool.execute("regex_escape", {"text": "a+b"})
    assert "+" not in result or result == re.escape("a+b")


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
