"""Tests for JSONTool."""

import json
import pytest

from agent_friend.tools.json_tool import JSONTool


@pytest.fixture
def jt():
    return JSONTool()


# ── basic properties ──────────────────────────────────────────────────────────

def test_name(jt):
    assert jt.name == "json"

def test_description(jt):
    assert "json" in jt.description.lower()

def test_definitions_returns_six(jt):
    assert len(jt.definitions()) == 6

def test_definition_names(jt):
    names = {d["name"] for d in jt.definitions()}
    assert names == {"json_get", "json_set", "json_keys", "json_filter", "json_format", "json_merge"}


# ── json_get ──────────────────────────────────────────────────────────────────

def test_get_top_level_key(jt):
    data = '{"name": "Alice", "age": 30}'
    assert json.loads(jt.json_get(data, "name")) == "Alice"

def test_get_nested_key(jt):
    data = '{"user": {"name": "Bob", "role": "admin"}}'
    assert json.loads(jt.json_get(data, "user.name")) == "Bob"

def test_get_array_index(jt):
    data = '{"tags": ["ai", "python", "agents"]}'
    assert json.loads(jt.json_get(data, "tags[0]")) == "ai"
    assert json.loads(jt.json_get(data, "tags[2]")) == "agents"

def test_get_nested_array(jt):
    data = '{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}'
    assert json.loads(jt.json_get(data, "users[1].name")) == "Bob"

def test_get_wildcard(jt):
    data = '[{"id": 1}, {"id": 2}, {"id": 3}]'
    result = json.loads(jt.json_get(data, "[*].id"))
    assert result == [1, 2, 3]

def test_get_integer_value(jt):
    data = '{"count": 42}'
    assert json.loads(jt.json_get(data, "count")) == 42

def test_get_boolean(jt):
    data = '{"active": true}'
    assert json.loads(jt.json_get(data, "active")) is True

def test_get_null(jt):
    data = '{"value": null}'
    assert json.loads(jt.json_get(data, "value")) is None


# ── json_set ──────────────────────────────────────────────────────────────────

def test_set_top_level(jt):
    data = '{"name": "Alice"}'
    result = json.loads(jt.json_set(data, "name", '"Bob"'))
    assert result["name"] == "Bob"

def test_set_nested(jt):
    data = '{"user": {"name": "Alice"}}'
    result = json.loads(jt.json_set(data, "user.name", '"Charlie"'))
    assert result["user"]["name"] == "Charlie"

def test_set_new_key(jt):
    data = '{"name": "Alice"}'
    result = json.loads(jt.json_set(data, "email", '"a@b.com"'))
    assert result["email"] == "a@b.com"
    assert result["name"] == "Alice"  # original preserved

def test_set_integer(jt):
    data = '{"count": 0}'
    result = json.loads(jt.json_set(data, "count", "99"))
    assert result["count"] == 99

def test_set_does_not_mutate_original(jt):
    data = '{"name": "Alice"}'
    jt.json_set(data, "name", '"Bob"')
    # Original data should be unchanged
    assert json.loads(data)["name"] == "Alice"


# ── json_keys ─────────────────────────────────────────────────────────────────

def test_keys_simple(jt):
    data = '{"a": 1, "b": 2, "c": 3}'
    keys = json.loads(jt.json_keys(data))
    assert set(keys) == {"a", "b", "c"}

def test_keys_empty(jt):
    assert json.loads(jt.json_keys("{}")) == []


# ── json_filter ───────────────────────────────────────────────────────────────

def test_filter_by_string(jt):
    data = '[{"role": "admin", "name": "Alice"}, {"role": "user", "name": "Bob"}]'
    result = json.loads(jt.json_filter(data, "role", '"admin"'))
    assert len(result) == 1
    assert result[0]["name"] == "Alice"

def test_filter_by_integer(jt):
    data = '[{"id": 1, "x": "a"}, {"id": 2, "x": "b"}, {"id": 1, "x": "c"}]'
    result = json.loads(jt.json_filter(data, "id", "1"))
    assert len(result) == 2

def test_filter_returns_empty_when_no_match(jt):
    data = '[{"role": "admin"}]'
    result = json.loads(jt.json_filter(data, "role", '"god"'))
    assert result == []

def test_filter_multi_result(jt):
    data = '[{"type": "a"}, {"type": "b"}, {"type": "a"}]'
    result = json.loads(jt.json_filter(data, "type", '"a"'))
    assert len(result) == 2


# ── json_format ───────────────────────────────────────────────────────────────

def test_format_produces_valid_json(jt):
    data = '{"a":1,"b":{"c":2}}'
    formatted = jt.json_format(data)
    assert json.loads(formatted) == {"a": 1, "b": {"c": 2}}
    assert "\n" in formatted  # multi-line

def test_format_custom_indent(jt):
    data = '{"x": 1}'
    formatted = jt.json_format(data, indent=4)
    assert "    " in formatted


# ── json_merge ────────────────────────────────────────────────────────────────

def test_merge_adds_keys(jt):
    base = '{"name": "Alice"}'
    patch = '{"email": "a@b.com"}'
    result = json.loads(jt.json_merge(base, patch))
    assert result == {"name": "Alice", "email": "a@b.com"}

def test_merge_overrides_keys(jt):
    base = '{"name": "Alice", "age": 30}'
    patch = '{"age": 31}'
    result = json.loads(jt.json_merge(base, patch))
    assert result["age"] == 31
    assert result["name"] == "Alice"

def test_merge_empty_patch(jt):
    base = '{"name": "Alice"}'
    result = json.loads(jt.json_merge(base, "{}"))
    assert result == {"name": "Alice"}


# ── execute dispatch ──────────────────────────────────────────────────────────

def test_execute_json_get(jt):
    result = jt.execute("json_get", {"data": '{"x": 42}', "path": "x"})
    assert json.loads(result) == 42

def test_execute_json_keys(jt):
    result = jt.execute("json_keys", {"data": '{"a": 1}'})
    assert json.loads(result) == ["a"]

def test_execute_unknown_tool(jt):
    result = jt.execute("nonexistent", {})
    assert "Unknown" in result

def test_execute_returns_error_on_bad_path(jt):
    result = jt.execute("json_get", {"data": '{"x": 1}', "path": "nonexistent.deep"})
    assert "Error" in result


# ── tool definition schemas ───────────────────────────────────────────────────

def test_json_get_schema_required(jt):
    defs = {d["name"]: d for d in jt.definitions()}
    schema = defs["json_get"]["input_schema"]
    assert set(schema["required"]) == {"data", "path"}

def test_json_filter_schema_required(jt):
    defs = {d["name"]: d for d in jt.definitions()}
    schema = defs["json_filter"]["input_schema"]
    assert set(schema["required"]) == {"data", "key", "value"}

def test_json_format_indent_optional(jt):
    defs = {d["name"]: d for d in jt.definitions()}
    schema = defs["json_format"]["input_schema"]
    assert "indent" in schema["properties"]
    assert "indent" not in schema.get("required", [])
