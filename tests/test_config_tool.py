"""Tests for ConfigTool — hierarchical key-value configuration."""

import json
import os
import pytest
from agent_friend.tools.config_tool import ConfigTool


@pytest.fixture
def tool():
    return ConfigTool()


# ── config_set / config_get ────────────────────────────────────────────────

def test_set_and_get_string(tool):
    tool.config_set("app", "host", "localhost")
    r = json.loads(tool.config_get("app", "host"))
    assert r["value"] == "localhost"
    assert r["found"] is True


def test_set_and_get_int(tool):
    tool.config_set("app", "port", 5432)
    r = json.loads(tool.config_get("app", "port"))
    assert r["value"] == 5432


def test_set_and_get_bool(tool):
    tool.config_set("app", "debug", True)
    r = json.loads(tool.config_get("app", "debug"))
    assert r["value"] is True


def test_set_and_get_none(tool):
    tool.config_set("app", "x", None)
    r = json.loads(tool.config_get("app", "x"))
    assert r["value"] is None
    assert r["found"] is True


def test_get_missing_key_returns_default(tool):
    r = json.loads(tool.config_get("app", "missing"))
    assert r["value"] is None
    assert r["found"] is False


def test_get_missing_key_with_explicit_default(tool):
    r = json.loads(tool.config_get("app", "missing", default="fallback"))
    assert r["value"] == "fallback"
    assert r["found"] is False


def test_get_from_missing_config_store(tool):
    r = json.loads(tool.config_get("no_such_store", "key"))
    assert r["found"] is False
    assert r["value"] is None


def test_set_returns_ok(tool):
    r = json.loads(tool.config_set("app", "k", "v"))
    assert r["ok"] is True
    assert r["key"] == "k"
    assert r["value"] == "v"


def test_set_overwrites_existing(tool):
    tool.config_set("app", "k", "v1")
    tool.config_set("app", "k", "v2")
    r = json.loads(tool.config_get("app", "k"))
    assert r["value"] == "v2"


def test_dot_notation_key(tool):
    tool.config_set("app", "db.host", "127.0.0.1")
    r = json.loads(tool.config_get("app", "db.host"))
    assert r["value"] == "127.0.0.1"


def test_dot_notation_multiple_levels(tool):
    tool.config_set("app", "a.b.c", 42)
    r = json.loads(tool.config_get("app", "a.b.c"))
    assert r["value"] == 42


# ── type coercion ──────────────────────────────────────────────────────────

def test_coerce_to_int(tool):
    tool.config_set("app", "port", "8080")
    r = json.loads(tool.config_get("app", "port", as_type="int"))
    assert r["value"] == 8080


def test_coerce_to_float(tool):
    tool.config_set("app", "rate", "3.14")
    r = json.loads(tool.config_get("app", "rate", as_type="float"))
    assert abs(r["value"] - 3.14) < 0.001


def test_coerce_to_bool_true_string(tool):
    tool.config_set("app", "flag", "true")
    r = json.loads(tool.config_get("app", "flag", as_type="bool"))
    assert r["value"] is True


def test_coerce_to_bool_yes(tool):
    tool.config_set("app", "flag", "yes")
    r = json.loads(tool.config_get("app", "flag", as_type="bool"))
    assert r["value"] is True


def test_coerce_to_bool_one(tool):
    tool.config_set("app", "flag", "1")
    r = json.loads(tool.config_get("app", "flag", as_type="bool"))
    assert r["value"] is True


def test_coerce_to_bool_false_string(tool):
    tool.config_set("app", "flag", "false")
    r = json.loads(tool.config_get("app", "flag", as_type="bool"))
    assert r["value"] is False


def test_coerce_to_bool_already_bool(tool):
    tool.config_set("app", "flag", True)
    r = json.loads(tool.config_get("app", "flag", as_type="bool"))
    assert r["value"] is True


def test_coerce_to_str(tool):
    tool.config_set("app", "x", 42)
    r = json.loads(tool.config_get("app", "x", as_type="str"))
    assert r["value"] == "42"


def test_coerce_to_json_parses_string(tool):
    tool.config_set("app", "data", '[1, 2, 3]')
    r = json.loads(tool.config_get("app", "data", as_type="json"))
    assert r["value"] == [1, 2, 3]


def test_coerce_to_json_passthrough_list(tool):
    tool.config_set("app", "data", [1, 2, 3])
    r = json.loads(tool.config_get("app", "data", as_type="json"))
    assert r["value"] == [1, 2, 3]


def test_coerce_invalid_int_returns_error(tool):
    tool.config_set("app", "x", "not_a_number")
    r = json.loads(tool.config_get("app", "x", as_type="int"))
    assert "error" in r


def test_coerce_invalid_json_returns_error(tool):
    tool.config_set("app", "x", "not json {{{")
    r = json.loads(tool.config_get("app", "x", as_type="json"))
    assert "error" in r


def test_coerce_none_skipped(tool):
    # default is None, as_type should not be applied to None
    r = json.loads(tool.config_get("app", "missing", as_type="int"))
    assert r["value"] is None


# ── config_defaults ────────────────────────────────────────────────────────

def test_defaults_sets_new_keys(tool):
    r = json.loads(tool.config_defaults("app", {"a": 1, "b": 2}))
    assert r["set"] == 2


def test_defaults_skips_existing_keys(tool):
    tool.config_set("app", "a", 99)
    r = json.loads(tool.config_defaults("app", {"a": 1, "b": 2}))
    assert r["set"] == 1
    # existing key unchanged
    assert json.loads(tool.config_get("app", "a"))["value"] == 99


def test_defaults_empty_dict(tool):
    r = json.loads(tool.config_defaults("app", {}))
    assert r["set"] == 0


def test_defaults_creates_new_store(tool):
    r = json.loads(tool.config_defaults("brand_new", {"k": "v"}))
    assert r["set"] == 1
    assert r["name"] == "brand_new"


# ── config_list ────────────────────────────────────────────────────────────

def test_list_keys_all(tool):
    tool.config_set("app", "a", 1)
    tool.config_set("app", "b", 2)
    tool.config_set("app", "c", 3)
    keys = json.loads(tool.config_list("app"))
    assert sorted(keys) == ["a", "b", "c"]


def test_list_keys_with_prefix(tool):
    tool.config_set("app", "db.host", "x")
    tool.config_set("app", "db.port", 1)
    tool.config_set("app", "cache.host", "y")
    keys = json.loads(tool.config_list("app", prefix="db."))
    assert "db.host" in keys
    assert "db.port" in keys
    assert "cache.host" not in keys


def test_list_missing_store_empty(tool):
    keys = json.loads(tool.config_list("no_such"))
    assert keys == []


# ── config_delete ──────────────────────────────────────────────────────────

def test_delete_existing_key(tool):
    tool.config_set("app", "k", "v")
    r = json.loads(tool.config_delete("app", "k"))
    assert r["deleted"] is True
    # should be gone
    r2 = json.loads(tool.config_get("app", "k"))
    assert r2["found"] is False


def test_delete_missing_key(tool):
    r = json.loads(tool.config_delete("app", "nope"))
    assert r["deleted"] is False


def test_delete_from_missing_store(tool):
    r = json.loads(tool.config_delete("no_store", "k"))
    assert r["deleted"] is False


# ── config_dump ────────────────────────────────────────────────────────────

def test_dump_returns_all_keys(tool):
    tool.config_set("app", "a", 1)
    tool.config_set("app", "b", "hello")
    d = json.loads(tool.config_dump("app"))
    assert d["a"] == 1
    assert d["b"] == "hello"


def test_dump_missing_store_empty(tool):
    d = json.loads(tool.config_dump("no_such"))
    assert d == {}


# ── config_require ─────────────────────────────────────────────────────────

def test_require_all_present(tool):
    tool.config_set("app", "a", 1)
    tool.config_set("app", "b", 2)
    r = json.loads(tool.config_require("app", ["a", "b"]))
    assert r["ok"] is True
    assert r["missing"] == []


def test_require_some_missing(tool):
    tool.config_set("app", "a", 1)
    r = json.loads(tool.config_require("app", ["a", "b", "c"]))
    assert r["ok"] is False
    assert "b" in r["missing"]
    assert "c" in r["missing"]


def test_require_none_value_counts_as_missing(tool):
    tool.config_set("app", "a", None)
    r = json.loads(tool.config_require("app", ["a"]))
    assert r["ok"] is False
    assert "a" in r["missing"]


def test_require_missing_store(tool):
    r = json.loads(tool.config_require("no_store", ["a"]))
    assert r["ok"] is False
    assert "a" in r["missing"]


# ── config_load_env ────────────────────────────────────────────────────────

def test_load_env_with_prefix(tool, monkeypatch):
    monkeypatch.setenv("MYAPP_HOST", "localhost")
    monkeypatch.setenv("MYAPP_PORT", "8080")
    monkeypatch.setenv("OTHER_VAR", "ignored")
    r = json.loads(tool.config_load_env("app", prefix="MYAPP_"))
    assert r["loaded"] == 2
    assert "host" in r["keys"]
    assert "port" in r["keys"]


def test_load_env_strip_prefix_true(tool, monkeypatch):
    monkeypatch.setenv("APP_HOST", "localhost")
    tool.config_load_env("app", prefix="APP_", strip_prefix=True)
    r = json.loads(tool.config_get("app", "host"))
    assert r["found"] is True


def test_load_env_no_strip_prefix(tool, monkeypatch):
    monkeypatch.setenv("APP_HOST", "localhost")
    tool.config_load_env("app", prefix="APP_", strip_prefix=False)
    r = json.loads(tool.config_get("app", "app_host"))
    assert r["found"] is True


def test_load_env_double_underscore_to_dot(tool, monkeypatch):
    monkeypatch.setenv("APP_DB__HOST", "localhost")
    tool.config_load_env("app", prefix="APP_", strip_prefix=True, lowercase=True)
    r = json.loads(tool.config_get("app", "db.host"))
    assert r["found"] is True


def test_load_env_no_prefix_loads_all(tool, monkeypatch):
    monkeypatch.setenv("TEST_ONLY_VAR_XYZ", "abc123")
    r = json.loads(tool.config_load_env("env_all", prefix=""))
    assert r["loaded"] >= 1


# ── config_drop / config_list_stores ──────────────────────────────────────

def test_drop_existing_store(tool):
    tool.config_set("tmp", "k", "v")
    r = json.loads(tool.config_drop("tmp"))
    assert r["dropped"] is True
    assert r["name"] == "tmp"
    # gone from stores
    stores = json.loads(tool.config_list_stores())
    names = [s["name"] for s in stores]
    assert "tmp" not in names


def test_drop_missing_store_error(tool):
    r = json.loads(tool.config_drop("no_such"))
    assert "error" in r


def test_list_stores_empty(tool):
    stores = json.loads(tool.config_list_stores())
    assert stores == []


def test_list_stores_multiple(tool):
    tool.config_set("a", "k", 1)
    tool.config_set("b", "k1", 1)
    tool.config_set("b", "k2", 2)
    stores = json.loads(tool.config_list_stores())
    by_name = {s["name"]: s for s in stores}
    assert "a" in by_name
    assert by_name["a"]["keys"] == 1
    assert by_name["b"]["keys"] == 2


# ── limits ─────────────────────────────────────────────────────────────────

def test_max_configs_enforced():
    t = ConfigTool(max_configs=2)
    t.config_set("s1", "k", 1)
    t.config_set("s2", "k", 1)
    r = json.loads(t.config_set("s3", "k", 1))
    assert "error" in r


def test_max_keys_enforced():
    t = ConfigTool(max_keys=2)
    t.config_set("app", "a", 1)
    t.config_set("app", "b", 2)
    r = json.loads(t.config_set("app", "c", 3))
    assert "error" in r


def test_overwrite_does_not_count_against_max_keys():
    t = ConfigTool(max_keys=2)
    t.config_set("app", "a", 1)
    t.config_set("app", "b", 2)
    # overwriting existing key should succeed
    r = json.loads(t.config_set("app", "a", 99))
    assert r["ok"] is True


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_config_set(tool):
    r = json.loads(tool.execute("config_set", {"name": "app", "key": "x", "value": 1}))
    assert r["ok"] is True


def test_execute_config_get(tool):
    tool.config_set("app", "y", 42)
    r = json.loads(tool.execute("config_get", {"name": "app", "key": "y"}))
    assert r["value"] == 42


def test_execute_config_list(tool):
    tool.config_set("app", "a", 1)
    r = json.loads(tool.execute("config_list", {"name": "app"}))
    assert "a" in r


def test_execute_unknown_tool(tool):
    r = json.loads(tool.execute("unknown_tool", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "config"


def test_description(tool):
    assert "config" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 10


def test_definitions_have_required_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
