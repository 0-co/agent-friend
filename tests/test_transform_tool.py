"""Tests for TransformTool — structured data transformation."""

import json
import pytest
from agent_friend.tools.transform_tool import TransformTool


@pytest.fixture
def tool():
    return TransformTool()


# ── transform_pick ────────────────────────────────────────────────────────────

def test_pick_basic(tool):
    r = json.loads(tool.transform_pick({"a": 1, "b": 2, "c": 3}, keys=["a", "c"]))
    assert r["result"] == {"a": 1, "c": 3}
    assert r["picked"] == 2
    assert r["missing"] == []


def test_pick_missing_key(tool):
    r = json.loads(tool.transform_pick({"a": 1}, keys=["a", "z"]))
    assert r["result"] == {"a": 1}
    assert r["picked"] == 1
    assert "z" in r["missing"]


def test_pick_all_missing(tool):
    r = json.loads(tool.transform_pick({"a": 1}, keys=["x", "y"]))
    assert r["result"] == {}
    assert r["picked"] == 0
    assert set(r["missing"]) == {"x", "y"}


def test_pick_empty_keys(tool):
    r = json.loads(tool.transform_pick({"a": 1}, keys=[]))
    assert r["result"] == {}
    assert r["picked"] == 0


def test_pick_non_dict_error(tool):
    r = json.loads(tool.transform_pick("not a dict", keys=["a"]))
    assert "error" in r


def test_pick_non_list_keys_error(tool):
    r = json.loads(tool.transform_pick({"a": 1}, keys="a"))
    assert "error" in r


def test_pick_nested_value(tool):
    r = json.loads(tool.transform_pick({"a": {"x": 1}, "b": 2}, keys=["a"]))
    assert r["result"] == {"a": {"x": 1}}


def test_pick_preserves_types(tool):
    r = json.loads(tool.transform_pick({"a": None, "b": False, "c": 0}, keys=["a", "b", "c"]))
    assert r["result"]["a"] is None
    assert r["result"]["b"] is False
    assert r["result"]["c"] == 0


# ── transform_omit ────────────────────────────────────────────────────────────

def test_omit_basic(tool):
    r = json.loads(tool.transform_omit({"a": 1, "b": 2, "c": 3}, keys=["b"]))
    assert r["result"] == {"a": 1, "c": 3}
    assert r["omitted"] == 1


def test_omit_all(tool):
    r = json.loads(tool.transform_omit({"a": 1, "b": 2}, keys=["a", "b"]))
    assert r["result"] == {}
    assert r["omitted"] == 2


def test_omit_nonexistent_key(tool):
    r = json.loads(tool.transform_omit({"a": 1}, keys=["z"]))
    assert r["result"] == {"a": 1}
    assert r["omitted"] == 0


def test_omit_empty_keys(tool):
    r = json.loads(tool.transform_omit({"a": 1, "b": 2}, keys=[]))
    assert r["result"] == {"a": 1, "b": 2}
    assert r["omitted"] == 0


def test_omit_non_dict_error(tool):
    r = json.loads(tool.transform_omit(42, keys=["a"]))
    assert "error" in r


def test_omit_non_list_keys_error(tool):
    r = json.loads(tool.transform_omit({"a": 1}, keys="a"))
    assert "error" in r


# ── transform_rename ──────────────────────────────────────────────────────────

def test_rename_basic(tool):
    r = json.loads(tool.transform_rename({"name": "alice", "age": 30}, mapping={"name": "full_name"}))
    assert "full_name" in r["result"]
    assert r["result"]["full_name"] == "alice"
    assert "age" in r["result"]
    assert r["renamed"] == 1


def test_rename_multiple(tool):
    r = json.loads(tool.transform_rename({"a": 1, "b": 2}, mapping={"a": "x", "b": "y"}))
    assert r["result"] == {"x": 1, "y": 2}
    assert r["renamed"] == 2


def test_rename_unmapped_kept(tool):
    r = json.loads(tool.transform_rename({"a": 1, "b": 2}, mapping={"a": "x"}))
    assert r["result"]["b"] == 2
    assert "a" not in r["result"]


def test_rename_empty_mapping(tool):
    r = json.loads(tool.transform_rename({"a": 1}, mapping={}))
    assert r["result"] == {"a": 1}
    assert r["renamed"] == 0


def test_rename_non_dict_record_error(tool):
    r = json.loads(tool.transform_rename("bad", mapping={"a": "b"}))
    assert "error" in r


def test_rename_non_dict_mapping_error(tool):
    r = json.loads(tool.transform_rename({"a": 1}, mapping="bad"))
    assert "error" in r


# ── transform_coerce ──────────────────────────────────────────────────────────

def test_coerce_to_int(tool):
    r = json.loads(tool.transform_coerce({"x": "42"}, types={"x": "int"}))
    assert r["result"]["x"] == 42
    assert r["coerced"] == 1
    assert r["errors"] == []


def test_coerce_to_float(tool):
    r = json.loads(tool.transform_coerce({"x": "3.14"}, types={"x": "float"}))
    assert abs(r["result"]["x"] - 3.14) < 0.001
    assert r["coerced"] == 1


def test_coerce_to_str(tool):
    r = json.loads(tool.transform_coerce({"x": 99}, types={"x": "str"}))
    assert r["result"]["x"] == "99"


def test_coerce_to_bool_false_string(tool):
    r = json.loads(tool.transform_coerce({"x": "false"}, types={"x": "bool"}))
    assert r["result"]["x"] is False


def test_coerce_to_bool_true_string(tool):
    r = json.loads(tool.transform_coerce({"x": "yes"}, types={"x": "bool"}))
    assert r["result"]["x"] is True


def test_coerce_to_bool_already_bool(tool):
    r = json.loads(tool.transform_coerce({"x": True}, types={"x": "bool"}))
    assert r["result"]["x"] is True


def test_coerce_to_list_from_json(tool):
    r = json.loads(tool.transform_coerce({"x": "[1,2,3]"}, types={"x": "list"}))
    assert r["result"]["x"] == [1, 2, 3]


def test_coerce_to_list_already_list(tool):
    r = json.loads(tool.transform_coerce({"x": [1, 2]}, types={"x": "list"}))
    assert r["result"]["x"] == [1, 2]


def test_coerce_to_dict_from_json(tool):
    r = json.loads(tool.transform_coerce({"x": '{"a":1}'}, types={"x": "dict"}))
    assert r["result"]["x"] == {"a": 1}


def test_coerce_to_null(tool):
    r = json.loads(tool.transform_coerce({"x": "anything"}, types={"x": "null"}))
    assert r["result"]["x"] is None


def test_coerce_error_recorded(tool):
    r = json.loads(tool.transform_coerce({"x": "notanumber"}, types={"x": "int"}))
    assert len(r["errors"]) == 1
    assert r["errors"][0]["key"] == "x"


def test_coerce_key_not_in_record(tool):
    r = json.loads(tool.transform_coerce({"a": 1}, types={"z": "int"}))
    assert r["coerced"] == 0
    assert r["errors"] == []


def test_coerce_non_dict_record_error(tool):
    r = json.loads(tool.transform_coerce("bad", types={"x": "int"}))
    assert "error" in r


def test_coerce_unknown_type(tool):
    r = json.loads(tool.transform_coerce({"x": 1}, types={"x": "unknowntype"}))
    assert len(r["errors"]) == 1


# ── transform_flatten ─────────────────────────────────────────────────────────

def test_flatten_basic(tool):
    r = json.loads(tool.transform_flatten({"user": {"name": "alice", "age": 30}}))
    assert r["result"]["user.name"] == "alice"
    assert r["result"]["user.age"] == 30
    assert r["key_count"] == 2


def test_flatten_deeply_nested(tool):
    r = json.loads(tool.transform_flatten({"a": {"b": {"c": 42}}}))
    assert r["result"]["a.b.c"] == 42


def test_flatten_with_list(tool):
    r = json.loads(tool.transform_flatten({"tags": ["x", "y"]}))
    assert r["result"]["tags.0"] == "x"
    assert r["result"]["tags.1"] == "y"


def test_flatten_custom_sep(tool):
    r = json.loads(tool.transform_flatten({"a": {"b": 1}}, sep="__"))
    assert "a__b" in r["result"]


def test_flatten_flat_dict_unchanged(tool):
    r = json.loads(tool.transform_flatten({"a": 1, "b": 2}))
    assert r["result"] == {"a": 1, "b": 2}


def test_flatten_non_dict_error(tool):
    r = json.loads(tool.transform_flatten("not a dict"))
    assert "error" in r


def test_flatten_empty_sep_error(tool):
    r = json.loads(tool.transform_flatten({"a": 1}, sep=""))
    assert "error" in r


def test_flatten_list_input(tool):
    r = json.loads(tool.transform_flatten([{"a": 1}, {"b": 2}]))
    assert "0.a" in r["result"]
    assert "1.b" in r["result"]


# ── transform_unflatten ───────────────────────────────────────────────────────

def test_unflatten_basic(tool):
    r = json.loads(tool.transform_unflatten({"user.name": "alice", "user.age": 30}))
    assert r["result"]["user"]["name"] == "alice"
    assert r["result"]["user"]["age"] == 30


def test_unflatten_deeply_nested(tool):
    r = json.loads(tool.transform_unflatten({"a.b.c": 42}))
    assert r["result"]["a"]["b"]["c"] == 42


def test_unflatten_flat_keys_unchanged(tool):
    r = json.loads(tool.transform_unflatten({"a": 1, "b": 2}))
    assert r["result"] == {"a": 1, "b": 2}


def test_unflatten_custom_sep(tool):
    r = json.loads(tool.transform_unflatten({"a__b": 1}, sep="__"))
    assert r["result"]["a"]["b"] == 1


def test_unflatten_non_dict_error(tool):
    r = json.loads(tool.transform_unflatten("bad"))
    assert "error" in r


def test_unflatten_empty_sep_error(tool):
    r = json.loads(tool.transform_unflatten({"a.b": 1}, sep=""))
    assert "error" in r


def test_unflatten_roundtrip(tool):
    original = {"user": {"name": "alice", "scores": {"math": 90}}}
    flat_r = json.loads(tool.transform_flatten(original))
    unflat_r = json.loads(tool.transform_unflatten(flat_r["result"]))
    assert unflat_r["result"]["user"]["name"] == "alice"


# ── transform_map_records ─────────────────────────────────────────────────────

def test_map_records_pick(tool):
    records = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}]
    r = json.loads(tool.transform_map_records(records, pick=["a", "c"]))
    assert r["results"][0] == {"a": 1, "c": 3}
    assert r["count"] == 2


def test_map_records_omit(tool):
    records = [{"a": 1, "b": 2}]
    r = json.loads(tool.transform_map_records(records, omit=["b"]))
    assert r["results"][0] == {"a": 1}


def test_map_records_rename(tool):
    records = [{"n": "alice", "s": 95}]
    r = json.loads(tool.transform_map_records(records, rename={"n": "name", "s": "score"}))
    assert r["results"][0] == {"name": "alice", "score": 95}


def test_map_records_coerce(tool):
    records = [{"score": "95"}]
    r = json.loads(tool.transform_map_records(records, coerce={"score": "int"}))
    assert r["results"][0]["score"] == 95


def test_map_records_add(tool):
    records = [{"a": 1}]
    r = json.loads(tool.transform_map_records(records, add={"source": "test"}))
    assert r["results"][0]["source"] == "test"


def test_map_records_combined(tool):
    records = [{"n": "alice", "s": "95", "junk": True}]
    r = json.loads(tool.transform_map_records(
        records,
        omit=["junk"],
        rename={"n": "name", "s": "score"},
        coerce={"score": "int"},
        add={"version": 1},
    ))
    res = r["results"][0]
    assert res["name"] == "alice"
    assert res["score"] == 95
    assert "junk" not in res
    assert res["version"] == 1


def test_map_records_non_dict_record_skipped(tool):
    records = [{"a": 1}, "bad", {"b": 2}]
    r = json.loads(tool.transform_map_records(records))
    assert r["count"] == 2
    assert len(r["errors"]) == 1


def test_map_records_non_list_error(tool):
    r = json.loads(tool.transform_map_records("not a list"))
    assert "error" in r


def test_map_records_empty_list(tool):
    r = json.loads(tool.transform_map_records([]))
    assert r["results"] == []
    assert r["count"] == 0


def test_map_records_coerce_error_recorded(tool):
    records = [{"x": "bad"}]
    r = json.loads(tool.transform_map_records(records, coerce={"x": "int"}))
    assert len(r["errors"]) == 1


# ── transform_merge ───────────────────────────────────────────────────────────

def test_merge_basic(tool):
    r = json.loads(tool.transform_merge({"a": 1}, {"b": 2}))
    assert r["result"] == {"a": 1, "b": 2}
    assert r["merged_from"] == 2


def test_merge_override(tool):
    r = json.loads(tool.transform_merge({"a": 1}, {"a": 2}))
    assert r["result"]["a"] == 2


def test_merge_deep(tool):
    r = json.loads(tool.transform_merge({"x": {"a": 1}}, {"x": {"b": 2}}))
    assert r["result"]["x"]["a"] == 1
    assert r["result"]["x"]["b"] == 2


def test_merge_three_dicts(tool):
    r = json.loads(tool.transform_merge({"a": 1}, {"b": 2}, {"c": 3}))
    assert r["result"] == {"a": 1, "b": 2, "c": 3}
    assert r["merged_from"] == 3


def test_merge_single_dict(tool):
    r = json.loads(tool.transform_merge({"a": 1}))
    assert r["result"] == {"a": 1}
    assert r["merged_from"] == 1


def test_merge_empty_no_args_error(tool):
    r = json.loads(tool.transform_merge())
    assert "error" in r


def test_merge_non_dict_arg_error(tool):
    r = json.loads(tool.transform_merge({"a": 1}, "bad"))
    assert "error" in r


def test_merge_empty_dicts(tool):
    r = json.loads(tool.transform_merge({}, {}, {}))
    assert r["result"] == {}
    assert r["merged_from"] == 3


# ── execute dispatch ──────────────────────────────────────────────────────────

def test_execute_pick(tool):
    r = json.loads(tool.execute("transform_pick", {"record": {"a": 1, "b": 2}, "keys": ["a"]}))
    assert r["result"] == {"a": 1}


def test_execute_merge(tool):
    r = json.loads(tool.execute("transform_merge", {"dicts": [{"a": 1}, {"b": 2}]}))
    assert r["result"] == {"a": 1, "b": 2}


def test_execute_unknown(tool):
    r = json.loads(tool.execute("transform_unknown", {}))
    assert "error" in r


# ── metadata ──────────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "transform"


def test_description_nonempty(tool):
    assert len(tool.description) > 10


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8
