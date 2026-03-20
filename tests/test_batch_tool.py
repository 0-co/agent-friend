"""Tests for BatchTool — batch processing with map/filter/reduce/partition."""

import json
import pytest
from agent_friend.tools.batch_tool import BatchTool


@pytest.fixture
def tool():
    return BatchTool()


# ── fn_define ────────────────────────────────────────────────────────────────

def test_fn_define_basic(tool):
    r = json.loads(tool.fn_define("triple", "def fn(item): return item * 3"))
    assert r["registered"] is True
    assert r["name"] == "triple"


def test_fn_define_used_in_map(tool):
    tool.fn_define("triple", "def fn(item): return item * 3")
    r = json.loads(tool.batch_map([1, 2, 3], fn="triple"))
    assert r["results"] == [3, 6, 9]


def test_fn_define_empty_name_error(tool):
    r = json.loads(tool.fn_define("", "def fn(item): return item"))
    assert "error" in r


def test_fn_define_no_fn_error(tool):
    r = json.loads(tool.fn_define("x", "def other(item): return item"))
    assert "error" in r


def test_fn_define_syntax_error(tool):
    r = json.loads(tool.fn_define("x", "def fn(item return item"))
    assert "error" in r


def test_fn_define_conflicts_builtin(tool):
    r = json.loads(tool.fn_define("upper", "def fn(item): return item"))
    assert "error" in r


def test_fn_define_reducer(tool):
    r = json.loads(tool.fn_define("concat_space", "def fn(acc, item): return str(acc) + ' ' + str(item)", is_reducer=True))
    assert r["registered"] is True
    assert r["is_reducer"] is True


def test_fn_define_reducer_used(tool):
    tool.fn_define("add_all", "def fn(acc, item): return acc + item", is_reducer=True)
    r = json.loads(tool.batch_reduce([1, 2, 3, 4], fn="add_all", initial=0))
    assert r["result"] == 10


# ── batch_map ────────────────────────────────────────────────────────────────

def test_map_builtin_upper(tool):
    r = json.loads(tool.batch_map(["a", "b", "c"], fn="upper"))
    assert r["results"] == ["A", "B", "C"]


def test_map_builtin_double(tool):
    r = json.loads(tool.batch_map([1, 2, 3], fn="double"))
    assert r["results"] == [2, 4, 6]


def test_map_builtin_square(tool):
    r = json.loads(tool.batch_map([2, 3, 4], fn="square"))
    assert r["results"] == [4, 9, 16]


def test_map_builtin_len(tool):
    r = json.loads(tool.batch_map(["hi", "hello", "hey"], fn="len"))
    assert r["results"] == [2, 5, 3]


def test_map_ok_count(tool):
    r = json.loads(tool.batch_map([1, 2, 3], fn="double"))
    assert r["ok"] == 3
    assert r["errors"] == 0


def test_map_empty_list(tool):
    r = json.loads(tool.batch_map([], fn="upper"))
    assert r["results"] == []
    assert r["ok"] == 0


def test_map_on_error_null(tool):
    tool.fn_define("bad", "def fn(item): return 1/item")
    r = json.loads(tool.batch_map([1, 0, 2], fn="bad", on_error="null"))
    assert r["results"][0] == 1.0
    assert r["results"][1] is None
    assert r["ok"] == 2
    assert r["errors"] == 1


def test_map_on_error_skip(tool):
    tool.fn_define("bad", "def fn(item): return 1/item")
    r = json.loads(tool.batch_map([1, 0, 2], fn="bad", on_error="skip"))
    assert len(r["results"]) == 2
    assert r["errors"] == 1


def test_map_on_error_raise(tool):
    tool.fn_define("bad", "def fn(item): return 1/item")
    r = json.loads(tool.batch_map([1, 0, 2], fn="bad", on_error="raise"))
    assert "error" in r
    assert r["index"] == 1


def test_map_elapsed_ms(tool):
    r = json.loads(tool.batch_map([1, 2, 3], fn="identity"))
    assert "elapsed_ms" in r


def test_map_unknown_fn_error(tool):
    r = json.loads(tool.batch_map([1, 2], fn="does_not_exist"))
    assert "error" in r


def test_map_not_list_error(tool):
    r = json.loads(tool.batch_map("not a list", fn="upper"))
    assert "error" in r


def test_map_invalid_on_error(tool):
    r = json.loads(tool.batch_map([1], fn="identity", on_error="explode"))
    assert "error" in r


# ── batch_filter ──────────────────────────────────────────────────────────────

def test_filter_is_truthy(tool):
    r = json.loads(tool.batch_filter([0, 1, "", "a", [], [1]], fn="is_truthy"))
    assert r["results"] == [1, "a", [1]]
    assert r["kept"] == 3


def test_filter_custom(tool):
    tool.fn_define("is_even", "def fn(item): return item % 2 == 0")
    r = json.loads(tool.batch_filter([1, 2, 3, 4, 5], fn="is_even"))
    assert r["results"] == [2, 4]
    assert r["kept"] == 2
    assert r["rejected"] == 3


def test_filter_all_pass(tool):
    r = json.loads(tool.batch_filter([1, 2, 3], fn="is_truthy"))
    assert r["kept"] == 3
    assert r["rejected"] == 0


def test_filter_none_pass(tool):
    r = json.loads(tool.batch_filter([0, 0, 0], fn="is_truthy"))
    assert r["kept"] == 0


def test_filter_empty(tool):
    r = json.loads(tool.batch_filter([], fn="is_truthy"))
    assert r["results"] == []


def test_filter_unknown_fn_error(tool):
    r = json.loads(tool.batch_filter([1, 2], fn="nope"))
    assert "error" in r


# ── batch_reduce ──────────────────────────────────────────────────────────────

def test_reduce_sum(tool):
    r = json.loads(tool.batch_reduce([1, 2, 3, 4, 5], fn="sum"))
    assert r["result"] == 15


def test_reduce_product(tool):
    r = json.loads(tool.batch_reduce([1, 2, 3, 4], fn="product"))
    assert r["result"] == 24


def test_reduce_max(tool):
    r = json.loads(tool.batch_reduce([3, 1, 4, 1, 5, 9, 2], fn="max"))
    assert r["result"] == 9


def test_reduce_min(tool):
    r = json.loads(tool.batch_reduce([3, 1, 4, 1, 5], fn="min"))
    assert r["result"] == 1


def test_reduce_concat(tool):
    r = json.loads(tool.batch_reduce(["a", "b", "c"], fn="concat"))
    assert r["result"] == "abc"


def test_reduce_with_initial(tool):
    r = json.loads(tool.batch_reduce([1, 2, 3], fn="sum", initial=10))
    assert r["result"] == 16


def test_reduce_empty_error(tool):
    r = json.loads(tool.batch_reduce([], fn="sum"))
    assert "error" in r


def test_reduce_unknown_fn_error(tool):
    r = json.loads(tool.batch_reduce([1, 2], fn="nope"))
    assert "error" in r


def test_reduce_total_count(tool):
    r = json.loads(tool.batch_reduce([1, 2, 3], fn="sum"))
    assert r["total"] == 3


# ── batch_partition ───────────────────────────────────────────────────────────

def test_partition_basic(tool):
    tool.fn_define("pos", "def fn(item): return item > 0")
    r = json.loads(tool.batch_partition([-1, 2, -3, 4, 0], fn="pos"))
    assert r["passing"] == [2, 4]
    assert r["failing"] == [-1, -3, 0]
    assert r["pass_count"] == 2
    assert r["fail_count"] == 3


def test_partition_all_pass(tool):
    r = json.loads(tool.batch_partition([1, 2, 3], fn="is_truthy"))
    assert r["pass_count"] == 3
    assert r["fail_count"] == 0


def test_partition_none_pass(tool):
    r = json.loads(tool.batch_partition([0, "", []], fn="is_truthy"))
    assert r["pass_count"] == 0


def test_partition_empty(tool):
    r = json.loads(tool.batch_partition([], fn="is_truthy"))
    assert r["passing"] == []
    assert r["failing"] == []


def test_partition_unknown_fn_error(tool):
    r = json.loads(tool.batch_partition([1, 2], fn="nope"))
    assert "error" in r


# ── batch_chunk ───────────────────────────────────────────────────────────────

def test_chunk_basic(tool):
    r = json.loads(tool.batch_chunk([1, 2, 3, 4, 5], size=2))
    assert r["chunks"] == [[1, 2], [3, 4], [5]]
    assert r["chunk_count"] == 3


def test_chunk_exact(tool):
    r = json.loads(tool.batch_chunk([1, 2, 3, 4], size=2))
    assert r["chunks"] == [[1, 2], [3, 4]]


def test_chunk_size_1(tool):
    r = json.loads(tool.batch_chunk([1, 2, 3], size=1))
    assert r["chunks"] == [[1], [2], [3]]


def test_chunk_larger_than_list(tool):
    r = json.loads(tool.batch_chunk([1, 2, 3], size=10))
    assert r["chunks"] == [[1, 2, 3]]


def test_chunk_empty(tool):
    r = json.loads(tool.batch_chunk([], size=3))
    assert r["chunks"] == []
    assert r["chunk_count"] == 0


def test_chunk_size_zero_error(tool):
    r = json.loads(tool.batch_chunk([1, 2, 3], size=0))
    assert "error" in r


# ── batch_zip ─────────────────────────────────────────────────────────────────

def test_zip_basic(tool):
    r = json.loads(tool.batch_zip(["name", "score"], ["alice", "bob"], [95, 87]))
    assert r["results"] == [{"name": "alice", "score": 95}, {"name": "bob", "score": 87}]
    assert r["count"] == 2


def test_zip_single_list(tool):
    r = json.loads(tool.batch_zip(["x"], [10, 20, 30]))
    assert r["results"] == [{"x": 10}, {"x": 20}, {"x": 30}]


def test_zip_empty_keys_error(tool):
    r = json.loads(tool.batch_zip([]))
    assert "error" in r


def test_zip_length_mismatch_error(tool):
    r = json.loads(tool.batch_zip(["a", "b"], [1, 2], [3]))
    assert "error" in r


# ── batch_stats ───────────────────────────────────────────────────────────────

def test_stats_no_run(tool):
    r = json.loads(tool.batch_stats())
    assert "error" in r


def test_stats_after_map(tool):
    tool.batch_map([1, 2, 3], fn="double")
    r = json.loads(tool.batch_stats())
    assert r["ok"] == 3
    assert r["errors"] == 0
    assert r["total"] == 3


# ── builtin_fns ───────────────────────────────────────────────────────────────

def test_builtin_fns_list(tool):
    r = json.loads(tool.builtin_fns())
    assert "fns" in r
    assert "reducers" in r
    assert "upper" in r["fns"]
    assert "sum" in r["reducers"]


# ── execute dispatch ──────────────────────────────────────────────────────────

def test_execute_map(tool):
    r = json.loads(tool.execute("batch_map", {"items": [1, 2, 3], "fn": "double"}))
    assert r["results"] == [2, 4, 6]


def test_execute_filter(tool):
    r = json.loads(tool.execute("batch_filter", {"items": [0, 1, 2], "fn": "is_truthy"}))
    assert r["kept"] == 2


def test_execute_reduce(tool):
    r = json.loads(tool.execute("batch_reduce", {"items": [1, 2, 3], "fn": "sum"}))
    assert r["result"] == 6


def test_execute_chunk(tool):
    r = json.loads(tool.execute("batch_chunk", {"items": [1, 2, 3, 4], "size": 2}))
    assert r["chunk_count"] == 2


def test_execute_zip(tool):
    r = json.loads(tool.execute("batch_zip", {"keys": ["a", "b"], "lists": [[1, 2], [3, 4]]}))
    assert r["count"] == 2


def test_execute_builtin_fns(tool):
    r = json.loads(tool.execute("builtin_fns", {}))
    assert "fns" in r


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ─────────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "batch"


def test_description(tool):
    assert "batch" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 9


def test_definitions_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
