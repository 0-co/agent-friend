"""Tests for MapReduceTool."""

import json
import pytest

from agent_friend.tools.map_reduce import MapReduceTool


@pytest.fixture
def tool():
    return MapReduceTool()


PEOPLE = json.dumps([
    {"name": "Alice", "score": 90, "dept": "eng", "tags": ["python", "ai"]},
    {"name": "Bob", "score": 75, "dept": "mkt", "tags": ["excel"]},
    {"name": "Charlie", "score": 90, "dept": "eng", "tags": ["python", "java"]},
    {"name": "Diana", "score": 55, "dept": "mkt", "tags": []},
])

NUMS = json.dumps([10, 20, 30, 40, 50])


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "map_reduce"


def test_description(tool):
    assert "map" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 9


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "mr_map", "mr_filter", "mr_reduce", "mr_sort",
        "mr_group", "mr_flatten", "mr_zip", "mr_pick", "mr_slice",
    }


# ── mr_map ────────────────────────────────────────────────────────────────────


def test_map_extracts_field(tool):
    result = json.loads(tool.mr_map(PEOPLE, "name"))
    assert result == ["Alice", "Bob", "Charlie", "Diana"]


def test_map_dot_notation(tool):
    data = json.dumps([{"user": {"name": "Alice"}}, {"user": {"name": "Bob"}}])
    result = json.loads(tool.mr_map(data, "user.name"))
    assert result == ["Alice", "Bob"]


def test_map_missing_field_returns_none(tool):
    result = json.loads(tool.mr_map(PEOPLE, "nonexistent"))
    assert all(v is None for v in result)


def test_map_transform_upper(tool):
    result = json.loads(tool.mr_map(PEOPLE, "name", transform="upper"))
    assert result[0] == "ALICE"


def test_map_transform_lower(tool):
    data = json.dumps([{"v": "HELLO"}, {"v": "WORLD"}])
    result = json.loads(tool.mr_map(data, "v", transform="lower"))
    assert result == ["hello", "world"]


def test_map_transform_len(tool):
    result = json.loads(tool.mr_map(PEOPLE, "name", transform="len"))
    assert result[0] == 5  # "Alice"


def test_map_transform_int(tool):
    data = json.dumps([{"v": "42"}, {"v": "7"}])
    result = json.loads(tool.mr_map(data, "v", transform="int"))
    assert result == [42, 7]


def test_map_transform_bool(tool):
    data = json.dumps([{"v": 1}, {"v": 0}, {"v": ""}])
    result = json.loads(tool.mr_map(data, "v", transform="bool"))
    assert result == [True, False, False]


def test_map_invalid_transform(tool):
    result = json.loads(tool.mr_map(PEOPLE, "name", transform="nonexistent"))
    assert "error" in result


def test_map_whole_item(tool):
    data = json.dumps([1, 2, 3])
    result = json.loads(tool.mr_map(data, "."))
    assert result == [1, 2, 3]


def test_map_invalid_json(tool):
    result = json.loads(tool.mr_map("not json", "field"))
    assert "error" in result


def test_map_non_array(tool):
    result = json.loads(tool.mr_map('{"a": 1}', "a"))
    assert "error" in result


# ── mr_filter ─────────────────────────────────────────────────────────────────


def test_filter_eq(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "dept", "eq", "eng"))
    assert len(result) == 2
    assert all(p["dept"] == "eng" for p in result)


def test_filter_ne(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "dept", "ne", "eng"))
    assert len(result) == 2
    assert all(p["dept"] != "eng" for p in result)


def test_filter_gt(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "gt", 75))
    assert len(result) == 2  # Alice, Charlie (90 > 75)


def test_filter_gte(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "gte", 75))
    assert len(result) == 3  # Alice, Bob, Charlie


def test_filter_lt(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "lt", 75))
    assert len(result) == 1  # Diana


def test_filter_lte(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "lte", 75))
    assert len(result) == 2  # Bob, Diana


def test_filter_contains(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "name", "contains", "li"))
    assert len(result) == 2  # Alice, Charlie


def test_filter_startswith(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "name", "startswith", "A"))
    assert len(result) == 1  # Alice


def test_filter_endswith(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "name", "endswith", "e"))
    names = [p["name"] for p in result]
    assert "Alice" in names and "Charlie" in names


def test_filter_exists(tool):
    data = json.dumps([{"x": 1}, {"y": 2}, {"x": None}])
    result = json.loads(tool.mr_filter(data, "x", "exists"))
    assert len(result) == 1  # {"x": 1} only (None fails exists)


def test_filter_unknown_operator(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "name", "bad_op"))
    assert "error" in result


def test_filter_numeric_string_operand(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "gt", "80"))
    assert len(result) == 2


def test_filter_empty_result(tool):
    result = json.loads(tool.mr_filter(PEOPLE, "score", "gt", 100))
    assert result == []


# ── mr_reduce ─────────────────────────────────────────────────────────────────


def test_reduce_count(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, ".", "count"))
    assert result == 4


def test_reduce_sum(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "score", "sum"))
    assert result == 310.0


def test_reduce_avg(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "score", "avg"))
    assert abs(result - 77.5) < 0.01


def test_reduce_min(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "score", "min"))
    assert result == 55.0


def test_reduce_max(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "score", "max"))
    assert result == 90.0


def test_reduce_first(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "name", "first"))
    assert result == "Alice"


def test_reduce_last(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "name", "last"))
    assert result == "Diana"


def test_reduce_join(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "name", "join", separator="|"))
    assert result == "Alice|Bob|Charlie|Diana"


def test_reduce_join_default_separator(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "name", "join"))
    assert ", " in result


def test_reduce_unique(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "dept", "unique"))
    assert set(result) == {"eng", "mkt"}
    assert len(result) == 2


def test_reduce_unique_preserves_order(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "dept", "unique"))
    # first occurrence order: eng, mkt
    assert result[0] == "eng"


def test_reduce_no_numeric_values(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "name", "sum"))
    assert "error" in result


def test_reduce_unknown_operation(tool):
    result = json.loads(tool.mr_reduce(PEOPLE, "score", "product"))
    assert "error" in result


def test_reduce_empty_list(tool):
    result = json.loads(tool.mr_reduce("[]", "score", "first"))
    assert result is None


# ── mr_sort ───────────────────────────────────────────────────────────────────


def test_sort_ascending(tool):
    result = json.loads(tool.mr_sort(PEOPLE, "score"))
    scores = [p["score"] for p in result]
    assert scores == sorted(scores)


def test_sort_descending(tool):
    result = json.loads(tool.mr_sort(PEOPLE, "score", reverse=True))
    scores = [p["score"] for p in result]
    assert scores == sorted(scores, reverse=True)


def test_sort_by_string_field(tool):
    result = json.loads(tool.mr_sort(PEOPLE, "name"))
    names = [p["name"] for p in result]
    assert names == sorted(names)


def test_sort_none_last(tool):
    data = json.dumps([{"v": 3}, {"v": None}, {"v": 1}])
    result = json.loads(tool.mr_sort(data, "v"))
    assert result[-1]["v"] is None


# ── mr_group ──────────────────────────────────────────────────────────────────


def test_group_by_dept(tool):
    result = json.loads(tool.mr_group(PEOPLE, "dept"))
    assert set(result.keys()) == {"eng", "mkt"}
    assert len(result["eng"]) == 2
    assert len(result["mkt"]) == 2


def test_group_values_are_lists(tool):
    result = json.loads(tool.mr_group(PEOPLE, "dept"))
    for v in result.values():
        assert isinstance(v, list)


# ── mr_flatten ────────────────────────────────────────────────────────────────


def test_flatten_basic(tool):
    data = json.dumps([[1, 2], [3, 4], [5]])
    result = json.loads(tool.mr_flatten(data))
    assert result == [1, 2, 3, 4, 5]


def test_flatten_mixed(tool):
    data = json.dumps([[1, 2], 3, [4]])
    result = json.loads(tool.mr_flatten(data))
    assert result == [1, 2, 3, 4]


def test_flatten_empty(tool):
    result = json.loads(tool.mr_flatten("[]"))
    assert result == []


# ── mr_zip ────────────────────────────────────────────────────────────────────


def test_zip_basic(tool):
    left = json.dumps([1, 2, 3])
    right = json.dumps(["a", "b", "c"])
    result = json.loads(tool.mr_zip(left, right))
    assert result == [{"left": 1, "right": "a"}, {"left": 2, "right": "b"}, {"left": 3, "right": "c"}]


def test_zip_truncates_to_shorter(tool):
    left = json.dumps([1, 2, 3])
    right = json.dumps(["a", "b"])
    result = json.loads(tool.mr_zip(left, right))
    assert len(result) == 2


# ── mr_pick ───────────────────────────────────────────────────────────────────


def test_pick_fields(tool):
    result = json.loads(tool.mr_pick(PEOPLE, ["name", "score"]))
    for item in result:
        assert set(item.keys()) == {"name", "score"}


def test_pick_missing_field_excluded(tool):
    result = json.loads(tool.mr_pick(PEOPLE, ["name", "nonexistent"]))
    for item in result:
        assert "nonexistent" not in item
        assert "name" in item


# ── mr_slice ──────────────────────────────────────────────────────────────────


def test_slice_basic(tool):
    result = json.loads(tool.mr_slice(PEOPLE, start=1, end=3))
    assert len(result) == 2
    assert result[0]["name"] == "Bob"


def test_slice_start_only(tool):
    result = json.loads(tool.mr_slice(PEOPLE, start=2))
    assert len(result) == 2


def test_slice_default_full(tool):
    result = json.loads(tool.mr_slice(PEOPLE))
    assert len(result) == 4


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_map(tool):
    result = json.loads(tool.execute("mr_map", {"data": PEOPLE, "field": "name"}))
    assert isinstance(result, list)


def test_execute_filter(tool):
    result = json.loads(tool.execute("mr_filter", {"data": PEOPLE, "field": "score", "operator": "gt", "value": 80}))
    assert isinstance(result, list)


def test_execute_reduce(tool):
    result = json.loads(tool.execute("mr_reduce", {"data": PEOPLE, "field": "score", "operation": "avg"}))
    assert isinstance(result, float)


def test_execute_sort(tool):
    result = json.loads(tool.execute("mr_sort", {"data": PEOPLE, "field": "score"}))
    assert isinstance(result, list)


def test_execute_group(tool):
    result = json.loads(tool.execute("mr_group", {"data": PEOPLE, "field": "dept"}))
    assert isinstance(result, dict)


def test_execute_flatten(tool):
    data = json.dumps([[1, 2], [3]])
    result = json.loads(tool.execute("mr_flatten", {"data": data}))
    assert result == [1, 2, 3]


def test_execute_zip(tool):
    l = json.dumps([1, 2])
    r = json.dumps(["a", "b"])
    result = json.loads(tool.execute("mr_zip", {"left": l, "right": r}))
    assert len(result) == 2


def test_execute_pick(tool):
    result = json.loads(tool.execute("mr_pick", {"data": PEOPLE, "fields": ["name"]}))
    assert all("score" not in item for item in result)


def test_execute_slice(tool):
    result = json.loads(tool.execute("mr_slice", {"data": PEOPLE, "start": 0, "end": 2}))
    assert len(result) == 2


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result


# ── max_items guard ───────────────────────────────────────────────────────────


def test_max_items_guard():
    t = MapReduceTool(max_items=3)
    big = json.dumps(list(range(4)))
    result = json.loads(t.mr_map(big, "."))
    assert "error" in result
