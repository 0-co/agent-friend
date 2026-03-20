"""Tests for SamplerTool — random sampling and selection."""

import json
import pytest
from agent_friend.tools.sampler import SamplerTool


@pytest.fixture
def tool():
    return SamplerTool()


ITEMS = list(range(100))


# ── sample_list ────────────────────────────────────────────────────────────

def test_sample_list_basic(tool):
    r = json.loads(tool.sample_list(ITEMS, n=5))
    assert len(r["sample"]) == 5
    assert r["n"] == 5
    assert r["total"] == 100


def test_sample_list_deterministic(tool):
    r1 = json.loads(tool.sample_list(ITEMS, n=5, seed=42))
    r2 = json.loads(tool.sample_list(ITEMS, n=5, seed=42))
    assert r1["sample"] == r2["sample"]


def test_sample_list_different_seeds(tool):
    r1 = json.loads(tool.sample_list(ITEMS, n=5, seed=42))
    r2 = json.loads(tool.sample_list(ITEMS, n=5, seed=99))
    assert r1["sample"] != r2["sample"]


def test_sample_list_no_duplicates_without_replacement(tool):
    r = json.loads(tool.sample_list(ITEMS, n=50, seed=1))
    assert len(set(r["sample"])) == 50


def test_sample_list_with_replacement(tool):
    small = [1, 2, 3]
    r = json.loads(tool.sample_list(small, n=10, seed=42, replacement=True))
    assert len(r["sample"]) == 10


def test_sample_list_n_exceeds_no_replacement_error(tool):
    r = json.loads(tool.sample_list([1, 2, 3], n=5))
    assert "error" in r


def test_sample_list_empty_error(tool):
    r = json.loads(tool.sample_list([], n=3))
    assert "error" in r


def test_sample_list_n_zero_error(tool):
    r = json.loads(tool.sample_list(ITEMS, n=0))
    assert "error" in r


def test_sample_list_all_items(tool):
    items = [1, 2, 3]
    r = json.loads(tool.sample_list(items, n=3, seed=1))
    assert sorted(r["sample"]) == [1, 2, 3]


def test_sample_list_returns_subset(tool):
    r = json.loads(tool.sample_list(ITEMS, n=10, seed=7))
    assert all(x in ITEMS for x in r["sample"])


# ── sample_weighted ────────────────────────────────────────────────────────

def test_weighted_basic(tool):
    items = ["a", "b", "c"]
    weights = [0.8, 0.1, 0.1]
    r = json.loads(tool.sample_weighted(items, weights, n=100, seed=42))
    counts = {x: r["sample"].count(x) for x in "abc"}
    assert counts["a"] > counts["b"]  # a should dominate


def test_weighted_deterministic(tool):
    items = ["x", "y", "z"]
    weights = [1, 2, 3]
    r1 = json.loads(tool.sample_weighted(items, weights, n=5, seed=42))
    r2 = json.loads(tool.sample_weighted(items, weights, n=5, seed=42))
    assert r1["sample"] == r2["sample"]


def test_weighted_normalization(tool):
    items = ["a", "b"]
    weights = [100, 100]  # equal, not summing to 1
    r = json.loads(tool.sample_weighted(items, weights, n=2, seed=1))
    assert abs(r["weights_normalized"][0] - 0.5) < 0.01


def test_weighted_length_mismatch_error(tool):
    r = json.loads(tool.sample_weighted(["a", "b"], [1, 2, 3], n=1))
    assert "error" in r


def test_weighted_zero_weights_error(tool):
    r = json.loads(tool.sample_weighted(["a", "b"], [0, 0]))
    assert "error" in r


def test_weighted_negative_weight_error(tool):
    r = json.loads(tool.sample_weighted(["a", "b"], [-1, 2]))
    assert "error" in r


def test_weighted_single_n(tool):
    r = json.loads(tool.sample_weighted(["a", "b", "c"], [1, 1, 1], n=1, seed=42))
    assert len(r["sample"]) == 1
    assert r["sample"][0] in ["a", "b", "c"]


def test_weighted_without_replacement(tool):
    items = ["a", "b", "c", "d"]
    weights = [1, 1, 1, 1]
    r = json.loads(tool.sample_weighted(items, weights, n=3, seed=42, replacement=False))
    assert len(set(r["sample"])) == 3  # no duplicates


# ── sample_stratified ──────────────────────────────────────────────────────

def test_stratified_basic(tool):
    groups = {"cat": [1, 2, 3, 4, 5], "dog": [6, 7, 8, 9, 10]}
    r = json.loads(tool.sample_stratified(groups, n_per_group=2, seed=42))
    assert len(r["sample"]["cat"]) == 2
    assert len(r["sample"]["dog"]) == 2
    assert r["total"] == 4


def test_stratified_deterministic(tool):
    groups = {"a": list(range(10)), "b": list(range(10, 20))}
    r1 = json.loads(tool.sample_stratified(groups, n_per_group=3, seed=7))
    r2 = json.loads(tool.sample_stratified(groups, n_per_group=3, seed=7))
    assert r1["sample"] == r2["sample"]


def test_stratified_oversample(tool):
    # n_per_group > group size → replacement sampling
    groups = {"small": [1, 2]}
    r = json.loads(tool.sample_stratified(groups, n_per_group=5, seed=1))
    assert len(r["sample"]["small"]) == 5


def test_stratified_empty_group(tool):
    groups = {"a": [], "b": [1, 2, 3]}
    r = json.loads(tool.sample_stratified(groups, n_per_group=2, seed=1))
    assert r["sample"]["a"] == []
    assert len(r["sample"]["b"]) == 2


def test_stratified_empty_groups_error(tool):
    r = json.loads(tool.sample_stratified({}, n_per_group=3))
    assert "error" in r


# ── shuffle ────────────────────────────────────────────────────────────────

def test_shuffle_same_elements(tool):
    items = [1, 2, 3, 4, 5]
    r = json.loads(tool.shuffle(items, seed=1))
    assert sorted(r["items"]) == sorted(items)


def test_shuffle_deterministic(tool):
    items = list(range(20))
    r1 = json.loads(tool.shuffle(items, seed=42))
    r2 = json.loads(tool.shuffle(items, seed=42))
    assert r1["items"] == r2["items"]


def test_shuffle_different_result(tool):
    items = list(range(20))
    r1 = json.loads(tool.shuffle(items, seed=1))
    r2 = json.loads(tool.shuffle(items, seed=2))
    assert r1["items"] != r2["items"]


def test_shuffle_empty(tool):
    r = json.loads(tool.shuffle([]))
    assert r["items"] == []


def test_shuffle_does_not_mutate_original(tool):
    items = [1, 2, 3]
    json.loads(tool.shuffle(items))
    assert items == [1, 2, 3]


# ── random_split ───────────────────────────────────────────────────────────

def test_split_default_80_20(tool):
    items = list(range(100))
    r = json.loads(tool.random_split(items, seed=42))
    assert len(r["splits"]) == 2
    assert r["sizes"][0] == 80
    assert r["sizes"][1] == 20


def test_split_all_items_present(tool):
    items = list(range(50))
    r = json.loads(tool.random_split(items, ratios=[0.6, 0.4], seed=1))
    all_items = r["splits"][0] + r["splits"][1]
    assert sorted(all_items) == sorted(items)


def test_split_three_way(tool):
    items = list(range(90))
    r = json.loads(tool.random_split(items, ratios=[0.6, 0.2, 0.2], seed=1))
    assert len(r["splits"]) == 3
    total = sum(r["sizes"])
    assert total == 90


def test_split_empty_error(tool):
    r = json.loads(tool.random_split([]))
    assert "error" in r


def test_split_deterministic(tool):
    items = list(range(100))
    r1 = json.loads(tool.random_split(items, seed=99))
    r2 = json.loads(tool.random_split(items, seed=99))
    assert r1["splits"] == r2["splits"]


# ── random_choice ──────────────────────────────────────────────────────────

def test_choice_returns_item(tool):
    items = ["a", "b", "c"]
    r = json.loads(tool.random_choice(items, seed=42))
    assert r["choice"] in items
    assert 0 <= r["index"] < len(items)


def test_choice_deterministic(tool):
    items = list(range(50))
    r1 = json.loads(tool.random_choice(items, seed=7))
    r2 = json.loads(tool.random_choice(items, seed=7))
    assert r1["choice"] == r2["choice"]


def test_choice_empty_error(tool):
    r = json.loads(tool.random_choice([]))
    assert "error" in r


# ── random_int ─────────────────────────────────────────────────────────────

def test_random_int_range(tool):
    r = json.loads(tool.random_int(1, 10, n=20, seed=42))
    assert all(1 <= v <= 10 for v in r["values"])
    assert len(r["values"]) == 20


def test_random_int_deterministic(tool):
    r1 = json.loads(tool.random_int(0, 100, n=5, seed=1))
    r2 = json.loads(tool.random_int(0, 100, n=5, seed=1))
    assert r1["values"] == r2["values"]


def test_random_int_invalid_range(tool):
    r = json.loads(tool.random_int(10, 5))
    assert "error" in r


def test_random_int_single(tool):
    r = json.loads(tool.random_int(5, 5, n=3, seed=1))
    assert r["values"] == [5, 5, 5]


# ── random_float ───────────────────────────────────────────────────────────

def test_random_float_range(tool):
    r = json.loads(tool.random_float(0.0, 1.0, n=10, seed=1))
    assert all(0.0 <= v < 1.0 for v in r["values"])
    assert len(r["values"]) == 10


def test_random_float_deterministic(tool):
    r1 = json.loads(tool.random_float(0.0, 10.0, n=5, seed=99))
    r2 = json.loads(tool.random_float(0.0, 10.0, n=5, seed=99))
    assert r1["values"] == r2["values"]


def test_random_float_invalid_range(tool):
    r = json.loads(tool.random_float(5.0, 5.0))
    assert "error" in r


def test_random_float_decimals(tool):
    r = json.loads(tool.random_float(0.0, 1.0, n=5, decimals=2, seed=1))
    for v in r["values"]:
        assert len(str(v).split(".")[-1]) <= 2


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_sample_list(tool):
    r = json.loads(tool.execute("sample_list", {"items": [1, 2, 3, 4, 5], "n": 3}))
    assert len(r["sample"]) == 3


def test_execute_shuffle(tool):
    r = json.loads(tool.execute("shuffle", {"items": [1, 2, 3]}))
    assert sorted(r["items"]) == [1, 2, 3]


def test_execute_random_int(tool):
    r = json.loads(tool.execute("random_int", {"low": 1, "high": 100, "n": 5}))
    assert len(r["values"]) == 5


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "sampler"


def test_description(tool):
    assert "sampl" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_required(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
