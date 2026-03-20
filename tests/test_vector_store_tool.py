"""Tests for VectorStoreTool — in-memory vector store with cosine similarity."""

import json
import math
import pytest
from agent_friend.tools.vector_store import VectorStoreTool, _cosine_similarity, _euclidean_distance


# ── helpers ────────────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = [1.0, 0.0, 0.0]
    assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(_cosine_similarity(a, b) + 1.0) < 1e-9


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_euclidean_distance_identical():
    v = [1.0, 2.0, 3.0]
    assert _euclidean_distance(v, v) == 0.0


def test_euclidean_distance_known():
    a = [0.0, 0.0]
    b = [3.0, 4.0]
    assert abs(_euclidean_distance(a, b) - 5.0) < 1e-9


# ── fixture ────────────────────────────────────────────────────────────────

@pytest.fixture
def tool():
    return VectorStoreTool()


@pytest.fixture
def store_with_data(tool):
    """Store with 3 2D vectors."""
    tool.vector_add("test", [1.0, 0.0], metadata={"label": "right"}, doc_id="right")
    tool.vector_add("test", [0.0, 1.0], metadata={"label": "up"}, doc_id="up")
    tool.vector_add("test", [-1.0, 0.0], metadata={"label": "left"}, doc_id="left")
    return tool


# ── vector_add ─────────────────────────────────────────────────────────────

def test_add_returns_id(tool):
    r = json.loads(tool.vector_add("store", [1.0, 2.0, 3.0]))
    assert "id" in r
    assert r["dim"] == 3
    assert r["name"] == "store"


def test_add_custom_id(tool):
    r = json.loads(tool.vector_add("s", [1.0], doc_id="my-id"))
    assert r["id"] == "my-id"


def test_add_with_metadata(tool):
    tool.vector_add("s", [1.0, 2.0], metadata={"text": "hello"}, doc_id="v1")
    r = json.loads(tool.vector_get("s", "v1"))
    assert r["metadata"]["text"] == "hello"


def test_add_auto_id(tool):
    r = json.loads(tool.vector_add("s", [1.0, 2.0]))
    assert len(r["id"]) == 36  # UUID4 format


def test_add_creates_store_on_demand(tool):
    tool.vector_add("new_store", [1.0])
    stores = json.loads(tool.vector_list_stores())
    names = [s["name"] for s in stores]
    assert "new_store" in names


def test_add_empty_vector_error(tool):
    r = json.loads(tool.vector_add("s", []))
    assert "error" in r


def test_add_overwrites_same_id(tool):
    tool.vector_add("s", [1.0], doc_id="x")
    tool.vector_add("s", [9.0], doc_id="x")
    r = json.loads(tool.vector_get("s", "x"))
    assert r["vector"] == [9.0]


# ── vector_search / cosine ─────────────────────────────────────────────────

def test_search_cosine_nearest(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [0.9, 0.1]))
    assert results[0]["id"] == "right"
    assert results[0]["score"] > 0.9


def test_search_cosine_up(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [0.1, 0.9]))
    assert results[0]["id"] == "up"


def test_search_top_k(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0], top_k=2))
    assert len(results) == 2


def test_search_returns_score(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0]))
    assert "score" in results[0]
    assert isinstance(results[0]["score"], float)


def test_search_sorted_descending(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0]))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_metadata_returned(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0], top_k=1))
    assert "label" in results[0]["metadata"]


def test_search_missing_store(tool):
    results = json.loads(tool.vector_search("no_store", [1.0, 0.0]))
    assert results == []


def test_search_empty_query_error(tool):
    tool.vector_add("s", [1.0])
    r = json.loads(tool.vector_search("s", []))
    assert "error" in r


def test_search_threshold_filters(store_with_data):
    # Query opposite of "right" — left side
    results = json.loads(store_with_data.vector_search("test", [-1.0, 0.0], threshold=0.5))
    ids = [r["id"] for r in results]
    assert "left" in ids
    assert "right" not in ids


def test_search_dim_mismatch_skipped(tool):
    tool.vector_add("s", [1.0, 2.0], doc_id="v2d")
    tool.vector_add("s", [1.0, 2.0, 3.0], doc_id="v3d")
    # Query with 2D — 3D vector should be skipped
    results = json.loads(tool.vector_search("s", [1.0, 2.0]))
    ids = [r["id"] for r in results]
    assert "v3d" not in ids


# ── vector_search / euclidean ──────────────────────────────────────────────

def test_search_euclidean(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0], metric="euclidean"))
    assert results[0]["id"] == "right"


# ── vector_search / dot ────────────────────────────────────────────────────

def test_search_dot(store_with_data):
    results = json.loads(store_with_data.vector_search("test", [1.0, 0.0], metric="dot"))
    assert results[0]["id"] == "right"


# ── vector_get ─────────────────────────────────────────────────────────────

def test_get_existing(tool):
    tool.vector_add("s", [1.0, 2.0, 3.0], doc_id="v1")
    r = json.loads(tool.vector_get("s", "v1"))
    assert r["id"] == "v1"
    assert r["vector"] == [1.0, 2.0, 3.0]


def test_get_missing_id(tool):
    tool.vector_add("s", [1.0])
    r = json.loads(tool.vector_get("s", "nope"))
    assert "error" in r


def test_get_missing_store(tool):
    r = json.loads(tool.vector_get("no_store", "id"))
    assert "error" in r


# ── vector_delete ──────────────────────────────────────────────────────────

def test_delete_existing(tool):
    tool.vector_add("s", [1.0], doc_id="v1")
    r = json.loads(tool.vector_delete("s", "v1"))
    assert r["deleted"] is True
    # should be gone
    r2 = json.loads(tool.vector_get("s", "v1"))
    assert "error" in r2


def test_delete_missing(tool):
    tool.vector_add("s", [1.0])
    r = json.loads(tool.vector_delete("s", "nope"))
    assert r["deleted"] is False


def test_delete_missing_store(tool):
    r = json.loads(tool.vector_delete("no_store", "id"))
    assert r["deleted"] is False


# ── vector_list ────────────────────────────────────────────────────────────

def test_list_ids(tool):
    tool.vector_add("s", [1.0], doc_id="a")
    tool.vector_add("s", [2.0], doc_id="b")
    ids = json.loads(tool.vector_list("s"))
    assert "a" in ids
    assert "b" in ids


def test_list_empty_store(tool):
    ids = json.loads(tool.vector_list("no_store"))
    assert ids == []


def test_list_pagination(tool):
    for i in range(10):
        tool.vector_add("s", [float(i)], doc_id=str(i))
    page1 = json.loads(tool.vector_list("s", offset=0, limit=5))
    page2 = json.loads(tool.vector_list("s", offset=5, limit=5))
    assert len(page1) == 5
    assert len(page2) == 5
    assert set(page1).isdisjoint(set(page2))


# ── vector_stats ───────────────────────────────────────────────────────────

def test_stats_basic(tool):
    tool.vector_add("s", [1.0, 2.0, 3.0])
    tool.vector_add("s", [4.0, 5.0, 6.0])
    s = json.loads(tool.vector_stats("s"))
    assert s["count"] == 2
    assert s["dim"] == 3


def test_stats_empty_store(tool):
    s = json.loads(tool.vector_stats("no_store"))
    assert s["count"] == 0
    assert s["dim"] is None


# ── vector_drop / vector_list_stores ──────────────────────────────────────

def test_drop_store(tool):
    tool.vector_add("s", [1.0])
    r = json.loads(tool.vector_drop("s"))
    assert r["dropped"] is True
    stores = json.loads(tool.vector_list_stores())
    assert not any(s["name"] == "s" for s in stores)


def test_drop_missing_store(tool):
    r = json.loads(tool.vector_drop("no_store"))
    assert "error" in r


def test_list_stores_multiple(tool):
    tool.vector_add("a", [1.0])
    tool.vector_add("b", [1.0])
    tool.vector_add("b", [2.0])
    stores = json.loads(tool.vector_list_stores())
    by_name = {s["name"]: s for s in stores}
    assert by_name["a"]["count"] == 1
    assert by_name["b"]["count"] == 2


# ── limits ─────────────────────────────────────────────────────────────────

def test_max_stores_enforced():
    t = VectorStoreTool(max_stores=2)
    t.vector_add("s1", [1.0])
    t.vector_add("s2", [1.0])
    r = json.loads(t.vector_add("s3", [1.0]))
    assert "error" in r


def test_max_vectors_enforced():
    t = VectorStoreTool(max_vectors=2)
    t.vector_add("s", [1.0], doc_id="a")
    t.vector_add("s", [2.0], doc_id="b")
    r = json.loads(t.vector_add("s", [3.0], doc_id="c"))
    assert "error" in r


def test_overwrite_does_not_count_against_max():
    t = VectorStoreTool(max_vectors=2)
    t.vector_add("s", [1.0], doc_id="a")
    t.vector_add("s", [2.0], doc_id="b")
    r = json.loads(t.vector_add("s", [99.0], doc_id="a"))  # overwrite
    assert r.get("id") == "a"


# ── execute dispatch ───────────────────────────────────────────────────────

def test_execute_vector_add(tool):
    r = json.loads(tool.execute("vector_add", {"name": "s", "vector": [1.0, 2.0]}))
    assert "id" in r


def test_execute_vector_search(tool):
    tool.vector_add("s", [1.0, 0.0])
    r = json.loads(tool.execute("vector_search", {"name": "s", "query": [1.0, 0.0]}))
    assert isinstance(r, list)


def test_execute_vector_stats(tool):
    tool.vector_add("s", [1.0])
    r = json.loads(tool.execute("vector_stats", {"name": "s"}))
    assert r["count"] == 1


def test_execute_unknown(tool):
    r = json.loads(tool.execute("nope", {}))
    assert "error" in r


# ── tool metadata ──────────────────────────────────────────────────────────

def test_name(tool):
    assert tool.name == "vector_store"


def test_description(tool):
    assert "vector" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definitions_have_required_fields(tool):
    for d in tool.definitions():
        assert "name" in d
        assert "description" in d
        assert "input_schema" in d
