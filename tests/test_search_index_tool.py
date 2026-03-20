"""Tests for SearchIndexTool."""

import json
import pytest

from agent_friend.tools.search_index import SearchIndexTool


@pytest.fixture
def tool():
    return SearchIndexTool()


DOCS = [
    {"id": 1, "title": "Python packaging guide", "body": "learn how to publish python packages to PyPI"},
    {"id": 2, "title": "Agent memory patterns", "body": "persistent memory for AI agents using SQLite"},
    {"id": 3, "title": "Rate limiting API calls", "body": "how to rate limit openai and anthropic API calls"},
    {"id": 4, "title": "LLM evaluation techniques", "body": "unit testing and evaluation for language models"},
    {"id": 5, "title": "Python async programming", "body": "asyncio event loop and coroutines in Python"},
]


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "search_index"


def test_description(tool):
    assert "search" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "index_create", "index_add", "index_search",
        "index_delete_doc", "index_list_docs",
        "index_status", "index_drop", "index_list",
    }


# ── index_create ──────────────────────────────────────────────────────────────


def test_create_basic(tool):
    result = json.loads(tool.index_create("docs"))
    assert result["created"] is True
    assert result["name"] == "docs"


def test_create_with_fields(tool):
    result = json.loads(tool.index_create("docs", fields=["title"]))
    assert result["fields"] == ["title"]


def test_create_duplicate_fails(tool):
    tool.index_create("docs")
    result = json.loads(tool.index_create("docs"))
    assert "error" in result


def test_create_max_indexes():
    t = SearchIndexTool(max_indexes=2)
    t.index_create("a")
    t.index_create("b")
    result = json.loads(t.index_create("c"))
    assert "error" in result


# ── index_add ─────────────────────────────────────────────────────────────────


def test_add_docs(tool):
    result = json.loads(tool.index_add("docs", DOCS))
    assert result["added"] == 5
    assert result["total"] == 5


def test_add_auto_creates_index(tool):
    tool.index_add("auto", DOCS)
    result = json.loads(tool.index_status("auto"))
    assert result["doc_count"] == 5


def test_add_assigns_ids(tool):
    tool.index_add("docs", DOCS)
    docs = json.loads(tool.index_list_docs("docs"))
    assert all("_id" in d for d in docs)


def test_add_non_list_fails(tool):
    result = json.loads(tool.index_add("docs", {"not": "a list"}))
    assert "error" in result


def test_add_max_docs():
    t = SearchIndexTool(max_docs=3)
    t.index_add("docs", DOCS[:2])
    result = json.loads(t.index_add("docs", DOCS[2:]))
    assert "error" in result


# ── index_search ──────────────────────────────────────────────────────────────


def test_search_returns_results(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "python"))
    assert len(results) > 0


def test_search_python_finds_python_docs(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "python"))
    ids = [r["id"] for r in results]
    assert 1 in ids  # Python packaging guide
    assert 5 in ids  # Python async programming


def test_search_returns_score(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "python"))
    assert all("_score" in r for r in results)
    assert all(r["_score"] > 0 for r in results)


def test_search_relevance_order(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "python"))
    scores = [r["_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_top_n(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "python", top_n=1))
    assert len(results) == 1


def test_search_no_results(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "xyzzy"))
    assert results == []


def test_search_empty_query(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", ""))
    assert results == []


def test_search_field_restriction(tool):
    tool.index_add("docs", DOCS)
    # "python" is in title for docs 1 and 5, also in body for 1, 5
    # restrict to "title" field
    results = json.loads(tool.index_search("docs", "python", field="title"))
    assert all("python" in r["title"].lower() for r in results)


def test_search_case_insensitive(tool):
    tool.index_add("docs", DOCS)
    r1 = json.loads(tool.index_search("docs", "Python"))
    r2 = json.loads(tool.index_search("docs", "python"))
    assert len(r1) == len(r2)


def test_search_multi_word(tool):
    tool.index_add("docs", DOCS)
    results = json.loads(tool.index_search("docs", "rate limit api"))
    assert len(results) > 0
    # doc 3 should rank highly
    assert results[0]["id"] == 3


def test_search_unknown_index(tool):
    result = json.loads(tool.index_search("ghost", "anything"))
    assert "error" in result


def test_search_stop_words_ignored(tool):
    tool.index_add("docs", DOCS)
    # "is" and "the" are stop words — shouldn't crash
    results = json.loads(tool.index_search("docs", "the is a"))
    assert results == []


# ── index_delete_doc ──────────────────────────────────────────────────────────


def test_delete_doc(tool):
    tool.index_add("docs", DOCS)
    docs_before = json.loads(tool.index_list_docs("docs"))
    doc_id = docs_before[0]["_id"]
    result = json.loads(tool.index_delete_doc("docs", doc_id))
    assert result["deleted"] is True
    docs_after = json.loads(tool.index_list_docs("docs"))
    assert len(docs_after) == len(docs_before) - 1


def test_delete_doc_not_found(tool):
    tool.index_add("docs", DOCS)
    result = json.loads(tool.index_delete_doc("docs", 99999))
    assert "error" in result


def test_delete_doc_removed_from_search(tool):
    tool.index_add("docs", DOCS)
    # Delete the Python packaging doc (id 1, _id 0)
    result_before = json.loads(tool.index_search("docs", "packaging"))
    assert len(result_before) > 0
    doc_id = result_before[0]["_id"]
    tool.index_delete_doc("docs", doc_id)
    result_after = json.loads(tool.index_search("docs", "packaging"))
    after_ids = [r["_id"] for r in result_after]
    assert doc_id not in after_ids


# ── index_list_docs ───────────────────────────────────────────────────────────


def test_list_docs(tool):
    tool.index_add("docs", DOCS)
    docs = json.loads(tool.index_list_docs("docs"))
    assert len(docs) == 5


def test_list_docs_limit(tool):
    tool.index_add("docs", DOCS)
    docs = json.loads(tool.index_list_docs("docs", limit=2))
    assert len(docs) == 2


def test_list_docs_offset(tool):
    tool.index_add("docs", DOCS)
    docs_all = json.loads(tool.index_list_docs("docs"))
    docs_offset = json.loads(tool.index_list_docs("docs", offset=2))
    assert docs_offset[0]["_id"] == docs_all[2]["_id"]


# ── index_status ──────────────────────────────────────────────────────────────


def test_status(tool):
    tool.index_add("docs", DOCS)
    status = json.loads(tool.index_status("docs"))
    assert status["doc_count"] == 5
    assert status["token_count"] > 0


def test_status_unknown(tool):
    result = json.loads(tool.index_status("ghost"))
    assert "error" in result


# ── index_drop ────────────────────────────────────────────────────────────────


def test_drop(tool):
    tool.index_add("docs", DOCS)
    tool.index_drop("docs")
    result = json.loads(tool.index_status("docs"))
    assert "error" in result


def test_drop_unknown(tool):
    result = json.loads(tool.index_drop("ghost"))
    assert "error" in result


# ── index_list ────────────────────────────────────────────────────────────────


def test_list_empty(tool):
    result = json.loads(tool.index_list())
    assert result == []


def test_list_shows_all(tool):
    tool.index_create("a")
    tool.index_create("b")
    result = json.loads(tool.index_list())
    names = {idx["name"] for idx in result}
    assert names == {"a", "b"}


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_create(tool):
    result = json.loads(tool.execute("index_create", {"name": "docs"}))
    assert result["created"] is True


def test_execute_add(tool):
    tool.execute("index_create", {"name": "docs"})
    result = json.loads(tool.execute("index_add", {"name": "docs", "docs": DOCS}))
    assert result["added"] == 5


def test_execute_search(tool):
    tool.execute("index_add", {"name": "docs", "docs": DOCS})
    result = json.loads(tool.execute("index_search", {"name": "docs", "query": "python"}))
    assert len(result) > 0


def test_execute_status(tool):
    tool.execute("index_add", {"name": "docs", "docs": DOCS})
    result = json.loads(tool.execute("index_status", {"name": "docs"}))
    assert "doc_count" in result


def test_execute_list_docs(tool):
    tool.execute("index_add", {"name": "docs", "docs": DOCS})
    result = json.loads(tool.execute("index_list_docs", {"name": "docs", "limit": 3}))
    assert len(result) == 3


def test_execute_delete_doc(tool):
    tool.execute("index_add", {"name": "docs", "docs": DOCS})
    docs = json.loads(tool.execute("index_list_docs", {"name": "docs"}))
    doc_id = docs[0]["_id"]
    result = json.loads(tool.execute("index_delete_doc", {"name": "docs", "doc_id": doc_id}))
    assert result["deleted"] is True


def test_execute_drop(tool):
    tool.execute("index_create", {"name": "docs"})
    result = json.loads(tool.execute("index_drop", {"name": "docs"}))
    assert result["dropped"] is True


def test_execute_list(tool):
    result = json.loads(tool.execute("index_list", {}))
    assert isinstance(result, list)


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
