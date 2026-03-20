"""Tests for QueueTool."""

import json
import pytest

from agent_friend.tools.queue_tool import QueueTool


@pytest.fixture
def tool():
    return QueueTool()


# ── basic properties ───────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "queue"


def test_description(tool):
    desc = tool.description.lower()
    assert "queue" in desc


def test_definitions_count(tool):
    assert len(tool.definitions()) == 8


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "queue_create", "queue_push", "queue_pop", "queue_peek",
        "queue_size", "queue_clear", "queue_delete", "queue_list",
    }


# ── queue_create ──────────────────────────────────────────────────────────────


def test_create_fifo(tool):
    result = json.loads(tool.queue_create("tasks", kind="fifo"))
    assert result["created"] is True
    assert result["kind"] == "fifo"


def test_create_lifo(tool):
    result = json.loads(tool.queue_create("stack", kind="lifo"))
    assert result["created"] is True
    assert result["kind"] == "lifo"


def test_create_priority(tool):
    result = json.loads(tool.queue_create("alerts", kind="priority"))
    assert result["created"] is True
    assert result["kind"] == "priority"


def test_create_default_fifo(tool):
    result = json.loads(tool.queue_create("q"))
    assert result["kind"] == "fifo"


def test_create_duplicate_fails(tool):
    tool.queue_create("dup")
    result = json.loads(tool.queue_create("dup"))
    assert "error" in result


def test_create_unknown_kind(tool):
    result = json.loads(tool.queue_create("bad", kind="circular"))
    assert "error" in result


def test_create_max_queues():
    t = QueueTool(max_queues=2)
    t.queue_create("a")
    t.queue_create("b")
    result = json.loads(t.queue_create("c"))
    assert "error" in result


# ── FIFO behaviour ────────────────────────────────────────────────────────────


def test_fifo_order(tool):
    tool.queue_create("q")
    tool.queue_push("q", "first")
    tool.queue_push("q", "second")
    tool.queue_push("q", "third")
    assert json.loads(tool.queue_pop("q"))["item"] == "first"
    assert json.loads(tool.queue_pop("q"))["item"] == "second"
    assert json.loads(tool.queue_pop("q"))["item"] == "third"


def test_fifo_size_updates(tool):
    tool.queue_create("q2")
    tool.queue_push("q2", 1)
    tool.queue_push("q2", 2)
    assert json.loads(tool.queue_size("q2"))["size"] == 2
    tool.queue_pop("q2")
    assert json.loads(tool.queue_size("q2"))["size"] == 1


def test_fifo_maxsize(tool):
    tool.queue_create("q3", maxsize=2)
    tool.queue_push("q3", "a")
    tool.queue_push("q3", "b")
    result = json.loads(tool.queue_push("q3", "c"))
    assert result["pushed"] is False


def test_fifo_complex_items(tool):
    tool.queue_create("q4")
    item = {"url": "https://example.com", "retry": 0, "tags": ["a", "b"]}
    tool.queue_push("q4", item)
    result = json.loads(tool.queue_pop("q4"))
    assert result["item"]["url"] == "https://example.com"
    assert result["item"]["tags"] == ["a", "b"]


# ── LIFO behaviour ────────────────────────────────────────────────────────────


def test_lifo_order(tool):
    tool.queue_create("stack", kind="lifo")
    tool.queue_push("stack", "first")
    tool.queue_push("stack", "second")
    tool.queue_push("stack", "third")
    assert json.loads(tool.queue_pop("stack"))["item"] == "third"
    assert json.loads(tool.queue_pop("stack"))["item"] == "second"
    assert json.loads(tool.queue_pop("stack"))["item"] == "first"


def test_lifo_peek(tool):
    tool.queue_create("s2", kind="lifo")
    tool.queue_push("s2", "bottom")
    tool.queue_push("s2", "top")
    result = json.loads(tool.queue_peek("s2"))
    assert result["item"] == "top"
    # Size unchanged
    assert json.loads(tool.queue_size("s2"))["size"] == 2


# ── Priority queue behaviour ───────────────────────────────────────────────────


def test_priority_order(tool):
    tool.queue_create("pq", kind="priority")
    tool.queue_push("pq", "medium", priority=5)
    tool.queue_push("pq", "urgent", priority=1)
    tool.queue_push("pq", "low", priority=10)
    assert json.loads(tool.queue_pop("pq"))["item"] == "urgent"
    assert json.loads(tool.queue_pop("pq"))["item"] == "medium"
    assert json.loads(tool.queue_pop("pq"))["item"] == "low"


def test_priority_tiebreak_fifo(tool):
    """Same priority → insertion order (FIFO via sequence counter)."""
    tool.queue_create("pq2", kind="priority")
    tool.queue_push("pq2", "first", priority=1)
    tool.queue_push("pq2", "second", priority=1)
    assert json.loads(tool.queue_pop("pq2"))["item"] == "first"


def test_priority_default_priority(tool):
    tool.queue_create("pq3", kind="priority")
    tool.queue_push("pq3", "no-priority")
    result = json.loads(tool.queue_pop("pq3"))
    assert result["item"] == "no-priority"


def test_priority_maxsize(tool):
    tool.queue_create("pq4", kind="priority", maxsize=2)
    tool.queue_push("pq4", "a", priority=1)
    tool.queue_push("pq4", "b", priority=2)
    result = json.loads(tool.queue_push("pq4", "c", priority=0))
    assert result["pushed"] is False


# ── queue_pop empty ────────────────────────────────────────────────────────────


def test_pop_empty_fifo(tool):
    tool.queue_create("empty")
    result = json.loads(tool.queue_pop("empty"))
    assert result["empty"] is True
    assert result["item"] is None


def test_pop_empty_priority(tool):
    tool.queue_create("epq", kind="priority")
    result = json.loads(tool.queue_pop("epq"))
    assert result["empty"] is True


def test_pop_unknown_queue(tool):
    result = json.loads(tool.queue_pop("ghost"))
    assert "error" in result


# ── queue_peek ─────────────────────────────────────────────────────────────────


def test_peek_fifo(tool):
    tool.queue_create("peek")
    tool.queue_push("peek", "first")
    tool.queue_push("peek", "second")
    result = json.loads(tool.queue_peek("peek"))
    assert result["item"] == "first"
    assert json.loads(tool.queue_size("peek"))["size"] == 2


def test_peek_empty(tool):
    tool.queue_create("pe2")
    result = json.loads(tool.queue_peek("pe2"))
    assert result["empty"] is True


# ── queue_clear ────────────────────────────────────────────────────────────────


def test_clear_removes_all(tool):
    tool.queue_create("cl")
    tool.queue_push("cl", "a")
    tool.queue_push("cl", "b")
    tool.queue_clear("cl")
    assert json.loads(tool.queue_size("cl"))["size"] == 0


def test_clear_unknown(tool):
    result = json.loads(tool.queue_clear("nobody"))
    assert "error" in result


# ── queue_delete ───────────────────────────────────────────────────────────────


def test_delete_removes_queue(tool):
    tool.queue_create("del")
    tool.queue_delete("del")
    result = json.loads(tool.queue_size("del"))
    assert "error" in result


def test_delete_unknown(tool):
    result = json.loads(tool.queue_delete("ghost"))
    assert "error" in result


# ── queue_list ─────────────────────────────────────────────────────────────────


def test_list_empty(tool):
    result = json.loads(tool.queue_list())
    assert result == []


def test_list_shows_all(tool):
    tool.queue_create("a")
    tool.queue_create("b", kind="lifo")
    result = json.loads(tool.queue_list())
    names = {r["name"] for r in result}
    assert names == {"a", "b"}
    kinds = {r["name"]: r["kind"] for r in result}
    assert kinds["b"] == "lifo"


def test_list_shows_size(tool):
    tool.queue_create("sized")
    tool.queue_push("sized", 1)
    tool.queue_push("sized", 2)
    result = json.loads(tool.queue_list())
    entry = next(r for r in result if r["name"] == "sized")
    assert entry["size"] == 2


# ── execute dispatch ───────────────────────────────────────────────────────────


def test_execute_create(tool):
    result = json.loads(tool.execute("queue_create", {"name": "ex1"}))
    assert result["created"] is True


def test_execute_push(tool):
    tool.execute("queue_create", {"name": "ex2"})
    result = json.loads(tool.execute("queue_push", {"name": "ex2", "item": "hello"}))
    assert result["pushed"] is True


def test_execute_pop(tool):
    tool.execute("queue_create", {"name": "ex3"})
    tool.execute("queue_push", {"name": "ex3", "item": 42})
    result = json.loads(tool.execute("queue_pop", {"name": "ex3"}))
    assert result["item"] == 42


def test_execute_peek(tool):
    tool.execute("queue_create", {"name": "ex4"})
    tool.execute("queue_push", {"name": "ex4", "item": "peek"})
    result = json.loads(tool.execute("queue_peek", {"name": "ex4"}))
    assert result["item"] == "peek"


def test_execute_size(tool):
    tool.execute("queue_create", {"name": "ex5"})
    result = json.loads(tool.execute("queue_size", {"name": "ex5"}))
    assert result["size"] == 0


def test_execute_clear(tool):
    tool.execute("queue_create", {"name": "ex6"})
    tool.execute("queue_push", {"name": "ex6", "item": "x"})
    result = json.loads(tool.execute("queue_clear", {"name": "ex6"}))
    assert result["cleared"] is True


def test_execute_delete(tool):
    tool.execute("queue_create", {"name": "ex7"})
    result = json.loads(tool.execute("queue_delete", {"name": "ex7"}))
    assert result["deleted"] is True


def test_execute_list(tool):
    result = json.loads(tool.execute("queue_list", {}))
    assert isinstance(result, list)


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
