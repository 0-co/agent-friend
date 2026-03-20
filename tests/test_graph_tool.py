"""Tests for GraphTool."""

import json
import pytest

from agent_friend.tools.graph import GraphTool


@pytest.fixture
def tool():
    return GraphTool()


@pytest.fixture
def dep_graph(tool):
    """a→b→d, a→c→d, e (isolated)."""
    tool.graph_create("g")
    tool.graph_add_edge("g", "a", "b")
    tool.graph_add_edge("g", "a", "c")
    tool.graph_add_edge("g", "b", "d")
    tool.graph_add_edge("g", "c", "d")
    tool.graph_add_node("g", "e")
    return tool


# ── basic properties ──────────────────────────────────────────────────────────


def test_name(tool):
    assert tool.name == "graph"


def test_description(tool):
    assert "graph" in tool.description.lower()


def test_definitions_count(tool):
    assert len(tool.definitions()) == 15


def test_definition_names(tool):
    names = {d["name"] for d in tool.definitions()}
    assert names == {
        "graph_create", "graph_add_node", "graph_add_edge",
        "graph_remove_edge", "graph_remove_node",
        "graph_nodes", "graph_edges",
        "graph_has_cycle", "graph_topo_sort",
        "graph_path", "graph_ancestors", "graph_descendants",
        "graph_status", "graph_delete", "graph_list",
    }


# ── graph_create ──────────────────────────────────────────────────────────────


def test_create_basic(tool):
    result = json.loads(tool.graph_create("g"))
    assert result["created"] is True
    assert result["name"] == "g"


def test_create_duplicate_fails(tool):
    tool.graph_create("g")
    result = json.loads(tool.graph_create("g"))
    assert "error" in result


def test_create_max_graphs():
    t = GraphTool(max_graphs=2)
    t.graph_create("a")
    t.graph_create("b")
    result = json.loads(t.graph_create("c"))
    assert "error" in result


# ── graph_add_node ────────────────────────────────────────────────────────────


def test_add_node(tool):
    tool.graph_create("g")
    result = json.loads(tool.graph_add_node("g", "x"))
    assert result["added"] is True


def test_add_node_with_meta(tool):
    tool.graph_create("g")
    tool.graph_add_node("g", "x", {"label": "X"})
    nodes = json.loads(tool.graph_nodes("g"))
    assert "x" in nodes


def test_add_node_unknown_graph(tool):
    result = json.loads(tool.graph_add_node("ghost", "x"))
    assert "error" in result


# ── graph_add_edge ────────────────────────────────────────────────────────────


def test_add_edge(tool):
    tool.graph_create("g")
    result = json.loads(tool.graph_add_edge("g", "a", "b"))
    assert result["added"] is True


def test_add_edge_auto_creates_nodes(tool):
    tool.graph_create("g")
    tool.graph_add_edge("g", "x", "y")
    nodes = json.loads(tool.graph_nodes("g"))
    assert "x" in nodes and "y" in nodes


def test_add_edge_unknown_graph(tool):
    result = json.loads(tool.graph_add_edge("ghost", "a", "b"))
    assert "error" in result


# ── graph_remove_edge ─────────────────────────────────────────────────────────


def test_remove_edge(dep_graph):
    result = json.loads(dep_graph.graph_remove_edge("g", "a", "b"))
    assert result["removed"] is True
    edges = json.loads(dep_graph.graph_edges("g"))
    assert not any(e["from"] == "a" and e["to"] == "b" for e in edges)


def test_remove_edge_nonexistent(dep_graph):
    result = json.loads(dep_graph.graph_remove_edge("g", "a", "e"))
    assert "error" in result


# ── graph_remove_node ─────────────────────────────────────────────────────────


def test_remove_node(dep_graph):
    result = json.loads(dep_graph.graph_remove_node("g", "b"))
    assert result["removed"] is True
    nodes = json.loads(dep_graph.graph_nodes("g"))
    assert "b" not in nodes


def test_remove_node_cleans_edges(dep_graph):
    dep_graph.graph_remove_node("g", "b")
    edges = json.loads(dep_graph.graph_edges("g"))
    assert not any(e["from"] == "b" or e["to"] == "b" for e in edges)


def test_remove_node_nonexistent(dep_graph):
    result = json.loads(dep_graph.graph_remove_node("g", "ghost"))
    assert "error" in result


# ── graph_nodes ───────────────────────────────────────────────────────────────


def test_nodes_sorted(dep_graph):
    nodes = json.loads(dep_graph.graph_nodes("g"))
    assert nodes == sorted(nodes)


def test_nodes_all_present(dep_graph):
    nodes = json.loads(dep_graph.graph_nodes("g"))
    assert set(nodes) == {"a", "b", "c", "d", "e"}


# ── graph_edges ───────────────────────────────────────────────────────────────


def test_edges_count(dep_graph):
    edges = json.loads(dep_graph.graph_edges("g"))
    assert len(edges) == 4


def test_edges_structure(dep_graph):
    edges = json.loads(dep_graph.graph_edges("g"))
    for e in edges:
        assert "from" in e and "to" in e


# ── graph_has_cycle ───────────────────────────────────────────────────────────


def test_no_cycle(dep_graph):
    result = json.loads(dep_graph.graph_has_cycle("g"))
    assert result["has_cycle"] is False


def test_has_cycle(tool):
    tool.graph_create("g")
    tool.graph_add_edge("g", "a", "b")
    tool.graph_add_edge("g", "b", "c")
    tool.graph_add_edge("g", "c", "a")  # cycle
    result = json.loads(tool.graph_has_cycle("g"))
    assert result["has_cycle"] is True


def test_self_loop_is_cycle(tool):
    tool.graph_create("g")
    tool.graph_add_edge("g", "a", "a")
    result = json.loads(tool.graph_has_cycle("g"))
    assert result["has_cycle"] is True


# ── graph_topo_sort ───────────────────────────────────────────────────────────


def test_topo_sort_dag(dep_graph):
    order = json.loads(dep_graph.graph_topo_sort("g"))
    assert isinstance(order, list)
    # a must come before b, c; b,c before d
    idx = {n: i for i, n in enumerate(order)}
    assert idx["a"] < idx["b"]
    assert idx["a"] < idx["c"]
    assert idx["b"] < idx["d"]
    assert idx["c"] < idx["d"]


def test_topo_sort_cycle_returns_error(tool):
    tool.graph_create("g")
    tool.graph_add_edge("g", "a", "b")
    tool.graph_add_edge("g", "b", "a")
    result = json.loads(tool.graph_topo_sort("g"))
    assert "error" in result


def test_topo_sort_single_node(tool):
    tool.graph_create("g")
    tool.graph_add_node("g", "x")
    result = json.loads(tool.graph_topo_sort("g"))
    assert result == ["x"]


# ── graph_path ────────────────────────────────────────────────────────────────


def test_path_exists(dep_graph):
    result = json.loads(dep_graph.graph_path("g", "a", "d"))
    assert result["reachable"] is True
    assert result["path"][0] == "a"
    assert result["path"][-1] == "d"


def test_path_length(dep_graph):
    result = json.loads(dep_graph.graph_path("g", "a", "d"))
    assert result["length"] == len(result["path"]) - 1


def test_path_unreachable(dep_graph):
    result = json.loads(dep_graph.graph_path("g", "d", "a"))
    assert result["reachable"] is False
    assert result["path"] == []


def test_path_same_node(dep_graph):
    result = json.loads(dep_graph.graph_path("g", "a", "a"))
    assert result["reachable"] is True
    assert result["path"] == ["a"]


def test_path_to_isolated_node(dep_graph):
    result = json.loads(dep_graph.graph_path("g", "a", "e"))
    assert result["reachable"] is False


# ── graph_ancestors ───────────────────────────────────────────────────────────


def test_ancestors_d(dep_graph):
    result = json.loads(dep_graph.graph_ancestors("g", "d"))
    assert set(result) == {"a", "b", "c"}


def test_ancestors_a_is_empty(dep_graph):
    result = json.loads(dep_graph.graph_ancestors("g", "a"))
    assert result == []


def test_ancestors_unknown_node(dep_graph):
    result = json.loads(dep_graph.graph_ancestors("g", "ghost"))
    assert "error" in result


# ── graph_descendants ─────────────────────────────────────────────────────────


def test_descendants_a(dep_graph):
    result = json.loads(dep_graph.graph_descendants("g", "a"))
    assert set(result) == {"b", "c", "d"}


def test_descendants_d_is_empty(dep_graph):
    result = json.loads(dep_graph.graph_descendants("g", "d"))
    assert result == []


# ── graph_status ──────────────────────────────────────────────────────────────


def test_status(dep_graph):
    status = json.loads(dep_graph.graph_status("g"))
    assert status["node_count"] == 5
    assert status["edge_count"] == 4


# ── graph_delete ──────────────────────────────────────────────────────────────


def test_delete(tool):
    tool.graph_create("g")
    tool.graph_delete("g")
    result = json.loads(tool.graph_status("g"))
    assert "error" in result


def test_delete_unknown(tool):
    result = json.loads(tool.graph_delete("ghost"))
    assert "error" in result


# ── graph_list ────────────────────────────────────────────────────────────────


def test_list_empty(tool):
    result = json.loads(tool.graph_list())
    assert result == []


def test_list_shows_all(tool):
    tool.graph_create("a")
    tool.graph_create("b")
    result = json.loads(tool.graph_list())
    names = {g["name"] for g in result}
    assert names == {"a", "b"}


# ── execute dispatch ──────────────────────────────────────────────────────────


def test_execute_create(tool):
    result = json.loads(tool.execute("graph_create", {"name": "g"}))
    assert result["created"] is True


def test_execute_add_node(tool):
    tool.execute("graph_create", {"name": "g"})
    result = json.loads(tool.execute("graph_add_node", {"name": "g", "node": "x"}))
    assert result["added"] is True


def test_execute_add_edge(tool):
    tool.execute("graph_create", {"name": "g"})
    result = json.loads(tool.execute("graph_add_edge", {"name": "g", "src": "a", "dst": "b"}))
    assert result["added"] is True


def test_execute_topo_sort(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "a", "dst": "b"})
    result = json.loads(tool.execute("graph_topo_sort", {"name": "g"}))
    assert result == ["a", "b"]


def test_execute_has_cycle(tool):
    tool.execute("graph_create", {"name": "g"})
    result = json.loads(tool.execute("graph_has_cycle", {"name": "g"}))
    assert result["has_cycle"] is False


def test_execute_path(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "x", "dst": "y"})
    result = json.loads(tool.execute("graph_path", {"name": "g", "src": "x", "dst": "y"}))
    assert result["reachable"] is True


def test_execute_ancestors(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "p", "dst": "c"})
    result = json.loads(tool.execute("graph_ancestors", {"name": "g", "node": "c"}))
    assert result == ["p"]


def test_execute_descendants(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "p", "dst": "c"})
    result = json.loads(tool.execute("graph_descendants", {"name": "g", "node": "p"}))
    assert result == ["c"]


def test_execute_nodes(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_node", {"name": "g", "node": "x"})
    result = json.loads(tool.execute("graph_nodes", {"name": "g"}))
    assert "x" in result


def test_execute_edges(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "a", "dst": "b"})
    result = json.loads(tool.execute("graph_edges", {"name": "g"}))
    assert len(result) == 1


def test_execute_status(tool):
    tool.execute("graph_create", {"name": "g"})
    result = json.loads(tool.execute("graph_status", {"name": "g"}))
    assert "node_count" in result


def test_execute_remove_edge(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_edge", {"name": "g", "src": "a", "dst": "b"})
    result = json.loads(tool.execute("graph_remove_edge", {"name": "g", "src": "a", "dst": "b"}))
    assert result["removed"] is True


def test_execute_remove_node(tool):
    tool.execute("graph_create", {"name": "g"})
    tool.execute("graph_add_node", {"name": "g", "node": "x"})
    result = json.loads(tool.execute("graph_remove_node", {"name": "g", "node": "x"}))
    assert result["removed"] is True


def test_execute_delete(tool):
    tool.execute("graph_create", {"name": "g"})
    result = json.loads(tool.execute("graph_delete", {"name": "g"}))
    assert result["deleted"] is True


def test_execute_list(tool):
    result = json.loads(tool.execute("graph_list", {}))
    assert isinstance(result, list)


def test_execute_unknown(tool):
    result = json.loads(tool.execute("no_such", {}))
    assert "error" in result
