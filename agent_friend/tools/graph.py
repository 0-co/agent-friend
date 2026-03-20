"""graph.py — GraphTool for agent-friend (stdlib only).

A directed graph tool for agents that need to model dependencies,
workflows, and relationships.  Pure Python — no networkx required.

Features:
* Add nodes and directed edges
* Topological sort (Kahn's algorithm)
* Cycle detection
* Shortest path (BFS)
* Ancestors / descendants of a node
* Reverse graph
* Subgraph extraction
* Node metadata (arbitrary JSON-serialisable dict)
* Multiple named graphs per tool instance

Usage::

    tool = GraphTool()

    tool.graph_create("deps")
    tool.graph_add_node("deps", "a")
    tool.graph_add_node("deps", "b")
    tool.graph_add_node("deps", "c")
    tool.graph_add_edge("deps", "a", "b")  # a → b
    tool.graph_add_edge("deps", "b", "c")  # b → c

    tool.graph_topo_sort("deps")   # ["a", "b", "c"]
    tool.graph_ancestors("deps", "c")  # ["a", "b"]
    tool.graph_has_cycle("deps")   # False
"""

import json
from collections import deque
from typing import Any, Dict, List, Optional, Set

from .base import BaseTool


class _Graph:
    """A single named directed graph."""

    def __init__(self, name: str) -> None:
        self.name = name
        # adjacency: node → set of successors
        self.adj: Dict[str, Set[str]] = {}
        # reverse adjacency: node → set of predecessors
        self.radj: Dict[str, Set[str]] = {}
        # per-node metadata
        self.meta: Dict[str, Dict[str, Any]] = {}

    # ── internal ──────────────────────────────────────────────────────

    def _ensure_node(self, node: str) -> None:
        if node not in self.adj:
            self.adj[node] = set()
            self.radj[node] = set()
            self.meta[node] = {}

    def nodes(self) -> List[str]:
        return sorted(self.adj.keys())

    def edges(self) -> List[Dict[str, str]]:
        result = []
        for src, dsts in self.adj.items():
            for dst in sorted(dsts):
                result.append({"from": src, "to": dst})
        return result

    def add_node(self, node: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_node(node)
        if meta:
            self.meta[node].update(meta)

    def add_edge(self, src: str, dst: str) -> None:
        self._ensure_node(src)
        self._ensure_node(dst)
        self.adj[src].add(dst)
        self.radj[dst].add(src)

    def remove_edge(self, src: str, dst: str) -> bool:
        if src in self.adj and dst in self.adj[src]:
            self.adj[src].discard(dst)
            self.radj[dst].discard(src)
            return True
        return False

    def remove_node(self, node: str) -> bool:
        if node not in self.adj:
            return False
        # Remove all edges involving this node
        for dst in list(self.adj[node]):
            self.radj[dst].discard(node)
        for src in list(self.radj[node]):
            self.adj[src].discard(node)
        del self.adj[node]
        del self.radj[node]
        self.meta.pop(node, None)
        return True

    def has_cycle(self) -> bool:
        """Kahn's algorithm — return True if any cycle exists."""
        in_degree = {n: len(preds) for n, preds in self.radj.items()}
        queue = deque(n for n, d in in_degree.items() if d == 0)
        visited = 0
        while queue:
            node = queue.popleft()
            visited += 1
            for succ in self.adj[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return visited != len(self.adj)

    def topo_sort(self) -> Optional[List[str]]:
        """Return topological order or None if cycle exists."""
        in_degree = {n: len(preds) for n, preds in self.radj.items()}
        queue = deque(sorted(n for n, d in in_degree.items() if d == 0))
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for succ in sorted(self.adj[node]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        if len(result) != len(self.adj):
            return None   # cycle
        return result

    def bfs_path(self, src: str, dst: str) -> Optional[List[str]]:
        """BFS shortest path from src to dst, or None if unreachable."""
        if src not in self.adj or dst not in self.adj:
            return None
        if src == dst:
            return [src]
        visited = {src}
        queue: deque = deque([[src]])
        while queue:
            path = queue.popleft()
            node = path[-1]
            for succ in sorted(self.adj[node]):
                if succ == dst:
                    return path + [succ]
                if succ not in visited:
                    visited.add(succ)
                    queue.append(path + [succ])
        return None

    def ancestors(self, node: str) -> List[str]:
        """All nodes that can reach *node* (BFS on reversed graph)."""
        visited: Set[str] = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for pred in self.radj.get(n, set()):
                if pred not in visited:
                    visited.add(pred)
                    queue.append(pred)
        return sorted(visited)

    def descendants(self, node: str) -> List[str]:
        """All nodes reachable from *node* (BFS)."""
        visited: Set[str] = set()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            for succ in self.adj.get(n, set()):
                if succ not in visited:
                    visited.add(succ)
                    queue.append(succ)
        return sorted(visited)

    def status(self) -> Dict[str, Any]:
        edge_count = sum(len(dsts) for dsts in self.adj.values())
        return {
            "name": self.name,
            "node_count": len(self.adj),
            "edge_count": edge_count,
            "nodes": self.nodes(),
        }


class GraphTool(BaseTool):
    """Directed graphs for dependency tracking and workflow DAGs.

    Add nodes and edges, detect cycles, topologically sort, find paths,
    and compute ancestors/descendants. Multiple named graphs per instance.

    Parameters
    ----------
    max_graphs:
        Maximum named graphs (default 50).
    max_nodes:
        Maximum nodes per graph (default 1_000).
    """

    def __init__(self, max_graphs: int = 50, max_nodes: int = 1_000) -> None:
        self.max_graphs = max_graphs
        self.max_nodes = max_nodes
        self._graphs: Dict[str, _Graph] = {}

    # ── helpers ───────────────────────────────────────────────────────

    def _get(self, name: str) -> _Graph:
        g = self._graphs.get(name)
        if g is None:
            raise KeyError(f"No graph named '{name}'")
        return g

    # ── public API ────────────────────────────────────────────────────

    def graph_create(self, name: str) -> str:
        """Create a named directed graph."""
        if name in self._graphs:
            return json.dumps({"error": f"Graph '{name}' already exists."})
        if len(self._graphs) >= self.max_graphs:
            return json.dumps({"error": f"Max graphs ({self.max_graphs}) reached."})
        self._graphs[name] = _Graph(name)
        return json.dumps({"created": True, "name": name})

    def graph_add_node(self, name: str, node: str, meta: Optional[Dict[str, Any]] = None) -> str:
        """Add a node to the graph. ``meta`` is an optional metadata dict."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if node not in g.adj and len(g.adj) >= self.max_nodes:
            return json.dumps({"error": f"Max nodes ({self.max_nodes}) reached."})
        g.add_node(node, meta or {})
        return json.dumps({"added": True, "node": node})

    def graph_add_edge(self, name: str, src: str, dst: str) -> str:
        """Add a directed edge src → dst. Both nodes are auto-created."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        # auto-create nodes
        for n in (src, dst):
            if n not in g.adj:
                if len(g.adj) >= self.max_nodes:
                    return json.dumps({"error": f"Max nodes ({self.max_nodes}) reached."})
                g.add_node(n)
        g.add_edge(src, dst)
        return json.dumps({"added": True, "from": src, "to": dst})

    def graph_remove_edge(self, name: str, src: str, dst: str) -> str:
        """Remove a directed edge. Returns error if edge does not exist."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if g.remove_edge(src, dst):
            return json.dumps({"removed": True, "from": src, "to": dst})
        return json.dumps({"error": f"No edge from '{src}' to '{dst}'."})

    def graph_remove_node(self, name: str, node: str) -> str:
        """Remove a node and all its edges."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if g.remove_node(node):
            return json.dumps({"removed": True, "node": node})
        return json.dumps({"error": f"Node '{node}' not found."})

    def graph_nodes(self, name: str) -> str:
        """List all node names (sorted)."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(g.nodes())

    def graph_edges(self, name: str) -> str:
        """List all edges as [{from, to}]."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(g.edges())

    def graph_has_cycle(self, name: str) -> str:
        """Return ``{has_cycle: true/false}``."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps({"has_cycle": g.has_cycle()})

    def graph_topo_sort(self, name: str) -> str:
        """Return topological order as a list, or error if graph has a cycle."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        order = g.topo_sort()
        if order is None:
            return json.dumps({"error": "Graph has a cycle; topological sort is not possible."})
        return json.dumps(order)

    def graph_path(self, name: str, src: str, dst: str) -> str:
        """Find the shortest path from src to dst (BFS).

        Returns ``{path: [...], length: N}`` or ``{reachable: false}`` if no path.
        """
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        path = g.bfs_path(src, dst)
        if path is None:
            return json.dumps({"reachable": False, "path": []})
        return json.dumps({"reachable": True, "path": path, "length": len(path) - 1})

    def graph_ancestors(self, name: str, node: str) -> str:
        """Return all nodes that can reach *node* (sorted)."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if node not in g.adj:
            return json.dumps({"error": f"Node '{node}' not found."})
        return json.dumps(g.ancestors(node))

    def graph_descendants(self, name: str, node: str) -> str:
        """Return all nodes reachable from *node* (sorted)."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if node not in g.adj:
            return json.dumps({"error": f"Node '{node}' not found."})
        return json.dumps(g.descendants(node))

    def graph_status(self, name: str) -> str:
        """Return node count, edge count, and node list for a graph."""
        try:
            g = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(g.status())

    def graph_delete(self, name: str) -> str:
        """Delete a graph."""
        if name not in self._graphs:
            return json.dumps({"error": f"No graph named '{name}'"})
        del self._graphs[name]
        return json.dumps({"deleted": True, "name": name})

    def graph_list(self) -> str:
        """List all graphs with their node/edge counts."""
        return json.dumps([g.status() for g in self._graphs.values()])

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "graph"

    @property
    def description(self) -> str:
        return (
            "Directed graphs for dependency tracking and workflow DAGs. "
            "Add nodes and edges, detect cycles, topologically sort, find shortest paths, "
            "compute ancestors and descendants. Multiple named graphs. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "graph_create",
                "description": "Create a named directed graph.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Unique graph name"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_add_node",
                "description": "Add a node with optional metadata dict.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "node": {"type": "string"},
                        "meta": {"type": "object", "description": "Optional metadata"},
                    },
                    "required": ["name", "node"],
                },
            },
            {
                "name": "graph_add_edge",
                "description": "Add a directed edge src → dst. Both nodes are auto-created.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                    },
                    "required": ["name", "src", "dst"],
                },
            },
            {
                "name": "graph_remove_edge",
                "description": "Remove a directed edge.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                    },
                    "required": ["name", "src", "dst"],
                },
            },
            {
                "name": "graph_remove_node",
                "description": "Remove a node and all its edges.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "node": {"type": "string"},
                    },
                    "required": ["name", "node"],
                },
            },
            {
                "name": "graph_nodes",
                "description": "List all node names.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_edges",
                "description": "List all edges as [{from, to}].",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_has_cycle",
                "description": "Return {has_cycle: true/false}.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_topo_sort",
                "description": "Return topological order as a list, or error if the graph has a cycle.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_path",
                "description": "Find shortest path from src to dst. Returns {reachable, path, length}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                    },
                    "required": ["name", "src", "dst"],
                },
            },
            {
                "name": "graph_ancestors",
                "description": "Return all nodes that can reach the given node.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "node": {"type": "string"},
                    },
                    "required": ["name", "node"],
                },
            },
            {
                "name": "graph_descendants",
                "description": "Return all nodes reachable from the given node.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "node": {"type": "string"},
                    },
                    "required": ["name", "node"],
                },
            },
            {
                "name": "graph_status",
                "description": "Return node count, edge count, and node list.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_delete",
                "description": "Delete a graph.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "graph_list",
                "description": "List all graphs with their node/edge counts.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "graph_create":
            return self.graph_create(**arguments)
        if tool_name == "graph_add_node":
            return self.graph_add_node(**arguments)
        if tool_name == "graph_add_edge":
            return self.graph_add_edge(**arguments)
        if tool_name == "graph_remove_edge":
            return self.graph_remove_edge(**arguments)
        if tool_name == "graph_remove_node":
            return self.graph_remove_node(**arguments)
        if tool_name == "graph_nodes":
            return self.graph_nodes(**arguments)
        if tool_name == "graph_edges":
            return self.graph_edges(**arguments)
        if tool_name == "graph_has_cycle":
            return self.graph_has_cycle(**arguments)
        if tool_name == "graph_topo_sort":
            return self.graph_topo_sort(**arguments)
        if tool_name == "graph_path":
            return self.graph_path(**arguments)
        if tool_name == "graph_ancestors":
            return self.graph_ancestors(**arguments)
        if tool_name == "graph_descendants":
            return self.graph_descendants(**arguments)
        if tool_name == "graph_status":
            return self.graph_status(**arguments)
        if tool_name == "graph_delete":
            return self.graph_delete(**arguments)
        if tool_name == "graph_list":
            return self.graph_list()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
