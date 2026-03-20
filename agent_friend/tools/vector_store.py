"""vector_store.py — VectorStoreTool for agent-friend (stdlib only).

In-memory vector store with cosine similarity search.
Store embeddings from any LLM API, search for nearest neighbours,
and build simple RAG pipelines — no numpy, no external deps.

Features:
* vector_add — store a vector with metadata
* vector_search — find nearest neighbours by cosine similarity
* vector_get / vector_delete — retrieve or remove by ID
* vector_list — paginated list of all stored vectors
* vector_stats — store statistics
* Multiple named stores per instance

Usage::

    tool = VectorStoreTool()

    # Store embeddings (e.g. from Anthropic / OpenAI embedding API)
    tool.vector_add("docs", [0.1, 0.9, 0.3], metadata={"text": "cats"})
    tool.vector_add("docs", [0.8, 0.1, 0.5], metadata={"text": "dogs"})

    # Find nearest neighbours
    results = tool.vector_search("docs", [0.15, 0.85, 0.25], top_k=2)
    # [{"id": "...", "score": 0.999, "metadata": {"text": "cats"}}, ...]
"""

import json
import math
import uuid
from typing import Any, Dict, List, Optional

from .base import BaseTool


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _euclidean_distance(a: List[float], b: List[float]) -> float:
    """Euclidean distance between two vectors."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class _VectorStore:
    """A single named vector store."""

    def __init__(self) -> None:
        self._entries: Dict[str, Dict[str, Any]] = {}  # id → {vector, metadata}

    def add(self, vector: List[float], metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> str:
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        self._entries[doc_id] = {"vector": vector, "metadata": metadata or {}}
        return doc_id

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._entries.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._entries:
            del self._entries[doc_id]
            return True
        return False

    def search(
        self,
        query: List[float],
        top_k: int = 5,
        metric: str = "cosine",
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for doc_id, entry in self._entries.items():
            v = entry["vector"]
            if len(v) != len(query):
                continue
            if metric == "cosine":
                score = _cosine_similarity(query, v)
            elif metric == "euclidean":
                # Invert distance so higher = closer
                score = 1.0 / (1.0 + _euclidean_distance(query, v))
            elif metric == "dot":
                score = sum(x * y for x, y in zip(query, v))
            else:
                score = _cosine_similarity(query, v)

            if threshold is not None and score < threshold:
                continue
            results.append({"id": doc_id, "score": score, "metadata": entry["metadata"]})

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    def list_ids(self, offset: int = 0, limit: int = 100) -> List[str]:
        ids = list(self._entries.keys())
        return ids[offset: offset + limit]

    def count(self) -> int:
        return len(self._entries)

    def dim(self) -> Optional[int]:
        for e in self._entries.values():
            return len(e["vector"])
        return None


class VectorStoreTool(BaseTool):
    """In-memory vector store with cosine similarity search.

    Store embeddings (float lists), search by cosine/euclidean/dot
    similarity. Multiple named stores. No numpy, no external deps.

    Parameters
    ----------
    max_stores:
        Maximum named stores (default 20).
    max_vectors:
        Maximum vectors per store (default 10_000).
    """

    def __init__(self, max_stores: int = 20, max_vectors: int = 10_000) -> None:
        self.max_stores = max_stores
        self.max_vectors = max_vectors
        self._stores: Dict[str, _VectorStore] = {}

    def _get_or_create(self, name: str) -> _VectorStore:
        if name not in self._stores:
            if len(self._stores) >= self.max_stores:
                raise RuntimeError(f"Max stores ({self.max_stores}) reached.")
            self._stores[name] = _VectorStore()
        return self._stores[name]

    # ── public API ────────────────────────────────────────────────────

    def vector_add(
        self,
        name: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Add a vector to a named store.

        *vector* must be a list of floats.
        *metadata* is an optional dict stored alongside the vector.
        *doc_id* is auto-generated (UUID4) if not provided.

        Returns ``{id, name, dim}``.
        """
        try:
            store = self._get_or_create(name)
        except RuntimeError as exc:
            return json.dumps({"error": str(exc)})

        if store.count() >= self.max_vectors and doc_id not in store._entries:
            return json.dumps({"error": f"Max vectors ({self.max_vectors}) reached."})

        if not isinstance(vector, list) or not vector:
            return json.dumps({"error": "vector must be a non-empty list of floats."})

        doc_id = store.add(vector, metadata, doc_id)
        return json.dumps({"id": doc_id, "name": name, "dim": len(vector)})

    def vector_search(
        self,
        name: str,
        query: List[float],
        top_k: int = 5,
        metric: str = "cosine",
        threshold: Optional[float] = None,
    ) -> str:
        """Find nearest neighbours to *query*.

        *metric*: ``cosine`` (default), ``euclidean``, ``dot``.
        *threshold*: minimum score to include (optional).

        Returns a JSON array sorted by descending score:
        ``[{id, score, metadata}, ...]``.
        """
        store = self._stores.get(name)
        if store is None:
            return json.dumps([])

        if not isinstance(query, list) or not query:
            return json.dumps({"error": "query must be a non-empty list of floats."})

        results = store.search(query, top_k=top_k, metric=metric, threshold=threshold)
        # Round scores to 6 dp for cleaner output
        for r in results:
            r["score"] = round(r["score"], 6)
        return json.dumps(results)

    def vector_get(self, name: str, doc_id: str) -> str:
        """Retrieve a stored vector by ID.

        Returns ``{id, vector, metadata}`` or ``{error}``.
        """
        store = self._stores.get(name)
        if store is None:
            return json.dumps({"error": f"No store named '{name}'"})
        entry = store.get(doc_id)
        if entry is None:
            return json.dumps({"error": f"No vector with id '{doc_id}'"})
        return json.dumps({"id": doc_id, "vector": entry["vector"], "metadata": entry["metadata"]})

    def vector_delete(self, name: str, doc_id: str) -> str:
        """Delete a vector by ID.

        Returns ``{deleted: true/false}``.
        """
        store = self._stores.get(name)
        if store is None:
            return json.dumps({"deleted": False, "id": doc_id})
        removed = store.delete(doc_id)
        return json.dumps({"deleted": removed, "id": doc_id})

    def vector_list(self, name: str, offset: int = 0, limit: int = 100) -> str:
        """List stored vector IDs (paginated).

        Returns a JSON array of IDs.
        """
        store = self._stores.get(name)
        if store is None:
            return json.dumps([])
        return json.dumps(store.list_ids(offset=offset, limit=limit))

    def vector_stats(self, name: str) -> str:
        """Return statistics for a named store.

        Returns ``{name, count, dim, max_vectors}``.
        """
        store = self._stores.get(name)
        if store is None:
            return json.dumps({"name": name, "count": 0, "dim": None})
        return json.dumps({
            "name": name,
            "count": store.count(),
            "dim": store.dim(),
            "max_vectors": self.max_vectors,
        })

    def vector_drop(self, name: str) -> str:
        """Delete an entire named store."""
        if name not in self._stores:
            return json.dumps({"error": f"No store named '{name}'"})
        del self._stores[name]
        return json.dumps({"dropped": True, "name": name})

    def vector_list_stores(self) -> str:
        """List all named stores with vector counts."""
        result = [
            {"name": n, "count": s.count(), "dim": s.dim()}
            for n, s in self._stores.items()
        ]
        return json.dumps(result)

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "vector_store"

    @property
    def description(self) -> str:
        return (
            "In-memory vector store with cosine/euclidean/dot similarity search. "
            "Store embeddings, find nearest neighbours, build RAG pipelines. "
            "Multiple named stores. No numpy. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "vector_add",
                "description": "Add a vector (embedding) to a named store. Returns {id, name, dim}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "vector": {"type": "array", "items": {"type": "number"}, "description": "Embedding as list of floats"},
                        "metadata": {"type": "object", "description": "Arbitrary metadata stored with the vector"},
                        "doc_id": {"type": "string", "description": "Optional ID (UUID4 auto-generated if omitted)"},
                    },
                    "required": ["name", "vector"],
                },
            },
            {
                "name": "vector_search",
                "description": "Find nearest neighbours. metric: cosine|euclidean|dot. Returns [{id, score, metadata}] sorted by score.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "query": {"type": "array", "items": {"type": "number"}},
                        "top_k": {"type": "integer"},
                        "metric": {"type": "string", "description": "cosine | euclidean | dot"},
                        "threshold": {"type": "number", "description": "Minimum score to include"},
                    },
                    "required": ["name", "query"],
                },
            },
            {
                "name": "vector_get",
                "description": "Retrieve a vector and its metadata by ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "doc_id": {"type": "string"},
                    },
                    "required": ["name", "doc_id"],
                },
            },
            {
                "name": "vector_delete",
                "description": "Delete a vector by ID. Returns {deleted: bool}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "doc_id": {"type": "string"},
                    },
                    "required": ["name", "doc_id"],
                },
            },
            {
                "name": "vector_list",
                "description": "List stored vector IDs (paginated). Returns array of IDs.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "offset": {"type": "integer"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "vector_stats",
                "description": "Return {count, dim, max_vectors} for a named store.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "vector_drop",
                "description": "Delete an entire named vector store.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "vector_list_stores",
                "description": "List all named stores with vector counts and dimensions.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "vector_add":
            return self.vector_add(**arguments)
        if tool_name == "vector_search":
            return self.vector_search(**arguments)
        if tool_name == "vector_get":
            return self.vector_get(**arguments)
        if tool_name == "vector_delete":
            return self.vector_delete(**arguments)
        if tool_name == "vector_list":
            return self.vector_list(**arguments)
        if tool_name == "vector_stats":
            return self.vector_stats(**arguments)
        if tool_name == "vector_drop":
            return self.vector_drop(**arguments)
        if tool_name == "vector_list_stores":
            return self.vector_list_stores()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
