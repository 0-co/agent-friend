"""search_index.py — SearchIndexTool for agent-friend (stdlib only).

In-memory full-text search over JSON document collections.  Agents
frequently receive batches of structured data (API responses, scraped
pages, table rows) and need to find relevant items quickly without
spinning up a database.

Features:
* Index documents (list of dicts) by name
* Full-text search with TF-IDF-style scoring
* Field-specific keyword search
* BM25-lite relevance ranking (within stdlib)
* Multiple named indexes per tool instance

Usage::

    tool = SearchIndexTool()

    tool.index_add("docs", [
        {"id": 1, "title": "Python packaging guide", "body": "..."},
        {"id": 2, "title": "Agent memory patterns", "body": "..."},
    ])

    tool.index_search("docs", "python packaging")
    # Returns top matching documents with scores
"""

import json
import math
import re
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import BaseTool


_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with",
})

_PUNCT = str.maketrans("", "", string.punctuation)


def _tokenize(text: str) -> List[str]:
    text = text.lower().translate(_PUNCT)
    tokens = text.split()
    return [t for t in tokens if t and t not in _STOP_WORDS and len(t) > 1]


class _Index:
    """A single named document index."""

    def __init__(self, name: str, fields: Optional[List[str]] = None) -> None:
        self.name = name
        self.fields = fields  # None means all string fields
        self.docs: List[Dict[str, Any]] = []
        # inverted index: token → [(doc_id, field, count)]
        self.inv: Dict[str, List[tuple]] = defaultdict(list)
        self._next_id = 0

    def _text_for(self, doc: Dict[str, Any]) -> str:
        """Concatenate all indexed string fields of a doc."""
        parts = []
        for k, v in doc.items():
            if self.fields is None or k in self.fields:
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, (int, float)):
                    parts.append(str(v))
        return " ".join(parts)

    def add(self, docs: List[Dict[str, Any]]) -> int:
        added = 0
        for doc in docs:
            doc_id = self._next_id
            self._next_id += 1
            self.docs.append({"_id": doc_id, **doc})
            text = self._text_for(doc)
            token_counts: Dict[str, int] = defaultdict(int)
            for t in _tokenize(text):
                token_counts[t] += 1
            for token, count in token_counts.items():
                self.inv[token].append((doc_id, count))
            added += 1
        return added

    def search(self, query: str, top_n: int = 10, field: Optional[str] = None) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        if not tokens:
            return []

        n = len(self.docs)
        if n == 0:
            return []

        # BM25-lite: score = sum_t( idf(t) * tf(t,d) * (k1+1) / (tf(t,d) + k1*(1-b+b*dl/avgdl)) )
        k1 = 1.5
        b = 0.75
        avg_dl = sum(len(_tokenize(self._text_for(d))) for d in self.docs) / n if n else 1

        scores: Dict[int, float] = defaultdict(float)

        for token in tokens:
            posting = self.inv.get(token, [])
            # Filter by field if specified
            if field is not None:
                # Re-search within the specific field
                filtered = []
                for doc_id, count in posting:
                    doc = self.docs[doc_id]
                    fv = str(doc.get(field, ""))
                    field_count = sum(1 for t in _tokenize(fv) if t == token)
                    if field_count > 0:
                        filtered.append((doc_id, field_count))
                posting = filtered

            df = len(posting)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)

            for doc_id, tf in posting:
                doc = self.docs[doc_id]
                dl = len(_tokenize(self._text_for(doc)))
                denom = tf + k1 * (1 - b + b * dl / avg_dl)
                score = idf * tf * (k1 + 1) / denom if denom > 0 else 0
                scores[doc_id] += score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
        results = []
        for doc_id, score in ranked:
            item = dict(self.docs[doc_id])
            item["_score"] = round(score, 4)
            results.append(item)
        return results

    def delete(self, doc_id: int) -> bool:
        for i, doc in enumerate(self.docs):
            if doc["_id"] == doc_id:
                # Remove from docs list
                self.docs.pop(i)
                # Remove from inverted index
                for token in list(self.inv.keys()):
                    self.inv[token] = [(d, c) for d, c in self.inv[token] if d != doc_id]
                    if not self.inv[token]:
                        del self.inv[token]
                return True
        return False

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "doc_count": len(self.docs),
            "token_count": len(self.inv),
            "fields": self.fields,
        }


class SearchIndexTool(BaseTool):
    """In-memory full-text search over JSON document collections.

    Index batches of dicts and search them with BM25-lite relevance ranking.
    Pairs with HTTPTool and HTMLTool: fetch data, index it, search it — all
    without a database or external search service.

    Parameters
    ----------
    max_indexes:
        Maximum named indexes (default 20).
    max_docs:
        Maximum documents per index (default 10_000).
    """

    def __init__(self, max_indexes: int = 20, max_docs: int = 10_000) -> None:
        self.max_indexes = max_indexes
        self.max_docs = max_docs
        self._indexes: Dict[str, _Index] = {}

    def _get(self, name: str) -> _Index:
        idx = self._indexes.get(name)
        if idx is None:
            raise KeyError(f"No index named '{name}'")
        return idx

    # ── public API ────────────────────────────────────────────────────

    def index_create(
        self,
        name: str,
        fields: Optional[List[str]] = None,
    ) -> str:
        """Create a named document index.

        *fields* restricts which keys are indexed for search. If omitted,
        all string and numeric fields are indexed.

        Returns ``{created: true, name: "..."}``
        """
        if name in self._indexes:
            return json.dumps({"error": f"Index '{name}' already exists."})
        if len(self._indexes) >= self.max_indexes:
            return json.dumps({"error": f"Max indexes ({self.max_indexes}) reached."})
        self._indexes[name] = _Index(name, fields)
        return json.dumps({"created": True, "name": name, "fields": fields})

    def index_add(self, name: str, docs: List[Dict[str, Any]]) -> str:
        """Add documents to an index.

        *docs* is a list of dicts. If the index does not exist it is
        created automatically (indexing all fields).

        Returns ``{added: N, total: M}``
        """
        if name not in self._indexes:
            self._indexes[name] = _Index(name)
        try:
            idx = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        if not isinstance(docs, list):
            return json.dumps({"error": "docs must be a list of dicts"})
        if len(idx.docs) + len(docs) > self.max_docs:
            return json.dumps({"error": f"Max docs ({self.max_docs}) would be exceeded."})

        added = idx.add(docs)
        return json.dumps({"added": added, "total": len(idx.docs)})

    def index_search(
        self,
        name: str,
        query: str,
        top_n: int = 10,
        field: Optional[str] = None,
    ) -> str:
        """Search an index for documents matching *query*.

        BM25-lite relevance ranking. *field* restricts matching to a
        single field.

        Returns a JSON array of matching documents, each with a
        ``_score`` field (higher = more relevant) and ``_id``.
        """
        try:
            idx = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})

        results = idx.search(query, top_n=top_n, field=field)
        return json.dumps(results)

    def index_delete_doc(self, name: str, doc_id: int) -> str:
        """Remove a document by its ``_id``."""
        try:
            idx = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        if idx.delete(doc_id):
            return json.dumps({"deleted": True, "doc_id": doc_id})
        return json.dumps({"error": f"No document with _id={doc_id}"})

    def index_list_docs(self, name: str, limit: int = 20, offset: int = 0) -> str:
        """List documents in an index (paginated).

        Returns a JSON array of documents.
        """
        try:
            idx = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(idx.docs[offset: offset + limit])

    def index_status(self, name: str) -> str:
        """Return document count, token count, and indexed fields."""
        try:
            idx = self._get(name)
        except KeyError as exc:
            return json.dumps({"error": str(exc)})
        return json.dumps(idx.status())

    def index_drop(self, name: str) -> str:
        """Delete an index entirely."""
        if name not in self._indexes:
            return json.dumps({"error": f"No index named '{name}'"})
        del self._indexes[name]
        return json.dumps({"dropped": True, "name": name})

    def index_list(self) -> str:
        """List all indexes with their document counts."""
        return json.dumps([idx.status() for idx in self._indexes.values()])

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "search_index"

    @property
    def description(self) -> str:
        return (
            "In-memory full-text search over JSON document collections. "
            "BM25-lite relevance ranking. Index any list of dicts, search "
            "with keyword queries, filter by field. No external search "
            "service required. Pairs with HTTPTool, HTMLTool, TableTool. "
            "Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "index_create",
                "description": "Create a named document index. fields limits which keys are indexed.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields to index (default: all)",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "index_add",
                "description": (
                    "Add documents to an index. docs is a JSON array of dicts. "
                    "Auto-creates index if it doesn't exist."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "docs": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "List of document dicts to index",
                        },
                    },
                    "required": ["name", "docs"],
                },
            },
            {
                "name": "index_search",
                "description": (
                    "Search an index with a keyword query. Returns top matching "
                    "documents with _score (BM25). field restricts to one field."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "query": {"type": "string"},
                        "top_n": {"type": "integer", "description": "Max results (default 10)"},
                        "field": {"type": "string", "description": "Restrict search to this field"},
                    },
                    "required": ["name", "query"],
                },
            },
            {
                "name": "index_delete_doc",
                "description": "Remove a document by its _id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "doc_id": {"type": "integer"},
                    },
                    "required": ["name", "doc_id"],
                },
            },
            {
                "name": "index_list_docs",
                "description": "List documents in an index (paginated).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "index_status",
                "description": "Return doc count, token count, and indexed fields.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "index_drop",
                "description": "Delete an index entirely.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            {
                "name": "index_list",
                "description": "List all indexes with their document counts.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "index_create":
            return self.index_create(**arguments)
        if tool_name == "index_add":
            return self.index_add(**arguments)
        if tool_name == "index_search":
            return self.index_search(**arguments)
        if tool_name == "index_delete_doc":
            return self.index_delete_doc(**arguments)
        if tool_name == "index_list_docs":
            return self.index_list_docs(**arguments)
        if tool_name == "index_status":
            return self.index_status(**arguments)
        if tool_name == "index_drop":
            return self.index_drop(**arguments)
        if tool_name == "index_list":
            return self.index_list()
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
