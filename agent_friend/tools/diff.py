"""diff.py — DiffTool for agent-friend (stdlib only).

Code review agents that only read files don't see what changed. DiffTool
gives your agent unified diffs, word-level comparisons, and similarity
metrics — all using Python's stdlib difflib. Zero dependencies.

Usage::

    tool = DiffTool()
    tool.diff_text("hello world", "hello there")
    # {"unified": "--- a\\n+++ b\\n@@ -1 +1 @@\\n-hello world\\n+hello there\\n", ...}

    tool.diff_files("/path/to/old.py", "/path/to/new.py")
    # {"unified": "...", "stats": {...}}

    tool.diff_stats("apple", "applesauce")
    # {"added_chars": 5, "removed_chars": 0, "similarity": 0.69, ...}
"""

import difflib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseTool


class DiffTool(BaseTool):
    """Text and file diffing using Python's stdlib difflib.

    Produces unified diffs, word-level diffs, and similarity statistics.
    All operations are stdlib-only — zero dependencies.
    """

    @property
    def name(self) -> str:
        return "diff"

    @property
    def description(self) -> str:
        return (
            "Compare text strings or files and generate unified diffs, "
            "word-level differences, and similarity statistics. "
            "Useful for code review, change detection, and content comparison."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "diff_text",
                "description": (
                    "Compare two text strings and return a unified diff. "
                    "Returns {unified: str, added_lines: int, removed_lines: int, "
                    "unchanged_lines: int, has_changes: bool}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text_a": {"type": "string", "description": "Original text (before)."},
                        "text_b": {"type": "string", "description": "Modified text (after)."},
                        "context": {
                            "type": "integer",
                            "description": "Lines of context around changes (default 3).",
                        },
                        "label_a": {"type": "string", "description": "Label for original (default 'before')."},
                        "label_b": {"type": "string", "description": "Label for modified (default 'after')."},
                    },
                    "required": ["text_a", "text_b"],
                },
            },
            {
                "name": "diff_files",
                "description": (
                    "Compare two files and return a unified diff. "
                    "Returns {unified: str, added_lines: int, removed_lines: int, "
                    "unchanged_lines: int, has_changes: bool}. "
                    "Returns an error if either file cannot be read."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path_a": {"type": "string", "description": "Path to original file."},
                        "path_b": {"type": "string", "description": "Path to modified file."},
                        "context": {
                            "type": "integer",
                            "description": "Lines of context around changes (default 3).",
                        },
                    },
                    "required": ["path_a", "path_b"],
                },
            },
            {
                "name": "diff_words",
                "description": (
                    "Compare two strings at the word level and return an inline diff. "
                    "Added words are prefixed with '+', removed words with '-'. "
                    "Returns {inline: str, added_words: int, removed_words: int}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text_a": {"type": "string", "description": "Original text."},
                        "text_b": {"type": "string", "description": "Modified text."},
                    },
                    "required": ["text_a", "text_b"],
                },
            },
            {
                "name": "diff_stats",
                "description": (
                    "Compute similarity and change statistics between two texts. "
                    "Returns {similarity: float (0-1), added_chars: int, removed_chars: int, "
                    "added_lines: int, removed_lines: int, line_similarity: float}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text_a": {"type": "string", "description": "Original text."},
                        "text_b": {"type": "string", "description": "Modified text."},
                    },
                    "required": ["text_a", "text_b"],
                },
            },
            {
                "name": "diff_similar",
                "description": (
                    "Find the most similar strings to a query from a list of candidates. "
                    "Returns [{text, score}, ...] sorted by similarity, highest first. "
                    "Useful for fuzzy matching and near-duplicate detection."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Query string to match against."},
                        "candidates": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of candidate strings.",
                        },
                        "top_n": {
                            "type": "integer",
                            "description": "Return top N results (default 5).",
                        },
                        "threshold": {
                            "type": "number",
                            "description": "Minimum similarity score to include (0-1, default 0.0).",
                        },
                    },
                    "required": ["query", "candidates"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_lines(text: str) -> List[str]:
        """Split text into lines, preserving line endings for difflib."""
        lines = text.splitlines(keepends=True)
        # Ensure last line has a newline for clean diffs
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        return lines

    @staticmethod
    def _count_diff(diff_lines: List[str]) -> Dict[str, int]:
        added = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
        removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
        unchanged = sum(1 for l in diff_lines if l.startswith(" "))
        return {"added_lines": added, "removed_lines": removed, "unchanged_lines": unchanged}

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def diff_text(
        self,
        text_a: str,
        text_b: str,
        context: int = 3,
        label_a: str = "before",
        label_b: str = "after",
    ) -> Dict[str, Any]:
        lines_a = self._split_lines(text_a)
        lines_b = self._split_lines(text_b)

        unified = list(
            difflib.unified_diff(
                lines_a,
                lines_b,
                fromfile=label_a,
                tofile=label_b,
                n=context,
            )
        )

        counts = self._count_diff(unified)
        return {
            "unified": "".join(unified),
            "has_changes": len(unified) > 0,
            **counts,
        }

    def diff_files(
        self,
        path_a: str,
        path_b: str,
        context: int = 3,
    ) -> Dict[str, Any]:
        try:
            text_a = Path(path_a).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": f"Cannot read '{path_a}': {exc}"}
        try:
            text_b = Path(path_b).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return {"error": f"Cannot read '{path_b}': {exc}"}

        return self.diff_text(text_a, text_b, context, label_a=path_a, label_b=path_b)

    def diff_words(self, text_a: str, text_b: str) -> Dict[str, Any]:
        words_a = text_a.split()
        words_b = text_b.split()

        matcher = difflib.SequenceMatcher(None, words_a, words_b)
        inline_parts: List[str] = []
        added = 0
        removed = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                inline_parts.extend(words_a[i1:i2])
            elif tag == "replace":
                for w in words_a[i1:i2]:
                    inline_parts.append(f"-{w}")
                    removed += 1
                for w in words_b[j1:j2]:
                    inline_parts.append(f"+{w}")
                    added += 1
            elif tag == "delete":
                for w in words_a[i1:i2]:
                    inline_parts.append(f"-{w}")
                    removed += 1
            elif tag == "insert":
                for w in words_b[j1:j2]:
                    inline_parts.append(f"+{w}")
                    added += 1

        return {
            "inline": " ".join(inline_parts),
            "added_words": added,
            "removed_words": removed,
        }

    def diff_stats(self, text_a: str, text_b: str) -> Dict[str, Any]:
        # Character similarity
        char_matcher = difflib.SequenceMatcher(None, text_a, text_b)
        char_similarity = round(char_matcher.ratio(), 4)

        # Character-level adds/removes
        added_chars = 0
        removed_chars = 0
        for tag, i1, i2, j1, j2 in char_matcher.get_opcodes():
            if tag in ("replace", "delete"):
                removed_chars += i2 - i1
            if tag in ("replace", "insert"):
                added_chars += j2 - j1

        # Line-level similarity
        lines_a = self._split_lines(text_a)
        lines_b = self._split_lines(text_b)
        line_matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
        line_similarity = round(line_matcher.ratio(), 4)
        counts = self._count_diff(
            list(difflib.unified_diff(lines_a, lines_b, n=0))
        )

        return {
            "similarity": char_similarity,
            "line_similarity": line_similarity,
            "added_chars": added_chars,
            "removed_chars": removed_chars,
            **counts,
        }

    def diff_similar(
        self,
        query: str,
        candidates: List[str],
        top_n: int = 5,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        results = []
        for candidate in candidates:
            score = difflib.SequenceMatcher(None, query, candidate).ratio()
            if score >= threshold:
                results.append({"text": candidate, "score": round(score, 4)})

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_n]

    # ------------------------------------------------------------------
    # execute
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        try:
            if tool_name == "diff_text":
                return json.dumps(
                    self.diff_text(
                        arguments["text_a"],
                        arguments["text_b"],
                        int(arguments.get("context", 3)),
                        arguments.get("label_a", "before"),
                        arguments.get("label_b", "after"),
                    )
                )

            elif tool_name == "diff_files":
                return json.dumps(
                    self.diff_files(
                        arguments["path_a"],
                        arguments["path_b"],
                        int(arguments.get("context", 3)),
                    )
                )

            elif tool_name == "diff_words":
                return json.dumps(
                    self.diff_words(arguments["text_a"], arguments["text_b"])
                )

            elif tool_name == "diff_stats":
                return json.dumps(
                    self.diff_stats(arguments["text_a"], arguments["text_b"])
                )

            elif tool_name == "diff_similar":
                return json.dumps(
                    self.diff_similar(
                        arguments["query"],
                        arguments["candidates"],
                        int(arguments.get("top_n", 5)),
                        float(arguments.get("threshold", 0.0)),
                    )
                )

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except (KeyError, ValueError, TypeError, OSError) as exc:
            return json.dumps({"error": str(exc)})
