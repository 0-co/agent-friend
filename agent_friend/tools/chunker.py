"""chunker.py — ChunkerTool for agent-friend (stdlib only).

Split long text or lists into manageable chunks for LLM context windows.
Useful for RAG pipelines, document ingestion, and batch processing.

Features:
* chunk_text — split text by token estimate, character count, or sentence/paragraph
* chunk_list — split a list into batches
* chunk_by_separator — split on a custom delimiter with optional overlap
* chunk_sliding_window — sliding window over tokens/sentences
* chunk_stats — inspect a text before chunking

Usage::

    tool = ChunkerTool()

    chunks = tool.chunk_text("very long document ...", max_chars=1000, overlap=100)
    # [{"index": 0, "text": "...", "char_count": 998}, ...]

    batches = tool.chunk_list([1, 2, 3, ..., 100], size=25)
    # [{"index": 0, "items": [1..25], "count": 25}, ...]
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base import BaseTool


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT/Claude heuristic)."""
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences on . ! ? followed by whitespace."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p]


def _split_paragraphs(text: str) -> List[str]:
    """Split on blank lines."""
    parts = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in parts if p.strip()]


class ChunkerTool(BaseTool):
    """Split text and lists into chunks for LLM context management.

    Supports character-based, token-estimated, sentence, paragraph,
    and custom-separator chunking. All methods return JSON arrays of
    chunk objects with index and metadata.
    """

    # ── public API ────────────────────────────────────────────────────

    def chunk_text(
        self,
        text: str,
        max_chars: int = 2000,
        overlap: int = 0,
        mode: str = "chars",
    ) -> str:
        """Split text into chunks.

        *mode* options:

        * ``chars`` — hard split every *max_chars* characters
        * ``tokens`` — split on estimated token boundary (~4 chars each)
        * ``sentences`` — split on sentence boundaries (. ! ?)
        * ``paragraphs`` — split on blank lines

        *overlap* adds trailing characters from the previous chunk to the
        start of the next (for ``chars`` and ``tokens`` modes).

        Returns a JSON array of ``{index, text, char_count, token_estimate}``.
        """
        if not text:
            return json.dumps([])

        if mode == "chars":
            chunks = self._chunk_by_chars(text, max_chars, overlap)
        elif mode == "tokens":
            # Convert token limit to char limit
            max_c = max_chars * 4  # max_chars treated as max_tokens here
            overlap_c = overlap * 4
            chunks = self._chunk_by_chars(text, max_c, overlap_c)
        elif mode == "sentences":
            chunks = self._chunk_by_units(_split_sentences(text), max_chars, overlap)
        elif mode == "paragraphs":
            chunks = self._chunk_by_units(_split_paragraphs(text), max_chars, overlap)
        else:
            return json.dumps({"error": f"Unknown mode '{mode}'. Valid: chars, tokens, sentences, paragraphs."})

        result = []
        for i, c in enumerate(chunks):
            result.append({
                "index": i,
                "text": c,
                "char_count": len(c),
                "token_estimate": _estimate_tokens(c),
            })
        return json.dumps(result)

    def chunk_list(self, items: List[Any], size: int = 10) -> str:
        """Split a list into batches of *size*.

        Returns a JSON array of ``{index, items, count}``.
        """
        if size <= 0:
            return json.dumps({"error": "size must be > 0"})
        if not items:
            return json.dumps([])

        result = []
        for i in range(0, len(items), size):
            batch = items[i: i + size]
            result.append({"index": i // size, "items": batch, "count": len(batch)})
        return json.dumps(result)

    def chunk_by_separator(
        self,
        text: str,
        separator: str,
        max_chars: int = 0,
        keep_separator: bool = False,
    ) -> str:
        """Split text on a custom *separator* string.

        If *max_chars* > 0, adjacent parts are merged until the limit.
        If *keep_separator* is true, the separator is re-appended to each part.

        Returns a JSON array of ``{index, text, char_count}``.
        """
        if not separator:
            return json.dumps({"error": "separator must be non-empty"})

        parts = text.split(separator)
        if keep_separator:
            parts = [p + separator for p in parts[:-1]] + parts[-1:]

        parts = [p for p in parts if p]

        if max_chars > 0:
            merged: List[str] = []
            current = ""
            for p in parts:
                if current and len(current) + len(p) > max_chars:
                    merged.append(current)
                    current = p
                else:
                    current = current + p if not current else current + p
            if current:
                merged.append(current)
            parts = merged

        return json.dumps([
            {"index": i, "text": p, "char_count": len(p)}
            for i, p in enumerate(parts)
        ])

    def chunk_sliding_window(
        self,
        text: str,
        window_chars: int = 500,
        step_chars: int = 250,
    ) -> str:
        """Sliding window over text.

        Each chunk is *window_chars* characters. The window advances by
        *step_chars* each time, so consecutive chunks overlap by
        ``window_chars - step_chars`` characters.

        Returns a JSON array of ``{index, text, start, end, char_count}``.
        """
        if window_chars <= 0 or step_chars <= 0:
            return json.dumps({"error": "window_chars and step_chars must be > 0"})
        if not text:
            return json.dumps([])

        result = []
        i = 0
        idx = 0
        while i < len(text):
            chunk = text[i: i + window_chars]
            result.append({
                "index": idx,
                "text": chunk,
                "start": i,
                "end": i + len(chunk),
                "char_count": len(chunk),
            })
            i += step_chars
            idx += 1
            if i >= len(text):
                break
        return json.dumps(result)

    def chunk_stats(self, text: str) -> str:
        """Return statistics about a text to help choose chunking params.

        Returns ``{char_count, token_estimate, line_count, sentence_count,
        paragraph_count, word_count}``.
        """
        if not text:
            return json.dumps({
                "char_count": 0, "token_estimate": 0, "line_count": 0,
                "sentence_count": 0, "paragraph_count": 0, "word_count": 0,
            })

        return json.dumps({
            "char_count": len(text),
            "token_estimate": _estimate_tokens(text),
            "line_count": text.count("\n") + 1,
            "word_count": len(text.split()),
            "sentence_count": len(_split_sentences(text)),
            "paragraph_count": len(_split_paragraphs(text)),
        })

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _chunk_by_chars(text: str, max_chars: int, overlap: int) -> List[str]:
        if max_chars <= 0:
            return [text]
        overlap = min(overlap, max_chars - 1)
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end - overlap if overlap > 0 else end
        return chunks

    @staticmethod
    def _chunk_by_units(units: List[str], max_chars: int, overlap_chars: int) -> List[str]:
        """Merge units (sentences/paragraphs) until max_chars, then flush."""
        chunks: List[str] = []
        current_parts: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            if current_parts and current_len + unit_len + 1 > max_chars:
                chunk = " ".join(current_parts)
                chunks.append(chunk)
                # overlap: keep tail units that fit within overlap_chars
                if overlap_chars > 0:
                    tail: List[str] = []
                    tail_len = 0
                    for part in reversed(current_parts):
                        if tail_len + len(part) + 1 <= overlap_chars:
                            tail.insert(0, part)
                            tail_len += len(part) + 1
                        else:
                            break
                    current_parts = tail
                    current_len = tail_len
                else:
                    current_parts = []
                    current_len = 0
            current_parts.append(unit)
            current_len += unit_len + 1

        if current_parts:
            chunks.append(" ".join(current_parts))
        return chunks

    # ── BaseTool interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "chunker"

    @property
    def description(self) -> str:
        return (
            "Split long text or lists into chunks for LLM context windows. "
            "Modes: chars, tokens, sentences, paragraphs, sliding window, "
            "custom separator. Overlap support. Stats inspection. Zero deps."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "chunk_text",
                "description": "Split text into chunks. mode: chars/tokens/sentences/paragraphs. Returns [{index, text, char_count, token_estimate}].",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "max_chars": {"type": "integer", "description": "Max chars per chunk (or max tokens if mode=tokens)"},
                        "overlap": {"type": "integer", "description": "Overlap chars between chunks"},
                        "mode": {"type": "string", "description": "chars | tokens | sentences | paragraphs"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "chunk_list",
                "description": "Split a list into batches of size. Returns [{index, items, count}].",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array"},
                        "size": {"type": "integer"},
                    },
                    "required": ["items"],
                },
            },
            {
                "name": "chunk_by_separator",
                "description": "Split text on a custom separator. Optionally merge parts up to max_chars.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "separator": {"type": "string"},
                        "max_chars": {"type": "integer"},
                        "keep_separator": {"type": "boolean"},
                    },
                    "required": ["text", "separator"],
                },
            },
            {
                "name": "chunk_sliding_window",
                "description": "Sliding window over text. Returns [{index, text, start, end}].",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "window_chars": {"type": "integer"},
                        "step_chars": {"type": "integer"},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "chunk_stats",
                "description": "Return stats about text: char_count, token_estimate, line_count, sentence_count, paragraph_count, word_count.",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "chunk_text":
            return self.chunk_text(**arguments)
        if tool_name == "chunk_list":
            return self.chunk_list(**arguments)
        if tool_name == "chunk_by_separator":
            return self.chunk_by_separator(**arguments)
        if tool_name == "chunk_sliding_window":
            return self.chunk_sliding_window(**arguments)
        if tool_name == "chunk_stats":
            return self.chunk_stats(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
