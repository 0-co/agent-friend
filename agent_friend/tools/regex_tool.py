"""regex_tool.py — RegexTool for agent-friend (stdlib only).

Regular expressions are the universal text-processing primitive.  AgentTool
wraps Python's ``re`` module in an ergonomic interface that agents can use
to match, search, extract, replace, split, and validate text — without having
to know the exact ``re`` API or worry about flags and compilation.

Usage::

    tool = RegexTool()

    tool.regex_search(r"\\d+\\.\\d+", "Version 0.27.0 released")
    # {"matched": true, "match": "0.27", "start": 8, "end": 12, "groups": []}

    tool.regex_findall(r"\\w+@\\w+\\.\\w+", "alice@example.com, bob@test.org")
    # ["alice@example.com", "bob@test.org"]

    tool.regex_replace(r"\\s+", " ", "too   many    spaces")
    # "too many spaces"

    tool.regex_validate(r"(\\d+")   # bad pattern
    # {"valid": false, "error": "missing ), unterminated subpattern ..."}
"""

import json
import re
from typing import Any, Dict, List, Optional

from .base import BaseTool


_FLAGS: Dict[str, int] = {
    "IGNORECASE": re.IGNORECASE,
    "I": re.IGNORECASE,
    "MULTILINE": re.MULTILINE,
    "M": re.MULTILINE,
    "DOTALL": re.DOTALL,
    "S": re.DOTALL,
    "VERBOSE": re.VERBOSE,
    "X": re.VERBOSE,
}


def _parse_flags(flag_names: Optional[List[str]]) -> int:
    """Return combined re flags from a list of flag name strings."""
    if not flag_names:
        return 0
    total = 0
    for name in flag_names:
        f = _FLAGS.get(name.upper())
        if f is not None:
            total |= f
    return total


class RegexTool(BaseTool):
    """Regular-expression operations on text: match, search, findall, replace, split.

    All operations accept an optional ``flags`` list (e.g. ``["IGNORECASE"]``)
    and have a configurable ``max_results`` limit.

    Parameters
    ----------
    max_results:
        Maximum number of matches/groups returned per call (default 1000).
    max_text:
        Maximum input text length accepted (default 1 MB).
    """

    def __init__(self, max_results: int = 1000, max_text: int = 1_000_000) -> None:
        self.max_results = max_results
        self.max_text = max_text

    # ── helpers ───────────────────────────────────────────────────────────

    def _compile(self, pattern: str, flags: Optional[List[str]]) -> re.Pattern:
        return re.compile(pattern, _parse_flags(flags))

    def _match_to_dict(self, m: re.Match) -> Dict[str, Any]:
        return {
            "match": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "groups": list(m.groups()),
            "named_groups": m.groupdict(),
        }

    # ── public API ────────────────────────────────────────────────────────

    def regex_match(
        self,
        pattern: str,
        text: str,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Match *pattern* at the **start** of *text*.

        Returns ``{"matched": true/false, "match": "...", "start": 0, ...}``.
        """
        try:
            rx = self._compile(pattern, flags)
            m = rx.match(text[: self.max_text])
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        if m is None:
            return json.dumps({"matched": False})
        result = self._match_to_dict(m)
        result["matched"] = True
        return json.dumps(result)

    def regex_search(
        self,
        pattern: str,
        text: str,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Search anywhere in *text* for the first occurrence of *pattern*.

        Returns ``{"matched": true/false, "match": "...", "start": N, ...}``.
        """
        try:
            rx = self._compile(pattern, flags)
            m = rx.search(text[: self.max_text])
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        if m is None:
            return json.dumps({"matched": False})
        result = self._match_to_dict(m)
        result["matched"] = True
        return json.dumps(result)

    def regex_findall(
        self,
        pattern: str,
        text: str,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Find all non-overlapping matches of *pattern* in *text*.

        Returns a JSON list.  If the pattern has groups, returns list of
        group tuples; if one group, list of strings.
        """
        try:
            rx = self._compile(pattern, flags)
            matches = rx.findall(text[: self.max_text])
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        return json.dumps(matches[: self.max_results])

    def regex_findall_with_positions(
        self,
        pattern: str,
        text: str,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Find all matches with their positions.

        Returns list of ``{match, start, end, groups, named_groups}``.
        """
        try:
            rx = self._compile(pattern, flags)
            it = rx.finditer(text[: self.max_text])
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        results = []
        for m in it:
            results.append(self._match_to_dict(m))
            if len(results) >= self.max_results:
                break
        return json.dumps(results)

    def regex_replace(
        self,
        pattern: str,
        replacement: str,
        text: str,
        count: int = 0,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Replace occurrences of *pattern* in *text* with *replacement*.

        *count* = 0 replaces all.  *replacement* supports backreferences
        like ``\\1`` or ``\\g<name>``.  Returns the modified string.
        """
        try:
            rx = self._compile(pattern, flags)
            result = rx.sub(replacement, text[: self.max_text], count=count)
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        return result

    def regex_split(
        self,
        pattern: str,
        text: str,
        maxsplit: int = 0,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Split *text* by *pattern*.  Returns a JSON list of strings."""
        try:
            rx = self._compile(pattern, flags)
            parts = rx.split(text[: self.max_text], maxsplit=maxsplit)
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        return json.dumps(parts[: self.max_results])

    def regex_extract_groups(
        self,
        pattern: str,
        text: str,
        flags: Optional[List[str]] = None,
    ) -> str:
        """Extract all captured groups from all matches.

        Returns list of dicts: ``{match, groups, named_groups}`` for each match.
        Useful when the pattern defines multiple capture groups.
        """
        try:
            rx = self._compile(pattern, flags)
            results = []
            for m in rx.finditer(text[: self.max_text]):
                results.append({
                    "match": m.group(0),
                    "groups": list(m.groups()),
                    "named_groups": m.groupdict(),
                })
                if len(results) >= self.max_results:
                    break
        except re.error as exc:
            return json.dumps({"error": f"Pattern error: {exc}"})

        return json.dumps(results)

    def regex_validate(self, pattern: str) -> str:
        """Check whether *pattern* is a valid regular expression.

        Returns ``{"valid": true}`` or ``{"valid": false, "error": "..."}``.
        """
        try:
            re.compile(pattern)
            return json.dumps({"valid": True})
        except re.error as exc:
            return json.dumps({"valid": False, "error": str(exc)})

    def regex_escape(self, text: str) -> str:
        """Escape *text* so it matches literally in a regex pattern.

        Useful when you want to build a pattern that includes user-supplied text.
        """
        return re.escape(text)

    # ── BaseTool interface ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "regex"

    @property
    def description(self) -> str:
        return (
            "Regular expression operations: match, search, findall, replace, split, "
            "and group extraction.  Supports IGNORECASE, MULTILINE, DOTALL flags. "
            "All stdlib — no external dependencies."
        )

    def definitions(self) -> List[Dict[str, Any]]:
        flags_schema = {
            "type": "array",
            "items": {"type": "string"},
            "description": "Regex flags: IGNORECASE, MULTILINE, DOTALL",
        }
        return [
            {
                "name": "regex_match",
                "description": (
                    "Match a pattern at the START of text. "
                    "Returns {matched, match, start, end, groups, named_groups}. "
                    "Use regex_search for matching anywhere in text."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "Regular expression pattern"},
                        "text": {"type": "string", "description": "Text to match against"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_search",
                "description": (
                    "Search anywhere in text for the first match. "
                    "Returns {matched, match, start, end, groups, named_groups}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "text": {"type": "string"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_findall",
                "description": (
                    "Return all non-overlapping matches as a list. "
                    "If the pattern has groups, returns list of group strings. "
                    "Simple and fast — use regex_findall_with_positions for positions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "text": {"type": "string"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_findall_with_positions",
                "description": (
                    "Return all matches with their start/end positions and groups. "
                    "Returns list of {match, start, end, groups, named_groups}."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "text": {"type": "string"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_replace",
                "description": (
                    "Replace occurrences of pattern in text with replacement string. "
                    "count=0 (default) replaces all. "
                    "Replacement supports backreferences: \\\\1, \\\\g<name>."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "replacement": {"type": "string", "description": "Replacement string"},
                        "text": {"type": "string"},
                        "count": {"type": "integer", "description": "Max replacements (0 = all)"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "replacement", "text"],
                },
            },
            {
                "name": "regex_split",
                "description": "Split text by a regex pattern. Returns a list of strings.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "text": {"type": "string"},
                        "maxsplit": {"type": "integer", "description": "Max splits (0 = all)"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_extract_groups",
                "description": (
                    "Extract captured groups from all matches. "
                    "Returns list of {match, groups, named_groups} for each match. "
                    "Use when the pattern defines named or numbered groups."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "text": {"type": "string"},
                        "flags": flags_schema,
                    },
                    "required": ["pattern", "text"],
                },
            },
            {
                "name": "regex_validate",
                "description": "Check if a pattern is a valid regular expression. Returns {valid: true/false}.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            },
            {
                "name": "regex_escape",
                "description": (
                    "Escape special regex characters in a string so it matches literally. "
                    "Use when building patterns from user-supplied text."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                },
            },
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if tool_name == "regex_match":
            return self.regex_match(**arguments)
        if tool_name == "regex_search":
            return self.regex_search(**arguments)
        if tool_name == "regex_findall":
            return self.regex_findall(**arguments)
        if tool_name == "regex_findall_with_positions":
            return self.regex_findall_with_positions(**arguments)
        if tool_name == "regex_replace":
            return self.regex_replace(**arguments)
        if tool_name == "regex_split":
            return self.regex_split(**arguments)
        if tool_name == "regex_extract_groups":
            return self.regex_extract_groups(**arguments)
        if tool_name == "regex_validate":
            return self.regex_validate(**arguments)
        if tool_name == "regex_escape":
            return self.regex_escape(**arguments)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
