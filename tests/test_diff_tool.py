"""Tests for DiffTool."""

import json
import os
import tempfile

import pytest

from agent_friend.tools.diff import DiffTool


@pytest.fixture()
def tool():
    return DiffTool()


# ---------------------------------------------------------------------------
# BaseTool contract
# ---------------------------------------------------------------------------

class TestBaseContract:
    def test_name(self, tool):
        assert tool.name == "diff"

    def test_description(self, tool):
        assert len(tool.description) > 10

    def test_definitions(self, tool):
        defs = tool.definitions()
        assert isinstance(defs, list)
        assert len(defs) >= 5

    def test_definitions_keys(self, tool):
        for d in tool.definitions():
            assert "name" in d
            assert "description" in d
            assert "input_schema" in d


# ---------------------------------------------------------------------------
# diff_text
# ---------------------------------------------------------------------------

class TestDiffText:
    def test_identical(self, tool):
        result = tool.diff_text("hello", "hello")
        assert result["has_changes"] is False
        assert result["added_lines"] == 0
        assert result["removed_lines"] == 0

    def test_changed_line(self, tool):
        result = tool.diff_text("hello world\n", "hello there\n")
        assert result["has_changes"] is True
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 1
        assert "-hello world" in result["unified"]
        assert "+hello there" in result["unified"]

    def test_added_line(self, tool):
        result = tool.diff_text("line1\n", "line1\nline2\n")
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 0

    def test_removed_line(self, tool):
        result = tool.diff_text("line1\nline2\n", "line1\n")
        assert result["removed_lines"] == 1
        assert result["added_lines"] == 0

    def test_empty_to_content(self, tool):
        result = tool.diff_text("", "hello\n")
        assert result["has_changes"] is True
        assert result["added_lines"] == 1

    def test_content_to_empty(self, tool):
        result = tool.diff_text("hello\n", "")
        assert result["removed_lines"] == 1

    def test_custom_context(self, tool):
        # 5 lines, change middle
        a = "\n".join(["a", "b", "c", "d", "e"]) + "\n"
        b = "\n".join(["a", "b", "X", "d", "e"]) + "\n"
        result0 = tool.diff_text(a, b, context=0)
        result3 = tool.diff_text(a, b, context=3)
        # context=0 shows fewer lines than context=3
        assert len(result0["unified"]) <= len(result3["unified"])

    def test_custom_labels(self, tool):
        result = tool.diff_text("a\n", "b\n", label_a="old.py", label_b="new.py")
        assert "old.py" in result["unified"]
        assert "new.py" in result["unified"]

    def test_multiline(self, tool):
        a = "line1\nline2\nline3\n"
        b = "line1\nmodified\nline3\n"
        result = tool.diff_text(a, b)
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 1
        assert result["unchanged_lines"] >= 1


# ---------------------------------------------------------------------------
# diff_files
# ---------------------------------------------------------------------------

class TestDiffFiles:
    def test_diff_files_identical(self, tool, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content\n")
        result = tool.diff_files(str(f), str(f))
        assert result["has_changes"] is False

    def test_diff_files_changed(self, tool, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("old content\n")
        b.write_text("new content\n")
        result = tool.diff_files(str(a), str(b))
        assert result["has_changes"] is True
        assert result["added_lines"] == 1
        assert result["removed_lines"] == 1

    def test_diff_files_missing(self, tool):
        result = tool.diff_files("/nonexistent/file.txt", "/also/missing.txt")
        assert "error" in result


# ---------------------------------------------------------------------------
# diff_words
# ---------------------------------------------------------------------------

class TestDiffWords:
    def test_identical(self, tool):
        result = tool.diff_words("hello world", "hello world")
        assert result["added_words"] == 0
        assert result["removed_words"] == 0
        assert "hello" in result["inline"]

    def test_changed_word(self, tool):
        result = tool.diff_words("hello world", "hello there")
        assert result["added_words"] == 1
        assert result["removed_words"] == 1
        assert "+there" in result["inline"]
        assert "-world" in result["inline"]

    def test_added_word(self, tool):
        result = tool.diff_words("hello", "hello world")
        assert result["added_words"] == 1
        assert result["removed_words"] == 0

    def test_removed_word(self, tool):
        result = tool.diff_words("hello world", "hello")
        assert result["removed_words"] == 1
        assert result["added_words"] == 0

    def test_empty_inputs(self, tool):
        result = tool.diff_words("", "")
        assert result["added_words"] == 0
        assert result["removed_words"] == 0


# ---------------------------------------------------------------------------
# diff_stats
# ---------------------------------------------------------------------------

class TestDiffStats:
    def test_identical_similarity(self, tool):
        result = tool.diff_stats("hello world", "hello world")
        assert result["similarity"] == 1.0

    def test_completely_different(self, tool):
        result = tool.diff_stats("abc", "xyz")
        assert result["similarity"] < 0.5

    def test_similar_texts(self, tool):
        result = tool.diff_stats("hello world", "hello there")
        assert 0.5 < result["similarity"] < 1.0

    def test_added_chars(self, tool):
        result = tool.diff_stats("hi", "hi there")
        assert result["added_chars"] > 0

    def test_removed_chars(self, tool):
        result = tool.diff_stats("hi there", "hi")
        assert result["removed_chars"] > 0

    def test_returns_all_fields(self, tool):
        result = tool.diff_stats("a\nb\n", "a\nc\n")
        assert "similarity" in result
        assert "line_similarity" in result
        assert "added_chars" in result
        assert "removed_chars" in result
        assert "added_lines" in result
        assert "removed_lines" in result

    def test_empty_strings(self, tool):
        result = tool.diff_stats("", "")
        assert result["similarity"] == 1.0


# ---------------------------------------------------------------------------
# diff_similar
# ---------------------------------------------------------------------------

class TestDiffSimilar:
    def test_finds_closest(self, tool):
        results = tool.diff_similar("hello", ["hello", "world", "help"])
        assert results[0]["text"] == "hello"
        assert results[0]["score"] == 1.0

    def test_returns_sorted(self, tool):
        results = tool.diff_similar("abc", ["abd", "xyz", "abc"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n(self, tool):
        candidates = [f"word{i}" for i in range(20)]
        results = tool.diff_similar("word0", candidates, top_n=3)
        assert len(results) <= 3

    def test_threshold(self, tool):
        results = tool.diff_similar("hello", ["hello", "xyz"], threshold=0.9)
        assert all(r["score"] >= 0.9 for r in results)

    def test_empty_candidates(self, tool):
        results = tool.diff_similar("hello", [])
        assert results == []


# ---------------------------------------------------------------------------
# execute dispatch
# ---------------------------------------------------------------------------

class TestExecuteDispatch:
    def test_execute_diff_text(self, tool):
        out = json.loads(tool.execute("diff_text", {
            "text_a": "hello\n", "text_b": "world\n"
        }))
        assert "unified" in out
        assert "has_changes" in out

    def test_execute_diff_files_missing(self, tool):
        out = json.loads(tool.execute("diff_files", {
            "path_a": "/no/file.txt", "path_b": "/no/other.txt"
        }))
        assert "error" in out

    def test_execute_diff_words(self, tool):
        out = json.loads(tool.execute("diff_words", {
            "text_a": "foo bar", "text_b": "foo baz"
        }))
        assert "inline" in out

    def test_execute_diff_stats(self, tool):
        out = json.loads(tool.execute("diff_stats", {
            "text_a": "hello", "text_b": "hello world"
        }))
        assert "similarity" in out

    def test_execute_diff_similar(self, tool):
        out = json.loads(tool.execute("diff_similar", {
            "query": "foo", "candidates": ["foo", "bar"]
        }))
        assert isinstance(out, list)

    def test_execute_unknown(self, tool):
        out = json.loads(tool.execute("unknown_fn", {}))
        assert "error" in out
